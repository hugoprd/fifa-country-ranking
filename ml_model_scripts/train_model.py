import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger

import tqdm
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model_extensions import SynergyAttention, FIFANationalTeamDataset

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "model_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=False)

logger.info("=" * 32)
logger.info("train_model.py LOG INITIALIZED.")

REFINED_DATA_PATH = ROOT_DIR / "data/refined"
REFINED_DATA_PATH.mkdir(parents=True, exist_ok=True)


class NationalTeamTransformer(nn.Module):
    """
    The complete model: From individual player statistics to national team ranking.
    """

    def __init__(self, num_features, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        # 1. input Embedding: transforms the player numeric stats into a Dense Vector
        self.player_embedding = nn.Linear(num_features, embed_dim)

        # 2. transformer layers with the Synergy Attention
        self.layers = nn.ModuleList([SynergyAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
                for _ in range(num_layers)
            ]
        )
        # Pre-LN: normalization before each sub-layer (attn/FF), stabilizes the residual stream.
        # a "Transformer" without LayerNorm is the most likely cause of the loss spikes in the log
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # 3. prediction Head (Regression for predicting FIFA Points/Ranking)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)  # returns a single number (the team's points)
        )

    def forward(self, players, synergy_matrix, key_padding_mask=None):
        """
        players: Tensor of shape (Batch_Size, 11, Num_Features) -> The 11 starters
        synergy_matrix: Tensor of shape (Batch_Size, 11, 11) -> The chemistry between the 11
        key_padding_mask: Bool tensor (Batch_Size, 11), True = slot is padding (country had < 11 players).
        """
        # pass the player attributes to create the initial token
        x = self.player_embedding(players)

        # pass through the Transformer (Pre-LN: normalize before attn/FF, then residual add)
        for attn, ff, attn_norm, ff_norm in zip(self.layers, self.feed_forwards, self.attn_norms, self.ff_norms):
            attended = attn(attn_norm(x), synergy_mask=synergy_matrix, key_padding_mask=key_padding_mask)
            x = x + attended  # Conexão Residual

            forwarded = ff(ff_norm(x))
            x = x + forwarded

        # Masked Average Pooling: excludes padded ("ghost") player slots from the team average,
        # instead of plain x.mean(dim=1), which would dilute real players with padding zeros
        # for any country fielding fewer than 11 known players.
        if key_padding_mask is not None:
            real_player_mask = (~key_padding_mask).unsqueeze(-1).float()  # (Batch, 11, 1)
            summed = (x * real_player_mask).sum(dim=1)
            count = real_player_mask.sum(dim=1).clamp(min=1.0)
            team_representation = summed / count
        else:
            team_representation = x.mean(dim=1)

        # calculates the final strength/ranking
        ranking_points = self.regressor(team_representation)

        return ranking_points


def _verify_model_parameters() -> tuple[bool, dict | list | None]:
    """
    Verifies if the JSON transformers_architecture file exists, is not empty, and contains useful data.
    """
    logger.info("[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] Verifying the transformers_architecture.json file...")

    json_path = ROOT_DIR / "ml_model_scripts/transformers_architecture.json"

    if not json_path.is_file():
        logger.error("[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] The transformers_architecture.json file does not exist.")

        return False, None

    # 1. fisical verification: the file exists and has size greater than 0 bytes
    if not json_path.exists() or json_path.stat().st_size == 0:
        logger.error("[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] The file does not exist or is empty.")

        return False, None

    # 2. logic verification: JSON is valid and contains keys/items?
    try:
        with open(json_path, "r", encoding="utf-8") as arquivo:
            data = json.load(arquivo)

            if data:
                logger.info(f"[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] Success. The file contains {len(data)} items/keys.")

                return True, data
            else:
                logger.error(
                    "[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] The file is a valid JSON, but is empty (e.g., {} or [])."
                )

                return False, None
    except json.JSONDecodeError:
        logger.error(
            "[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] The file it's not empty, "
            "but the JSON format is invalid or is corrupted."
        )

        return False, None


def train_model():
    """
    Starts the training of the Transformer model using the data from the refine/ folder.
    """
    logger.info("[ TRAIN MODEL ] Loading the real data and starting training...")

    ind_path = REFINED_DATA_PATH / "ml_individual_features.csv"
    pairs_path = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    target_path = REFINED_DATA_PATH / "ml_national_team_ranking.csv"

    if not all(p.exists() for p in [ind_path, pairs_path, target_path]):
        logger.error("[ TRAIN MODEL ] CSV files not found at the refined folder!")

        return

    #### ==========================================
    # 1. create the Dataset and DataLoader
    #### ==========================================

    # Batch size = 4 ; means the model looks at 4 countries at the same time before updating the weights
    dataset = FIFANationalTeamDataset(ind_path, pairs_path, target_path, top_k_players=11)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    logger.success(f"[ TRAIN MODEL ] Dataset loaded. Total national teams: {len(dataset)}")

    #### ==========================================
    # 2. instantiate the Model
    #### ==========================================

    # verifies the transformers_architecture.json file and loads the parameters
    valid, architecture_data = _verify_model_parameters()

    if not valid or architecture_data is None:
        logger.error("[ TRAIN MODEL ] Invalid or missing transformers_architecture.json file. Cannot proceed with training.")

        return

    # defines de initial model
    initial_model = NationalTeamTransformer(
        num_features=4,  # 4: total_matches, total_weighted_wins, total_weighted_plus_minus, win_rate_percentage
        embed_dim=architecture_data["embed_dim"],
        num_heads=architecture_data["num_heads"],
        num_layers=architecture_data["num_layers"],
    )

    actual_params = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
    ideal_params = architecture_data.get("trainable_params")

    logger.info(f"[ TRAIN MODEL ] Actual Trainable Parameters in PyTorch: {actual_params}")

    if ideal_params:
        if actual_params == ideal_params:
            logger.success("[ TRAIN MODEL ] Parameter count matches the ideal architecture exactly!")
        else:
            logger.warning(f"[ TRAIN MODEL ] Parameter mismatch. Ideal: {ideal_params} | PyTorch Actual: {actual_params}")

    #### ==========================================
    # 3. hyperparameter test loop
    #### ==========================================
    epochs = 200  # base number

    configs_to_test = [
        {
            "embed_dim": architecture_data["embed_dim"],
            "num_heads": architecture_data["num_heads"],
            "num_layers": architecture_data["num_layers"],
        },
        # -- 32 DIMENSIONS --
        {"embed_dim": 32, "num_heads": 1, "num_layers": 1},
        {"embed_dim": 32, "num_heads": 1, "num_layers": 2},
        {"embed_dim": 32, "num_heads": 2, "num_layers": 1},
        {"embed_dim": 32, "num_heads": 2, "num_layers": 2},
        {"embed_dim": 32, "num_heads": 4, "num_layers": 1},
        {"embed_dim": 32, "num_heads": 4, "num_layers": 2},
        {"embed_dim": 32, "num_heads": 4, "num_layers": 3},
        {"embed_dim": 32, "num_heads": 8, "num_layers": 1},
        {"embed_dim": 32, "num_heads": 8, "num_layers": 2},
        # -- 64 DIMENSIONS --
        {"embed_dim": 64, "num_heads": 1, "num_layers": 1},
        {"embed_dim": 64, "num_heads": 1, "num_layers": 2},
        {"embed_dim": 64, "num_heads": 2, "num_layers": 1},
        {"embed_dim": 64, "num_heads": 2, "num_layers": 2},
        {"embed_dim": 64, "num_heads": 4, "num_layers": 1},
        {"embed_dim": 64, "num_heads": 4, "num_layers": 2},
        {"embed_dim": 64, "num_heads": 4, "num_layers": 3},
        {"embed_dim": 64, "num_heads": 8, "num_layers": 1},
        {"embed_dim": 64, "num_heads": 8, "num_layers": 2},
        # -- 128 DIMENSIONS --
        {"embed_dim": 128, "num_heads": 1, "num_layers": 1},
        {"embed_dim": 128, "num_heads": 2, "num_layers": 2},
        {"embed_dim": 128, "num_heads": 4, "num_layers": 2},
        {"embed_dim": 128, "num_heads": 4, "num_layers": 3},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 2},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 3},
    ]

    best_loss = float("inf")  # starts with a infinit error
    best_config = None

    logger.info("[ TRAIN MODEL ] Hyperparameter test started.")
    for i, config in enumerate(configs_to_test):
        logger.info(f"\n[ TRAIN MODEL ] Starting Session {i + 1}/{len(configs_to_test)} | Config: {config}")

        actual_model = NationalTeamTransformer(
            num_features=4,  # 4: total_matches, total_weighted_wins, total_weighted_plus_minus, win_rate_percentage
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(actual_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        actual_model.train()

        for epoch in tqdm.tqdm(range(epochs), desc="Hyperparameter Test Epochs", unit="epoch", colour="white", leave=False):
            epoch_loss = 0.0

            for players, synergy_mask, padding_mask, targets in dataloader:
                optimizer.zero_grad()
                predictions = actual_model(players, synergy_mask, key_padding_mask=padding_mask)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            scheduler.step()

        logger.info(f"[ TRAIN MODEL ] Hyperparameter Test Session {epoch+1}/{epochs} Completed. Final Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            logger.success(f"[ TRAIN MODEL ] New Best Model found. Loss dropped from {best_loss:.4f} to {avg_loss:.4f}")

            best_loss = avg_loss
            best_config = config

    logger.info("=" * 32)
    logger.success(f"[ TRAIN MODEL ] Best Configuration : {best_config}\nLowest Error (Loss): {best_loss:.4f}")

    new_data = {
        "num_features": 4,
        "embed_dim": best_config["embed_dim"],
        "num_heads": best_config["num_heads"],
        "num_layers": best_config["num_layers"],
        "trainable_params": actual_params,
        "avg_val_mse": best_loss,
    }

    json_path = ROOT_DIR / "ml_model_scripts/transformers_architecture.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

    logger.success(f"[ TRAIN MODEL ] {json_path.name} successfully updated with the best hyperparameters.")

    final_model = NationalTeamTransformer(
        num_features=4,  # 4: total_matches, total_weighted_wins, total_weighted_plus_minus, win_rate_percentage
        embed_dim=best_config["embed_dim"],
        num_heads=best_config["num_heads"],
        num_layers=best_config["num_layers"],
    )

    logger.info("[ TRAIN MODEL ] Starting final model train with the best configuration...")

    final_epochs = 300
    final_model.train()

    best_final_loss = float("inf")
    best_final_weights = None
    best_epoch = 0

    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=final_epochs)

    for epoch in tqdm.tqdm(range(final_epochs), desc="Final Train Epochs", unit="epoch", colour="white"):
        epoch_loss = 0.0

        for players, synergy_mask, padding_mask, targets in dataloader:
            final_optimizer.zero_grad()
            predictions = actual_model(players, synergy_mask, key_padding_mask=padding_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            final_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        final_scheduler.step()

        if avg_loss < best_final_loss:
            best_final_loss = avg_loss
            best_epoch = epoch + 1  # noqa: F841
            best_final_weights = copy.deepcopy(final_model.state_dict())

        logger.info(f"[ TRAIN MODEL ] Final Train Session {epoch+1}/{final_epochs} Completed. Current Loss: {avg_loss:.4f}")

    final_model.load_state_dict(best_final_weights)

    ideal_mse = architecture_data.get("avg_val_mse")

    if ideal_mse:
        logger.info("=" * 32)
        logger.info("[ TRAIN MODEL ] Comparing the best final training loss with the ideal validation MSE...")
        logger.info(f"[ TRAIN MODEL ] Target Ideal MSE from Optimization : {ideal_mse:.4f}")
        logger.info(f"[ TRAIN MODEL ] Best Training Loss Obtained        : {best_final_loss:.4f}")

        if best_final_loss <= ideal_mse:
            logger.success("[ TRAIN MODEL ] The model successfully reached or surpassed the ideal validation target!")
        else:
            logger.warning(
                "[ TRAIN MODEL ] The model did not reach the ideal target. Consider adjusting epochs or learning rate."
            )

    model_path = model_save_path = ROOT_DIR / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = model_path / "best_transformer_model.pth"
    torch.save(final_model.state_dict(), model_save_path)
    logger.success(f"[ TRAIN MODEL ] Model weights successfully saved at: {model_save_path}")


if __name__ == "__main__":
    train_model()
