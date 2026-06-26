import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger

import tqdm
import json
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score

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

    def __init__(self, num_features, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.player_embedding = nn.Linear(num_features, embed_dim)

        self.layers = nn.ModuleList([SynergyAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
                for _ in range(num_layers)
            ]
        )
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        # dropout on the residual branches only (not on the residual path itself), the
        # standard transformer placement. with only 78 training samples and 100k+
        # parameters, this is the single cheapest lever against overfitting.
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, players, synergy_matrix, key_padding_mask=None):
        x = self.player_embedding(players)

        for attn, ff, attn_norm, ff_norm in zip(self.layers, self.feed_forwards, self.attn_norms, self.ff_norms):
            attended = attn(attn_norm(x), synergy_mask=synergy_matrix, key_padding_mask=key_padding_mask)
            x = x + self.dropout(attended)

            forwarded = ff(ff_norm(x))
            x = x + self.dropout(forwarded)

        if key_padding_mask is not None:
            real_player_mask = (~key_padding_mask).unsqueeze(-1).float()
            summed = (x * real_player_mask).sum(dim=1)
            count = real_player_mask.sum(dim=1).clamp(min=1.0)
            team_representation = summed / count
        else:
            team_representation = x.mean(dim=1)

        ranking_points = self.regressor(team_representation)
        return ranking_points


def _verify_model_parameters() -> tuple[bool, dict | list | None]:
    logger.info("[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] Verifying the transformers_architecture.json file...")
    json_path = ROOT_DIR / "ml_model_scripts/transformers_architecture.json"

    if not json_path.is_file() or json_path.stat().st_size == 0:
        logger.error("[ TRAIN MODEL | VERIFY MODEL PARAMETERS ] The file does not exist or is empty.")
        return False, None

    try:
        with open(json_path, "r", encoding="utf-8") as arquivo:
            data = json.load(arquivo)
            if data:
                return True, data
            else:
                return False, None
    except json.JSONDecodeError:
        return False, None


def train_model():
    logger.info("[ TRAIN MODEL ] Loading the real data and starting training...")

    ind_path = REFINED_DATA_PATH / "ml_individual_features.csv"
    pairs_path = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    target_path = REFINED_DATA_PATH / "ml_national_team_ranking.csv"

    if not all(p.exists() for p in [ind_path, pairs_path, target_path]):
        logger.error("[ TRAIN MODEL ] CSV files not found at the refined folder!")
        return

    #### ==========================================
    # 1. Dataset, Random Split and DataLoaders
    #### ==========================================
    dataset = FIFANationalTeamDataset(ind_path, pairs_path, target_path, top_k_players=11)

    total_size = len(dataset)
    # separating 20% for test, 10% for validation and 70% for training
    test_size = int(0.20 * total_size)
    val_size = int(0.10 * total_size)
    train_size = total_size - test_size - val_size

    # the generator with 'manual_seed' guarantees that the random cut will be equal if we run the script 2 times in a row
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # only need the train to be shuffle. validation and test don't need
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logger.success(f"[ TRAIN MODEL ] Dataset Split: {train_size} Train | {val_size} Validation | {test_size} Test")

    #### ==========================================
    # 2. instantiate the Model
    #### ==========================================
    valid, architecture_data = _verify_model_parameters()

    if not valid or architecture_data is None:
        logger.error("[ TRAIN MODEL ] Invalid or missing transformers_architecture.json file.")

        return

    initial_model = NationalTeamTransformer(
        num_features=4,
        embed_dim=architecture_data["embed_dim"],
        num_heads=architecture_data["num_heads"],
        num_layers=architecture_data["num_layers"],
    )

    actual_params = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
    logger.info(f"[ TRAIN MODEL ] Actual Trainable Parameters in PyTorch: {actual_params}")

    #### ==========================================
    # 3. hyperparameter test loop (Using Validation Set)
    #### ==========================================
    epochs = 150

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
    ]

    best_val_loss = float("inf")
    best_config = None

    logger.info("[ TRAIN MODEL ] Hyperparameter test started (Evaluating on Validation Set).")
    for i, config in enumerate(configs_to_test):

        actual_model = NationalTeamTransformer(
            num_features=4,
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(actual_model.parameters(), lr=0.001, weight_decay=1e-4)

        for epoch in tqdm.tqdm(range(epochs), desc="Train Hyperparameters Epochs", unit="epoch", colour="white"):
            actual_model.train()

            for players, synergy_mask, padding_mask, targets in train_loader:
                optimizer.zero_grad()
                predictions = actual_model(players, synergy_mask, key_padding_mask=padding_mask)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

        actual_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for players, synergy_mask, padding_mask, targets in val_loader:
                preds = actual_model(players, synergy_mask, key_padding_mask=padding_mask)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            logger.success(f"[ TRAIN MODEL ] New Best Model. Val Loss dropped to {val_loss:.4f} | Config: {config}")
            best_val_loss = val_loss
            best_config = config

    logger.info("=" * 32)
    logger.success(f"[ TRAIN MODEL ] Best Configuration : {best_config}")

    new_data = {
        "num_features": 4,
        "embed_dim": best_config["embed_dim"],
        "num_heads": best_config["num_heads"],
        "num_layers": best_config["num_layers"],
        "trainable_params": actual_params,
        "avg_val_mse": best_val_loss,
    }

    json_path = ROOT_DIR / "ml_model_scripts/transformers_architecture.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

    #### ==========================================
    # 4. Final Training with Checkpointing
    #### ==========================================
    final_model = NationalTeamTransformer(
        num_features=4,
        embed_dim=best_config["embed_dim"],
        num_heads=best_config["num_heads"],
        num_layers=best_config["num_layers"],
    )

    logger.info("[ TRAIN MODEL ] Starting final model train...")

    final_epochs = 300
    best_final_val_loss = float("inf")
    best_final_weights = None

    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in tqdm.tqdm(range(final_epochs), desc="Final Train Epochs", unit="epoch", colour="white"):
        # TRAINING
        final_model.train()

        for players, synergy_mask, padding_mask, targets in train_loader:
            final_optimizer.zero_grad()
            predictions = final_model(players, synergy_mask, key_padding_mask=padding_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            final_optimizer.step()

        # VALIDATION (Early Stopping / Checkpoint)
        final_model.eval()
        current_val_loss = 0.0

        with torch.no_grad():
            for players, synergy_mask, padding_mask, targets in val_loader:
                preds = final_model(players, synergy_mask, key_padding_mask=padding_mask)
                loss = criterion(preds, targets)
                current_val_loss += loss.item()

        current_val_loss /= len(val_loader)

        # if the validation get better, get the weights
        if current_val_loss < best_final_val_loss:
            best_final_val_loss = current_val_loss
            best_final_weights = copy.deepcopy(final_model.state_dict())

    # run the best weigths during the loop
    final_model.load_state_dict(best_final_weights)

    #### ==========================================
    # 5. model evaluation (Test Set & Metrics)
    #### ==========================================
    logger.info("=" * 32)
    logger.info("[ TRAIN MODEL ] Evaluating model on unseen TEST SET...")

    final_model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for players, synergy_mask, padding_mask, targets in test_loader:
            preds = final_model(players, synergy_mask, key_padding_mask=padding_mask)
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    # MSE/val_loss above is intentionally measured in normalized (z-score) space, so it
    # stays comparable across configs regardless of the target's raw scale. MAE and R2
    # are reported back in the original "FIFA Points" scale for interpretability.
    all_predictions = all_predictions * dataset.target_std + dataset.target_mean
    all_targets = all_targets * dataset.target_std + dataset.target_mean

    # calculation of the new metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    logger.success(f"[ TRAIN MODEL ] Mean Absolute Error (MAE): {mae:.2f} FIFA Points")
    logger.success(f"[ TRAIN MODEL ] R-Squared (R2 Score)     : {r2:.4f} ({(r2*100):.1f}%)")
    logger.info("=" * 32)

    model_path = ROOT_DIR / "models"
    model_path.mkdir(parents=True, exist_ok=True)
    model_save_path = model_path / "best_transformer_model.pth"
    torch.save(final_model.state_dict(), model_save_path)

    logger.success(f"[ TRAIN MODEL ] Model weights successfully saved at: {model_save_path}")


if __name__ == "__main__":
    train_model()
