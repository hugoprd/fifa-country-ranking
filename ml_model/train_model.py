import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger

import tqdm

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

        # 1. input Embedding: Transforms the player numeric stats into a Dense Vector
        self.player_embedding = nn.Linear(num_features, embed_dim)

        # 2. transformer Layers with the Synergy Attention
        self.layers = nn.ModuleList([SynergyAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
                for _ in range(num_layers)
            ]
        )

        # 3. prediction Head (Regression for predicting FIFA Points/Ranking)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)  # returns a single number (the team's points)
        )

    def forward(self, players, synergy_matrix):
        """
        players: Tensor of shape (Batch_Size, 11, Num_Features) -> The 11 starters
        synergy_matrix: Tensor of shape (Batch_Size, 11, 11) -> The chemistry between the 11
        """
        # pass the player attributes to create the initial token
        x = self.player_embedding(players)

        # pass through the Transformer
        for attn, ff in zip(self.layers, self.feed_forwards):
            # A mágica do entrosamento!
            attended = attn(x, synergy_mask=synergy_matrix)
            x = x + attended  # Conexão Residual

            forwarded = ff(x)
            x = x + forwarded

        # Global Average Pooling (takes the average of the whole team to form the national team)
        # x shape before: (Batch, 11, Embed_Dim) -> after: (Batch, Embed_dim)
        team_representation = x.mean(dim=1)

        # calculates the final strength/ranking
        ranking_points = self.regressor(team_representation)

        return ranking_points


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

    # num_features = 4 (the 4 columns we set in the class: matches, wins, plus_minus, win_rate)
    model = NationalTeamTransformer(num_features=4, embed_dim=64, num_heads=4, num_layers=2)

    #### ==========================================
    # 3. configures the optimizer and the Loss Function (Loss Function = MSELoss)
    #### ==========================================

    # MSELoss = Medium Quadratic Error
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #### ==========================================
    # 4. training loop (epochs)
    #### ==========================================
    epochs = 150
    model.train()  # puts the model in training mode

    for epoch in tqdm.tqdm(range(epochs), desc="Training", unit="epoch", colour="white"):
        epoch_loss = 0.0

        for players, synergy_mask, targets in dataloader:
            optimizer.zero_grad()  # clears the previous calculations

            # forward pass: the model tries to predict the country's score
            predictions = model(players, synergy_mask)

            # calculates the error by comparing with the real Target
            loss = criterion(predictions, targets)

            # backward pass and optimization (learning with the mistake)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        logger.info(f"[ TRAIN MODEL ] Epoch {epoch+1}/{epochs} - Average Error (Loss): {avg_loss:.4f}")

    logger.success("[ TRAIN MODEL ] Training Completed.")


if __name__ == "__main__":
    train_model()
