"""
embed_dim, num_heads, and num_layers in train_model.py were set to fixed numbers
(64, 4, 2) with no justification tied to this project's actual data. There is no
closed-form formula that computes "the correct" values for these from a dataset -
they are hyperparameters, not statistics you can derive analytically. But picking
them by guessing is not the only alternative either.

Two things ARE true mathematically about this specific model and dataset:

1. HARD CONSTRAINT: embed_dim must be divisible by num_heads (head_dim =
   embed_dim / num_heads must be an integer). This eliminates invalid configs,
   but doesn't choose a value for you.

2. STRUCTURAL FACT: SynergyAttention is dense self-attention - every player
   already attends to all 10 others within a SINGLE layer (the synergy matrix
   only biases attention scores, it never restricts who can attend to whom).
   Unlike sparse Graph Neural Networks, stacking more layers here does not
   increase how far information can travel (it already reaches everyone in
   layer 1) - it only adds non-linear capacity. With only 112 labeled
   countries, that extra capacity is more likely to cause overfitting than
   to help, which argues for keeping num_layers small.

On top of that, the *current* config (embed_dim=64, num_layers=2) has ~102,000
trainable parameters for 112 training samples - roughly 900 parameters per
label. There's no universal threshold for "too many", but that ratio is a
classic overfitting red flag, which is why the search grid below starts much
smaller (8-32) instead of refining around 64.

----------------------
For every VALID (embed_dim, num_heads, num_layers) combination in a small
candidate grid, it runs K-fold cross-validation: the 112 countries are split
into K folds, and for each fold a FRESH model is trained on the other K-1
folds and evaluated (MSE) on the held-out fold. The average held-out MSE
across folds is that configuration's score. This is the standard, empirically
grounded way to compare architectures when no formula exists: it measures
generalization on data the model never trained on, instead of guessing or
reading training loss (which would just reward the biggest model memorizing
the data).

K-fold (not a single train/val split) is used specifically because 112 samples
is small enough that one split could be lucky/unlucky by chance; averaging
over K different held-out folds gives a much more reliable comparison.

-----------------------------
This script does NOT train your final model and does NOT modify train_model.py
automatically. It prints a ranked list and a winning (embed_dim, num_heads,
num_layers) tuple. You take that tuple and replace the hardcoded
NationalTeamTransformer(num_features=4, embed_dim=64, num_heads=4, num_layers=2)
call inside train_model() in train_model.py with those values, then run the
normal full training (all 112 countries, full epoch budget) to produce the
model you actually deploy. Cross-validation here is only for SELECTING the
architecture - the final model should still be trained on 100% of the data.
"""

import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_model import NationalTeamTransformer
from utils.model_extensions import FIFANationalTeamDataset

import tqdm

REFINED_DATA_PATH = ROOT_DIR / "data/refined"

NUM_FEATURES = 4  # total_matches, total_weighted_wins, total_weighted_plus_minus, win_rate_percentage
K_FOLDS = 5  # 112 countries is too small to trust a single train/val split
SEARCH_EPOCHS = 50  # shorter than the final 150-epoch run: here we only need to RANK architectures, not fully converge each one
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# deliberately small candidates: the current embed_dim=64 setup is already
# ~900 params/sample (see module docstring). going bigger would only make
# the overfitting risk worse, so the search starts well below it
EMBED_DIM_CANDIDATES = [8, 16, 32]
NUM_HEADS_OPTIONS = [1, 2, 4, 8]
NUM_LAYERS_CANDIDATES = [1, 2]


def _build_valid_configs() -> list[tuple[int, int, int]]:
    """
    Generates every (embed_dim, num_heads, num_layers) combination that:
      1. Satisfies the hard constraint: embed_dim % num_heads == 0.
      2. Satisfies a soft heuristic: head_dim (embed_dim / num_heads) >= 2.
        Not mathematically required, but a 1-dimensional attention head
        carries almost no information and would just waste search time.
    """
    configs = []
    for embed_dim in EMBED_DIM_CANDIDATES:
        for num_heads in NUM_HEADS_OPTIONS:
            if embed_dim % num_heads != 0 or embed_dim // num_heads < 2:
                continue

            for num_layers in NUM_LAYERS_CANDIDATES:
                configs.append((embed_dim, num_heads, num_layers))

    return configs


def _k_fold_indices(num_samples: int, k: int, seed: int) -> list[list[int]]:
    """
    Shuffles sample indices once, then splits them into k roughly-equal folds.
    """
    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    fold_sizes = [num_samples // k + (1 if i < num_samples % k else 0) for i in range(k)]
    folds, start = [], 0

    for size in fold_sizes:
        folds.append(indices[start : start + size])  # noqa: E203
        start += size

    return folds


def _train_one_fold(dataset, train_idx, val_idx, embed_dim, num_heads, num_layers) -> float:
    """
    Trains a brand-new model on train_idx, evaluates it on the held-out val_idx,
    and returns the validation MSE. A fresh model is required for every fold:
    reusing weights across folds would leak information between folds and
    invalidate the comparison between configurations.
    """
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = NationalTeamTransformer(num_features=NUM_FEATURES, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(SEARCH_EPOCHS):
        for players, synergy_mask, padding_mask, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(players, synergy_mask, key_padding_mask=padding_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for players, synergy_mask, padding_mask, targets in val_loader:
            predictions = model(players, synergy_mask, key_padding_mask=padding_mask)
            total_loss += criterion(predictions, targets).item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def architecture_loader():
    torch.manual_seed(RANDOM_SEED)

    ind_path = REFINED_DATA_PATH / "ml_individual_features.csv"
    pairs_path = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    target_path = REFINED_DATA_PATH / "ml_national_team_ranking.csv"
    dataset = FIFANationalTeamDataset(ind_path, pairs_path, target_path, top_k_players=11)

    folds = _k_fold_indices(len(dataset), K_FOLDS, RANDOM_SEED)
    configs = _build_valid_configs()
    logger.info(
        f"[ ARCHITECTURE LOADER ] Testing {len(configs)} valid architectures with {K_FOLDS}-fold CV on {len(dataset)} "
        "countries...\n"
    )

    results = []
    for embed_dim, num_heads, num_layers in tqdm.tqdm(configs, desc="Testing Architectures", colour="white"):
        fold_losses = []

        for fold_idx in range(K_FOLDS):
            val_idx = folds[fold_idx]
            train_idx = [i for f, fold in enumerate(folds) if f != fold_idx for i in fold]
            fold_losses.append(_train_one_fold(dataset, train_idx, val_idx, embed_dim, num_heads, num_layers))

        avg_val_loss = sum(fold_losses) / len(fold_losses)
        param_count = sum(
            p.numel()
            for p in NationalTeamTransformer(
                num_features=NUM_FEATURES, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
            ).parameters()
        )
        results.append((avg_val_loss, embed_dim, num_heads, num_layers, param_count))
        logger.info(
            f"[ ARCHITECTURE LOADER ] embed_dim={embed_dim:>2} num_heads={num_heads} num_layers={num_layers} "
            f"[ ARCHITECTURE LOADER ] avg val MSE: {avg_val_loss:.4f} ({param_count} params)"
        )

    results.sort(key=lambda r: r[0])
    best_loss, best_embed_dim, best_num_heads, best_num_layers, best_params = results[0]

    logger.info("\n" + "=" * 60)
    logger.info("[ ARCHITECTURE LOADER ] BEST CONFIGURATION (lowest average cross-validation MSE):")
    logger.info(
        f"[ ARCHITECTURE LOADER ]   embed_dim={best_embed_dim}, num_heads={best_num_heads}, num_layers={best_num_layers}"
    )
    logger.info(f"[ ARCHITECTURE LOADER ]  avg val MSE: {best_loss:.4f} | trainable params: {best_params}")

    architecture_data = {
        "embed_dim": best_embed_dim,
        "num_heads": best_num_heads,
        "num_layers": best_num_layers,
        "avg_val_mse": best_loss,
        "trainable_params": best_params,
    }

    architecture_data_path = ROOT_DIR / "ml_model_scripts/transformers_architecture.json"

    with open(architecture_data_path, "w", encoding="utf-8") as file:
        json.dump(architecture_data, file, indent=4, ensure_ascii=False)

    logger.info(f"[ ARCHITECTURE LOADER ] Data successfully saved at: {architecture_data_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    architecture_loader()
