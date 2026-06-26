import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger

from ml_model_scripts.train_model import NationalTeamTransformer, _verify_model_parameters
from utils.model_extensions import FIFANationalTeamDataset

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "predict_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

REFINED_DATA_PATH = ROOT_DIR / "data/refined"
MODEL_WEIGHTS_PATH = ROOT_DIR / "models/best_transformer_model.pth"


def generate_fifa_ranking():
    logger.info("=" * 32)
    logger.info("[ GENERATE FIFA RANKING ] Starting National Team Ranking Predictor...")

    # 1. loads the JSON parameters to build the net with the exact size
    valid, architecture_data = _verify_model_parameters()
    if not valid or architecture_data is None:
        logger.error("[ GENERATE FIFA RANKING ] Missing architecture JSON.")

        return

    # 2. instances the model (empty)
    model = NationalTeamTransformer(
        num_features=4,
        embed_dim=architecture_data["embed_dim"],
        num_heads=architecture_data["num_heads"],
        num_layers=architecture_data["num_layers"],
    )

    # 3. load the model weights
    if not MODEL_WEIGHTS_PATH.exists():
        logger.error(f"[ GENERATE FIFA RANKING ] Model weights not found at {MODEL_WEIGHTS_PATH}. Run train_model.py first!")

        return

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))

    # MUITO IMPORTANTE puts the model in avaliation mode (turns off the training Dropout and BatchNorms)
    model.eval()
    logger.success("[ GENERATE FIFA RANKING ] Trained model successfully loaded and set to evaluation mode.")

    # 4. prepare the data
    ind_path = REFINED_DATA_PATH / "ml_individual_features.csv"
    pairs_path = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    target_path = REFINED_DATA_PATH / "ml_national_team_ranking.csv"

    dataset = FIFANationalTeamDataset(ind_path, pairs_path, target_path, top_k_players=11)

    # batch_size=1 to predict a country per time in a clean way
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions_list = []

    # 5. predction loop
    logger.info("[ GENERATE FIFA RANKING ] Processing countries through the Transformer...")

    # torch.no_grad() turns off the gradient calculus
    with torch.no_grad():
        for i, (players, synergy_mask, padding_mask, _) in enumerate(dataloader):

            # gets the country name to the indice
            country_name = dataset.countries[i]

            # goes through the net
            predicted_score_tensor = model(players, synergy_mask, key_padding_mask=padding_mask)

            # gets the tensor number, then undoes the z-score normalization applied
            # to the training target so ml_synergy_power stays in the original scale
            predicted_score = predicted_score_tensor.item()
            predicted_score = predicted_score * dataset.target_std + dataset.target_mean

            predictions_list.append({"national_team": country_name, "ml_synergy_power": predicted_score})

    # 6. creates the final FIFA ranking and saves
    df_ranking = pd.DataFrame(predictions_list)

    # order the best to the worst
    df_ranking = df_ranking.sort_values(by="ml_synergy_power", ascending=False).reset_index(drop=True)

    df_ranking.index = df_ranking.index + 1
    df_ranking = df_ranking.reset_index()
    df_ranking.rename(columns={"index": "ai_rank"}, inplace=True)

    output_path = ROOT_DIR / "ml_fifa_ranking.csv"
    df_ranking.to_csv(output_path, index=False)

    logger.success("=" * 32)
    logger.success(f"[ GENERATE FIFA RANKING ] Ranking generated successfully. Saved at: {output_path}")

    # 7. compares the original FIFA ranking with the Synergy Model
    logger.info("[ GENERATE FIFA RANKING ] Performing direct comparison with the Official FIFA Top 10...")

    # official top 10 FIFA ranking (22/06/2026)
    fifa_oficial = {
        "Argentina": 1,
        "France": 2,
        "Spain": 3,
        "England": 4,
        "Brazil": 5,
        "Morocco": 6,
        "Netherlands": 7,
        "Germany": 8,
        "Portugal": 9,
        "Belgium": 10,
    }

    df_fifa = pd.DataFrame(list(fifa_oficial.items()), columns=["national_team", "fifa_rank"])

    df_top_ai = df_ranking.head(15).copy()

    df_comparacao = pd.merge(df_top_ai, df_fifa, on="national_team", how="left")

    # calculates the positions difference (delta)
    # if delta is positive (+), the synergy model valorizes more the country than FIFA
    # if delta is negative (-), the synergy model thinks the country is worse than what FIFA announced
    df_comparacao["rank_diff"] = df_comparacao["fifa_rank"] - df_comparacao["ai_rank"]

    df_comparacao["fifa_rank"] = df_comparacao["fifa_rank"].fillna("Outside Top 10")

    # delta visual formation (+3, -2, =)
    def format_delta(value):
        if pd.isna(value):
            return "N/A"
        elif value > 0:
            return f"↑ +{int(value)}"
        elif value < 0:
            return f"↓ {int(value)}"
        else:
            return "— ="

    df_comparacao["rank_diff"] = df_comparacao["rank_diff"].apply(format_delta)

    df_comparacao = df_comparacao.rename(
        columns={
            "ai_rank": "ML Rank",
            "national_team": "Country",
            "ai_synergy_power": "Synergy Score",
            "fifa_rank": "FIFA Rank",
            "rank_diff": "ML vs FIFA (Delta)",
        }
    )

    print("\n" + "=" * 70)
    print("[ GENERATE FIFA RANKING ] ML SYNERGY RANKING vs OFFICIAL FIFA RANKING (TOP 15 OVERLAP)")
    print("=" * 70)

    print(df_comparacao.to_string(index=False))
    print("=" * 70)
    print("Caption: [↑] The ML valorizes more the country than FIFA | [↓] The ML valorizes less.\n")


if __name__ == "__main__":
    generate_fifa_ranking()
