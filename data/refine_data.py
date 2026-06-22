import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("refine_data.py LOG INITIALIZED.")

PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

REFINED_DATA_PATH = ROOT_DIR / "data/refined"
REFINED_DATA_PATH.mkdir(parents=True, exist_ok=True)


def refine_data():
    """
    Transforms enriched player data into final Machine Learning features:
    Focuses on National Team Synergy (Chemistry between compatriots at the club level).
    """
    logger.info("[ REFINE DATA ] Starting Feature Engineering for Machine Learning...")

    players_file = PROCESSED_DATA_PATH / "processed_players.csv"
    appearances_file = PROCESSED_DATA_PATH / "processed_appearances.csv"

    if not players_file.exists() or not appearances_file.exists():
        logger.error("[ REFINE DATA ] Processed files not found! Run process_data.py first.")

        return

    df_players = pd.read_csv(players_file)
    df_appearances = pd.read_csv(appearances_file)

    #### =========================================================
    # 1. PREPARING THE SYNERGY MATRIX
    #### =========================================================
    logger.info("[ REFINE DATA ] Calculating National Synergy Matrix...")

    # reduceing the appearances to the minimum necessary to save RAM
    app_reduced = df_appearances[["game_id", "player_id", "player_club_id", "goals", "assists"]].copy()

    # get the nationality and confederation weight for each player before merging
    app_enriched = app_reduced.merge(
        df_players[["player_id", "name", "country_of_citizenship", "confederation_weight"]], on="player_id", how="inner"
    )

    #### =========================================================
    # 2. THE CROSS JOIN (SELF-MERGE)
    #### =========================================================
    # pandas cross join the table with it self: "who played with who, at the same club, at the same game?"
    df_pairs = app_enriched.merge(app_enriched, on=["game_id", "player_club_id"], suffixes=("_A", "_B"))

    # critical Filter 1: remove duplicates (A with B is the same as B with A) and self-matches (A with A)
    df_pairs = df_pairs[df_pairs["player_id_A"] < df_pairs["player_id_B"]]

    # critical filter 2: the selection magic
    # just matters duples of the same nationality. this reduces the table size by about 90%
    df_pairs = df_pairs[df_pairs["country_of_citizenship_A"] == df_pairs["country_of_citizenship_B"]]

    if df_pairs.empty:
        logger.warning("[ REFINE DATA ] Nenhuma dupla de compatriotas encontrada.")
        return

    #### =========================================================
    # 3. SINERGY STRENGTH CALCULATION
    #### =========================================================
    # the sinergy is the sum of the goals participated in by the pair in the match, multiplied by the confederation weight
    # (since they play at the same club, the confederation_weight_A is equal to B)
    df_pairs["combined_output"] = df_pairs["goals_A"] + df_pairs["goals_B"] + df_pairs["assists_A"] + df_pairs["assists_B"]
    df_pairs["synergy_points"] = df_pairs["combined_output"] * df_pairs["confederation_weight_A"]

    # groups all the history of that pair
    synergy_matrix = (
        df_pairs.groupby(["player_id_A", "name_A", "player_id_B", "name_B", "country_of_citizenship_A"])
        .agg(
            matches_played_together=("game_id", "count"),
            total_synergy_score=("synergy_points", "sum"),
            total_combined_goals_assists=("combined_output", "sum"),
        )
        .reset_index()
    )

    # renames columns for better readability
    synergy_matrix = synergy_matrix.rename(columns={"country_of_citizenship_A": "national_team"})

    # critical Filter 3: relevancy
    # just mantein pairs that player together at least 5 times to the ML model don't learn with 'noise'
    synergy_matrix = synergy_matrix[synergy_matrix["matches_played_together"] >= 5]

    # order to see the better pairs first
    synergy_matrix = synergy_matrix.sort_values(by="total_synergy_score", ascending=False)

    #### =========================================================
    # 4. SAVING
    #### =========================================================
    output_file = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    synergy_matrix.to_csv(output_file, index=False)

    logger.success(f"[ REFINE DATA ] Sinergy calculated with success! {len(synergy_matrix)} partnerships saved.")
    logger.info(f"[ REFINE DATA ] File saved at: {output_file}")


if __name__ == "__main__":
    refine_data()
