import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd
from tqdm import tqdm

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("process_data.py LOG INITIALIZED.")

RAW_DATA_PATH = ROOT_DIR / "data/raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_PATH = ROOT_DIR / "data/processed"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

EXTERNAL_METADATA_PATH = ROOT_DIR / "data/external_metadata"
EXTERNAL_METADATA_PATH.mkdir(parents=True, exist_ok=True)


def _read_club_world_cup_data() -> pd.DataFrame:
    """
    Reads all downloaded Club World Cup CSV files from the raw data directory and concatenates them into a single DataFrame.
    """
    all_files = list(RAW_DATA_PATH.glob("club_world_cup_*.csv"))
    if not all_files:
        logger.warning("[ PROCESS DATA | READ CLUB WORLD CUP DATA ] No Club World Cup data files found in raw data directory.")

        return pd.DataFrame()  # return empty DataFrame if no files found

    df_list = []
    for file in tqdm(all_files, desc="Reading Club World Cup Data", unit="file", colour="white"):
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            logger.info(f"[ PROCESS DATA | READ CLUB WORLD CUP DATA ] Successfully read file: {file.name}")
        except Exception as e:
            logger.error(f"[ PROCESS DATA | READ CLUB WORLD CUP DATA ] Failed to read file: {file.name}. Error: {e}")

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        logger.success(
            f"[ PROCESS DATA | READ CLUB WORLD CUP DATA ] Successfully combined {len(df_list)} "
            "Club World Cup data files into a single DataFrame."
        )
        return combined_df
    else:
        logger.warning("[ PROCESS DATA | READ CLUB WORLD CUP DATA ] No valid Club World Cup data files were read successfully.")

        return pd.DataFrame()  # return empty DataFrame if no valid files read


def _calculate_confederation_weights(club_world_cup_df: pd.DataFrame) -> dict[str, float] | None:
    """
    Cross the data from Club World Cup with the explicit mapping metadata
    to calculate the Strength/Weight of each Confederation.
    """
    logger.info("[ PROCESS DATA | CALCULATE CONFEDERATION WEIGHTS ] Starting calculation of confederation weights...")

    map_file_path = EXTERNAL_METADATA_PATH / "cwc_confederations_map.csv"
    if not map_file_path.exists():
        logger.error(f"[ PROCESS DATA | CALCULATE CONFEDERATION WEIGHTS ] Mapping not found in: {map_file_path}")

        return None

    df_map = pd.read_csv(map_file_path)

    cwc_confederation_map = dict(zip(df_map["team_name"], df_map["confederation"]))

    club_world_cup_df["home_confederation"] = club_world_cup_df["home_team_name"].map(cwc_confederation_map)
    club_world_cup_df["away_confederation"] = club_world_cup_df["away_team_name"].map(cwc_confederation_map)

    points_per_confederation = {
        "UEFA": {"pontos": 0, "jogos": 0},
        "CAF": {"pontos": 0, "jogos": 0},
        "CONMEBOL": {"pontos": 0, "jogos": 0},
        "CONCACAF": {"pontos": 0, "jogos": 0},
        "AFC": {"pontos": 0, "jogos": 0},
        "OFC": {"pontos": 0, "jogos": 0},
    }

    # run game by game to give the points
    for _, row in tqdm(club_world_cup_df.iterrows(), desc="Calculating Confederation Weights", unit="game", colour="white"):
        home_conf = row["home_confederation"]
        away_conf = row["away_confederation"]
        home_goals = row["home_team_goal_count"]
        away_goals = row["away_team_goal_count"]

        # contabilize the games
        if pd.notna(home_conf) and home_conf in points_per_confederation:
            points_per_confederation[home_conf]["jogos"] += 1
        if pd.notna(away_conf) and away_conf in points_per_confederation:
            points_per_confederation[away_conf]["jogos"] += 1

        # contabilize the points (3 for win, 1 for draw)
        if home_goals > away_goals:
            if pd.notna(home_conf):
                points_per_confederation[home_conf]["pontos"] += 3
        elif away_goals > home_goals:
            if pd.notna(away_conf):
                points_per_confederation[away_conf]["pontos"] += 3
        else:
            if pd.notna(home_conf):
                points_per_confederation[home_conf]["pontos"] += 1
            if pd.notna(away_conf):
                points_per_confederation[away_conf]["pontos"] += 1

    logger.success("[ PROCESS DATA | CALCULATE CONFEDERATION WEIGHTS ] Global Result and Calculated Weights:")

    uefa_ppg = 0
    if points_per_confederation["UEFA"]["jogos"] > 0:
        uefa_ppg = points_per_confederation["UEFA"]["pontos"] / points_per_confederation["UEFA"]["jogos"]

    weights = {}
    for conf, stats in points_per_confederation.items():
        if stats["jogos"] > 0:
            ppg = stats["pontos"] / stats["jogos"]
            stats["ppg"] = ppg

            # if UEFA is the base 1.0
            relative_weight = ppg / uefa_ppg if uefa_ppg > 0 else 0
            weights[conf] = relative_weight

            logger.success(
                f"[ PROCESS DATA | CALCULATE CONFEDERATION WEIGHTS ] {conf:>8} | PPG: {ppg:.2f} ({stats['jogos']} jogos) | "
                f"Peso Relativo: {relative_weight:.3f}"
            )

    return weights


def process_data():
    """
    Processes the downloaded Club World Cup data and external metadata.
    This function reads the raw data files, performs necessary transformations,
    and saves the processed data for further analysis.
    """
    logger.info("[ PROCESS DATA ] Starting data processing.")

    club_world_cup_df = _read_club_world_cup_data()

    if not club_world_cup_df.empty:
        weights = _calculate_confederation_weights(club_world_cup_df)

        if weights:
            metadata_path = EXTERNAL_METADATA_PATH / "world_teams_metadata.csv"
            df_world_teams = pd.read_csv(metadata_path)

            df_world_teams["confederation_weight"] = df_world_teams["confederation"].map(weights)

            # if a continent did not play in the World Cup (e.g., OFC - Oceania) or has no weight,
            # itś filled it with a base value (e.g., 0.1 or 0.0)
            df_world_teams["confederation_weight"] = df_world_teams["confederation_weight"].fillna(0.1)

            processed_teams_file = PROCESSED_DATA_PATH / "processed_world_teams.csv"
            df_world_teams.to_csv(processed_teams_file, index=False)

            processed_cwc_file = PROCESSED_DATA_PATH / "processed_club_world_cup_data.csv"
            club_world_cup_df.to_csv(processed_cwc_file, index=False)

            logger.success(f"[ PROCESS DATA ] Master teams table updated with weights and saved in: {processed_teams_file}")
        else:
            logger.error("[ PROCESS DATA ] Failed to generate weights. The master teams table was not updated.")
    else:
        logger.warning("[ PROCESS DATA ] No Club World Cup data to process.")


if __name__ == "__main__":
    process_data()
