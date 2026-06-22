import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd
import io
import requests
import soccerdata as sd
from tqdm import tqdm
import time
import shutil

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("extract_data.py LOG INITIALIZED.")

RAW_DATA_PATH = ROOT_DIR / "data/raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
EXTERNAL_METADATA_PATH = ROOT_DIR / "data/external_metadata"
EXTERNAL_METADATA_PATH.mkdir(parents=True, exist_ok=True)


def _cleanup_temp_files():
    """
    Removes folder 'downloaded_files' created automatically by soccerdata/FBref,
    keeping the project root clean.
    """
    logger.info(
        "[ EXTRACT DATA | CLEANUP TEMP FILES ] Removing temporary folder 'downloaded_files' created by soccerdata/FBref..."
    )
    temp_dir = ROOT_DIR / "downloaded_files"

    if temp_dir.exists() and temp_dir.is_dir():
        try:
            shutil.rmtree(temp_dir)
            logger.info("[ EXTRACT DATA | CLEANUP TEMP FILES ] Temporary folder 'downloaded_files' removed successfully.")
        except Exception as e:
            logger.warning(f"[ EXTRACT DATA | CLEANUP TEMP FILES ] Could not remove temporary folder: {e}")


def _download_club_world_cup_data(urls: dict[str, str]):
    """
    Downloads Club World Cup match data from footystats.org for the specified seasons.
    Saves each season's data as a separate CSV file in the raw data directory.
    """
    for season, url in tqdm(urls.items(), desc="Downloading Club World Cup Data", unit="season", colour="white"):
        logger.info(f"[ EXTRACT DATA | DOWNLOAD CLUB WORLD CUP DATA ] Downloading Club World Cup data for season: {season}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))
            output_file = RAW_DATA_PATH / f"club_world_cup_{season.replace('/', '-')}.csv"
            df.to_csv(output_file, index=False)
            logger.success(
                f"[ EXTRACT DATA | DOWNLOAD CLUB WORLD CUP DATA ] Successfully downloaded and "
                f"saved Club World Cup data for season: {season}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(
                f"[ EXTRACT DATA | DOWNLOAD CLUB WORLD CUP DATA ] Failed to download Club World Cup data for season: {season}. "
                f"ERROR: {e}"
            )


def _extract_world_cup_data():
    """
    Extracts Club World Cup match data from FBref.com for the specified seasons.
    Saves each season's data as a separate CSV file in the raw data directory.
    """
    logger.info("[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Starting extraction of World Cup data from FBref...")

    try:
        # in soccerdata, the World Cup is mapped to the league 'INT-World Cup'
        fbref = sd.FBref(leagues="INT-World Cup", seasons=["2018", "2022"])

        logger.info("[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Downloading games history...")
        df_games = fbref.read_schedule()

        games_file = RAW_DATA_PATH / "fbref_world_cup_games.csv"
        games_file.parent.mkdir(parents=True, exist_ok=True)
        df_games.to_csv(games_file)
        logger.success(f"[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Games saved to: {games_file}")

        time.sleep(3)

        # FBref limits 20 requisitions per minute
        # soccerdata pauses by itself if it reaches the limit
        logger.info(
            "[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Downloading player statistics by match. "
            "This may take a few minutes..."
        )
        df_players = fbref.read_player_match_stats(stat_type="summary")

        players_file = RAW_DATA_PATH / "fbref_world_cup_appearances.csv"
        players_file.parent.mkdir(parents=True, exist_ok=True)
        df_players.to_csv(players_file)
        logger.success(f"[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Appearances saved to: {players_file}")
    except Exception as e:
        logger.error(f"[ EXTRACT DATA | EXTRACT WORLD CUP DATA ] Failed to extract data from FBref: {e}")


def extract_data():
    """
    Extracts raw data for FIFA country rankings and related football metadata.
    """
    logger.info("[ EXTRACT DATA ] Starting data extraction process.")

    CLUB_WORLD_CUP_URLS = {
        "2025/2025": "https://footystats.org/c-dl.php?type=matches&comp=13878",
        "2024/2024": "https://footystats.org/c-dl.php?type=matches&comp=13557",
        "2023/2023": "https://footystats.org/c-dl.php?type=matches&comp=10958",
        "2022/2022": "https://footystats.org/c-dl.php?type=matches&comp=8830",
        "2021/2021": "https://footystats.org/c-dl.php?type=matches&comp=7069",
        "2020/2020": "https://footystats.org/c-dl.php?type=matches&comp=5517",
        "2019/2019": "https://footystats.org/c-dl.php?type=matches&comp=3370",
        "2018/2018": "https://footystats.org/c-dl.php?type=matches&comp=1813",
    }

    _download_club_world_cup_data(CLUB_WORLD_CUP_URLS)

    _extract_world_cup_data()

    _cleanup_temp_files()


if __name__ == "__main__":
    extract_data()
