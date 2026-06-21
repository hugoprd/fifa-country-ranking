import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd
import io
import requests
from tqdm import tqdm

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("extract_data.py LOG INITIALIZED.")

RAW_DATA_PATH = ROOT_DIR / "data/raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
EXTERNAL_METADATA_PATH = ROOT_DIR / "data/external_metadata"
EXTERNAL_METADATA_PATH.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    extract_data()
