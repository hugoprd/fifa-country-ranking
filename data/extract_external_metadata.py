import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd
import requests
import os
import sys
from io import StringIO

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("extract_external_metadata.py LOG INITIALIZED.")

EXTERNAL_METADATA_DIR = ROOT_DIR / "data" / "external_metadata"
EXTERNAL_METADATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_external_metadata():
    """
    Extracts metadata about African football clubs from Wikipedia and saves it as a CSV.
    """
    logger.info("Starting metadata extraction for CAF clubs from Wikipedia.")

    url = "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CAF_countries"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        logger.debug(f"Fetched URL {url} with status {response.status_code}; content length={len(response.text)}")

        # suppress accidental prints from underlying parsers by parsing from a StringIO
        sio = StringIO(response.text)
        # some HTML parsers or libraries may print to stdout on warnings/errors.
        # redirect stdout temporarily to avoid dumping HTML to console.
        _stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w")
            tables = pd.read_html(sio)
        finally:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = _stdout

        caf_data = []

        for table in tables:
            if "Club" in table.columns:
                # get the 'Club' column and removes empty names
                clubs = table["Club"].dropna().tolist()

                for club in clubs:
                    clean_club = str(club).split("[")[0].replace("(R)", "").strip()
                    caf_data.append({"team_name": clean_club, "confederation": "CAF"})

        df_metadata_caf = pd.DataFrame(caf_data)

        df_metadata_caf = df_metadata_caf.drop_duplicates(subset=["team_name"])

        out_path = EXTERNAL_METADATA_DIR / "caf_teams.csv"
        df_metadata_caf.to_csv(out_path, index=False)

        logger.success(f"Metadata extraction completed. {len(df_metadata_caf)} CAF clubs saved to '{out_path}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from Wikipedia: {e}")
    except ValueError as e:
        logger.error(f"No tables could be parsed from the HTML: {e}")
    except Exception as e:
        # catch-all to log unexpected errors (including parser exceptions)
        logger.exception(f"Unexpected error during metadata extraction: {e}")


if __name__ == "__main__":
    extract_external_metadata()
