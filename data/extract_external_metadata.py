import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger
from logs.set_logger import setup_logger
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("extract_external_metadata.py LOG INITIALIZED.")

EXTERNAL_METADATA_PATH = ROOT_DIR / "data" / "external_metadata"
EXTERNAL_METADATA_PATH.mkdir(parents=True, exist_ok=True)


def extract_external_metadata():
    """
    Extracts metadata about football clubs from all global confederations
    via Wikipedia and saves it as a master CSV dimension table.
    """
    logger.info("[ EXTRACT EXTERNAL METADATA ] Starting metadata extraction for ALL confederations from Wikipedia.")

    urls = {
        "CAF": "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CAF_countries",
        "CONMEBOL": "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONMEBOL_countries",
        "UEFA": "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_UEFA_countries",
        "CONCACAF": "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_CONCACAF_countries",
        "AFC": "https://en.wikipedia.org/wiki/List_of_top-division_football_clubs_in_AFC_countries",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    world_data = []

    for confederation, url in tqdm(urls.items(), desc="Downloading Data", unit="confed", colour="white"):
        logger.info(f"[ EXTRACT EXTERNAL METADATA ] Extracting teams for: {confederation}")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            import re

            def clean_text(t: str) -> str:
                if t is None:
                    return ""
                # remove bracketed references like [1], [a]
                t = re.sub(r"\[.*?\]", "", t)
                return t.replace("(R)", "").strip()

            collected_before = len(world_data)

            # iterate over section headings (country names) and find the first wikitable after each
            headings = []
            # prefer span.mw-headline when present
            for span in soup.find_all("span", class_="mw-headline"):
                headings.append(span)
            # also include raw h2/h3/h4 tags (some pages render differently)
            for h in soup.find_all(["h2", "h3", "h4"]):
                headings.append(h)

            for span in headings:
                # extract a reasonable country/section title
                country = span.get_text().strip()
                # normalize and skip boilerplate sections
                lower = country.lower()
                if any(
                    skip in lower
                    for skip in ("see also", "references", "further reading", "external links", "notes", "navigation")
                ):
                    continue

                # locate the heading tag (h2/h3/h4) and iterate siblings until next heading
                heading_tag = span.find_parent(["h2", "h3", "h4"]) or span.parent
                # table_found = False
                from bs4 import Tag

                for sibling in heading_tag.next_siblings:
                    if isinstance(sibling, Tag) and sibling.name in ("h2", "h3", "h4"):
                        # reached next section
                        break
                    if not isinstance(sibling, Tag):
                        continue
                    if sibling.name == "table":
                        table = sibling
                        # parse rows
                        for tr in table.find_all("tr"):
                            tds = tr.find_all(["td", "th"])
                            if not tds:
                                continue
                            club_cell = tds[0]
                            club_name = clean_text(club_cell.get_text())
                            if not club_name:
                                continue
                            world_data.append(
                                {
                                    "team_name": club_name,
                                    "confederation": confederation,
                                    "country": country,
                                    "competition": "",
                                }
                            )
                        # table_found = True
                        break
                    # sometimes tables are nested inside divs; search inside
                    if sibling.find_all("table"):
                        for table in sibling.find_all("table"):
                            for tr in table.find_all("tr"):
                                tds = tr.find_all(["td", "th"])
                                if not tds:
                                    continue
                                club_cell = tds[0]
                                club_name = clean_text(club_cell.get_text())
                                if not club_name:
                                    continue
                                world_data.append(
                                    {
                                        "team_name": club_name,
                                        "confederation": confederation,
                                        "country": country,
                                        "competition": "",
                                    }
                                )
                        # table_found = True

                        break

            logger.success(
                f"[ EXTRACT EXTERNAL METADATA ] Finished extracting {confederation} ({len(world_data)-collected_before} "
                f"collected this confederation, {len(world_data)} total)."
            )
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.error(f"[ EXTRACT EXTERNAL METADATA ] Failed to fetch data for {confederation}: {e}")

    df_world_metadata = pd.DataFrame(world_data)

    df_world_metadata = df_world_metadata.drop_duplicates(subset=["team_name"])

    output_path = EXTERNAL_METADATA_PATH / "world_teams_metadata.csv"
    df_world_metadata.to_csv(output_path, index=False)

    logger.success(f"[ EXTRACT EXTERNAL METADATA ] Extraction complete. {len(df_world_metadata)} teams saved to {output_path}.")


if __name__ == "__main__":
    extract_external_metadata()
