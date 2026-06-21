import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loguru import logger  # noqa: E402
from logs.set_logger import setup_logger  # noqa: E402
import soccerdata as sd  # noqa: E402
from soccerdata import FBref  # noqa: E402
import pandas as pd  # noqa: E402

LOG_FILE = ROOT_DIR / "logs"
LOG_NAME = "data_log"
setup_logger(log_file_path=LOG_FILE, log_name=LOG_NAME, overwrite=True)

logger.info("=" * 32)
logger.info("extract_data.py LOG INITIALIZED.")

RAW_DATA_PATH = ROOT_DIR / "data/raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

COMPETITIONS = [
    "INT-FIFA World Cup",
    "INT-FIFA Club World Cup",
    "INT-Copa Libertadores",
    "INT-Copa Sudamericana",
    "INT-UEFA Champions League",
    "INT-UEFA Europa League",
    "INT-UEFA Europa Conference League",
    "INT-CONCACAF Champions Cup",
]


# UEFA's reference weight in this model. CAF's final weight is derived as a
# fraction of this value, proportional to CAF vs. UEFA performance in the
# Club World Cup.
UEFA_BASE_WEIGHT = 1.00

# Fallback weight used only if UEFA's PPG in the dataset is 0 (i.e. no UEFA
# games found), which would make the ratio undefined.
CAF_FALLBACK_WEIGHT = 0.50


def calculate_ppg(df: pd.DataFrame, team_list: list[str]) -> float:
    """
    Calculates average points per game (PPG) for a group of teams using vectorized operations.

    Mathematically precise calculation:
    - Win = 3 points
    - Draw = 1 point
    - Loss = 0 points
    - PPG = Total Points / Total Games

    Note: "games" counts each team appearance. If two teams from team_list play each other,
    that single match contributes two team appearances (one per team) and the points each
    earned—equivalent to how a league table computes points per team.

    Args:
        df: DataFrame with columns: home_team, away_team, home_goals, away_goals
        team_list: List of teams to calculate PPG for

    Returns:
        Average points per game (float, 0-3 scale)
    """
    # filter games for teams in the list
    home_games = df[df["home_team"].isin(team_list)].copy()
    away_games = df[df["away_team"].isin(team_list)].copy()

    # total number of games (team appearances)
    total_games = len(home_games) + len(away_games)

    if total_games == 0:
        return 0.0

    # Calculate points using vectorized operations for numerical precision
    # Home team points: 3 if win, 1 if draw, 0 if loss
    home_points = (home_games["home_goals"] > home_games["away_goals"]).astype(int) * 3 + (
        home_games["home_goals"] == home_games["away_goals"]
    ).astype(int) * 1

    # Away team points: 3 if win, 1 if draw, 0 if loss
    away_points = (away_games["away_goals"] > away_games["home_goals"]).astype(int) * 3 + (
        away_games["away_goals"] == away_games["home_goals"]
    ).astype(int) * 1

    total_points = home_points.sum() + away_points.sum()

    return total_points / total_games


def calculate_performance_metrics(df: pd.DataFrame, team_list: list[str]) -> dict:
    """
    Calculates comprehensive performance metrics for a group of teams.

    This provides a mathematically robust evaluation including:
    - Points Per Game (PPG): Standard ranking metric
    - Goal Differential Per Game (GD/G): Attacking/Defensive strength
    - Win Percentage (W%): Consistency and reliability
    - Draw Percentage (D%): Defensive stability
    - Goals For Per Game (GF/G): Attacking efficiency
    - Goals Against Per Game (GA/G): Defensive efficiency

    Args:
        df: DataFrame with columns: home_team, away_team, home_goals, away_goals
        team_list: List of teams to calculate metrics for

    Returns:
        Dictionary with performance metrics
    """
    home_games = df[df["home_team"].isin(team_list)].copy()
    away_games = df[df["away_team"].isin(team_list)].copy()

    total_games = len(home_games) + len(away_games)

    if total_games == 0:
        return {
            "ppg": 0.0,
            "gd_per_game": 0.0,
            "win_percentage": 0.0,
            "draw_percentage": 0.0,
            "goals_for_per_game": 0.0,
            "goals_against_per_game": 0.0,
            "games": 0,
        }

    # Points calculation
    home_points = (home_games["home_goals"] > home_games["away_goals"]).astype(int) * 3 + (
        home_games["home_goals"] == home_games["away_goals"]
    ).astype(int) * 1
    away_points = (away_games["away_goals"] > away_games["home_goals"]).astype(int) * 3 + (
        away_games["away_goals"] == away_games["home_goals"]
    ).astype(int) * 1

    # Wins, draws, losses
    home_wins = (home_games["home_goals"] > home_games["away_goals"]).sum()
    away_wins = (away_games["away_goals"] > away_games["home_goals"]).sum()
    total_wins = home_wins + away_wins

    home_draws = (home_games["home_goals"] == home_games["away_goals"]).sum()
    away_draws = (away_games["away_goals"] == away_games["home_goals"]).sum()
    total_draws = home_draws + away_draws

    # Goals calculation
    goals_for = home_games["home_goals"].sum() + away_games["away_goals"].sum()
    goals_against = home_games["away_goals"].sum() + away_games["home_goals"].sum()
    goal_differential = goals_for - goals_against

    # Calculate metrics
    total_points = home_points.sum() + away_points.sum()
    ppg = total_points / total_games
    gd_per_game = goal_differential / total_games
    win_percentage = (total_wins / total_games) * 100
    draw_percentage = (total_draws / total_games) * 100
    goals_for_per_game = goals_for / total_games
    goals_against_per_game = goals_against / total_games

    return {
        "ppg": ppg,
        "gd_per_game": gd_per_game,
        "win_percentage": win_percentage,
        "draw_percentage": draw_percentage,
        "goals_for_per_game": goals_for_per_game,
        "goals_against_per_game": goals_against_per_game,
        "games": total_games,
        "total_points": total_points,
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def _warn_if_teams_missing(df: pd.DataFrame, team_list: list[str], label: str) -> None:
    """
    Logs a warning for any team in team_list that does not appear at all in
    df. This catches silent name-mismatch issues (accents, abbreviations,
    club suffixes, etc.) that would otherwise make calculate_ppg return 0
    without any visible error.
    """
    known_teams = set(df["home_team"]).union(df["away_team"])
    missing_teams = [team for team in team_list if team not in known_teams]
    if missing_teams:
        logger.warning(f"{label} teams not found in the dataset (check spelling): {missing_teams}")


def calculate_caf_weight() -> float:
    """
    Calculates the CAF (African leagues) weight using Club World Cup data
    as a proxy, since no free/open CAF competition dataset is available.

    The weight is derived as the ratio between CAF teams' PPG and UEFA
    teams' PPG in the Club World Cup, scaled by UEFA's base weight in this
    model.

    Mathematically, uses vectorized operations and comprehensive metrics:
    weight = UEFA_BASE_WEIGHT * (CAF_PPG / UEFA_PPG)

    Also logs additional metrics (win %, goal differential) for validation.
    """
    logger.info("Loading Club World Cup data to calculate the CAF weight...")

    try:
        club_world_cup_df = pd.read_csv(CLUB_WORLD_CUP_DATA_PATH, encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"Club World Cup data not found at: {CLUB_WORLD_CUP_DATA_PATH}")
        logger.warning(f"Using fallback CAF weight: {CAF_FALLBACK_WEIGHT}")
        return CAF_FALLBACK_WEIGHT

    caf_teams = [
        "Al Ahly",
        "TP Mazembe",
        "Wydad Casablanca",
        "Raja Casablanca",
        "Mamelodi Sundowns",
        "Espérance de Tunis",
        "Étoile du Sahel",
        "ES Sétif",
    ]

    uefa_teams = [
        "Real Madrid",
        "Barcelona",
        "Bayern Munich",
        "Chelsea",
        "Liverpool",
        "Manchester City",
    ]

    _warn_if_teams_missing(club_world_cup_df, caf_teams, "CAF")
    _warn_if_teams_missing(club_world_cup_df, uefa_teams, "UEFA")

    # Calculate comprehensive metrics for both groups
    caf_metrics = calculate_performance_metrics(club_world_cup_df, caf_teams)
    uefa_metrics = calculate_performance_metrics(club_world_cup_df, uefa_teams)

    caf_ppg = caf_metrics["ppg"]
    uefa_ppg = uefa_metrics["ppg"]

    # Calculate weight with fallback for zero UEFA_PPG
    if uefa_ppg == 0:
        caf_weight = CAF_FALLBACK_WEIGHT
        logger.warning("UEFA PPG is 0. Using fallback CAF weight.")
    else:
        ratio = caf_ppg / uefa_ppg
        caf_weight = UEFA_BASE_WEIGHT * ratio

    # Log detailed metrics for validation
    logger.success(f"CAF Performance Metrics at Club World Cup:")
    logger.success(
        f"  PPG: {caf_ppg:.3f} | GD/G: {caf_metrics['gd_per_game']:.3f} | "
        f"Win%: {caf_metrics['win_percentage']:.1f}% | Games: {caf_metrics['games']}"
    )
    logger.success(f"UEFA Performance Metrics at Club World Cup:")
    logger.success(
        f"  PPG: {uefa_ppg:.3f} | GD/G: {uefa_metrics['gd_per_game']:.3f} | "
        f"Win%: {uefa_metrics['win_percentage']:.1f}% | Games: {uefa_metrics['games']}"
    )
    logger.success(f"CAF/UEFA PPG Ratio: {ratio:.3f}")
    logger.success(f"Dynamic calculated weight for leagues in Africa: {caf_weight:.3f}")

    return caf_weight


def extract_data() -> tuple[dict[str, pd.DataFrame], float]:
    """
    Extracts data from all global competitions (including the World Cup),
    spanning seasons 2020 to 2024 inclusive.

    The starting year is set to 2020, as it is the earliest year
    consistently present across all combined datasets.

    Returns:
        - Dictionary with competition names as keys and DataFrames as values
        - CAF weight coefficient for league strength calculations
    """
    logger.info("Starting FBref extraction for individual competitions (2020-2024)...")

    caf_weight = calculate_caf_weight()

    competition_data = {}

    # Extract data for each competition separately
    for competition in COMPETITIONS:
        logger.info(f"Extracting data for: {competition}")
        try:
            scraper: FBref = sd.FBref(leagues=[competition], seasons=range(2020, 2025))
            results_df = scraper.read_match_results()

            # Save to individual CSV file
            sanitized_name = competition.replace(" ", "_").replace("-", "_").lower()
            file_path = RAW_DATA_PATH / f"{sanitized_name}.csv"
            results_df.to_csv(file_path, index=False)

            competition_data[competition] = results_df
            logger.success(f"Saved {competition} to: {file_path}")

        except Exception as e:
            logger.error(f"Failed to extract data for {competition}: {str(e)}")
            continue

    logger.success(f"Data extraction complete. Saved {len(competition_data)} competitions.")
    return competition_data, caf_weight


if __name__ == "__main__":
    final_df, calculated_caf_weight = extract_data()
