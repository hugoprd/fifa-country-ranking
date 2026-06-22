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


def _sinergy_aggregation(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the synergy points for each patriotic pair across all their shared matches,
    applying the confederation weight, that returns the synergy_matrix with the following columns:
    - player_id_A: ID of the first player in the pair.
    - name_A: Name of the first player.
    - player_id_B: ID of the second player in the pair.
    - name_B: Name of the second player.
    - national_team: The national team (country of citizenship) of the pair.
    - matches_played_together: The total number of matches the pair played together.
    - total_synergy_score: The sum of the synergy points for all matches they played together.
    - total_combined_goals_assists: The total number of goals and assists combined for the pair across all their shared matches.
    """
    logger.info("[ REFINE DATA | SINERGY AGGREGATION ] Consolideing metrics of sinergy and strength...")

    # applies the difficult weight on wins and plus-minus to give more importance to players from stronger confederations
    df_pairs["weighted_win"] = df_pairs["is_win"] * df_pairs["confederation_weight_A"]
    df_pairs["weighted_plus_minus"] = df_pairs["plus_minus"] * df_pairs["confederation_weight_A"]

    synergy_matrix = (
        df_pairs.groupby(["player_id_A", "name_A", "player_id_B", "name_B", "country_of_citizenship_A"])
        .agg(
            matches_played_together=("game_id", "count"),
            total_wins=("is_win", "sum"),
            total_weighted_wins=("weighted_win", "sum"),
            total_weighted_plus_minus=("weighted_plus_minus", "sum"),
        )
        .reset_index()
    )

    # creation of ML attributes
    synergy_matrix["win_rate_percentage"] = (synergy_matrix["total_wins"] / synergy_matrix["matches_played_together"]) * 100
    synergy_matrix = synergy_matrix.rename(columns={"country_of_citizenship_A": "national_team"})

    # relevance filter: only matters if they played together at least 5 times
    synergy_matrix = synergy_matrix[synergy_matrix["matches_played_together"] >= 5]
    synergy_matrix = synergy_matrix.sort_values(by="total_weighted_wins", ascending=False)

    return synergy_matrix


def _cross_join_sinergy(app_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Cross joins the enriched appearances data to create pairs of players who played together in the same game and club,
    then calculates the synergy score for patriotic pairs and aggregates it to create the final synergy matrix.
    """
    logger.info("[ REFINE DATA | CROSS JOIN SINERGY ] Realizing analysis of networks (Graphs) of patriotic pairs...")

    # filters columns to not explode the RAM
    cols_to_keep = [
        "game_id",
        "player_club_id",
        "player_id",
        "name",
        "country_of_citizenship",
        "confederation_weight",
        "is_win",
        "plus_minus",
    ]

    app_reduced = app_enriched[cols_to_keep]

    # self-merge: who played with whom on the same team and in the same game?
    df_pairs = app_reduced.merge(
        app_reduced, on=["game_id", "player_club_id", "is_win", "plus_minus"], suffixes=("_A", "_B")  # they share the score
    )

    # optimization filters: only patriotic pairs and without inverse duplication
    # (A with B is the same as B with A) and self-matches (A with A)
    df_pairs = df_pairs[df_pairs["player_id_A"] < df_pairs["player_id_B"]]
    df_pairs = df_pairs[df_pairs["country_of_citizenship_A"] == df_pairs["country_of_citizenship_B"]]

    if df_pairs.empty:
        logger.warning("[ REFINE DATA | CROSS JOIN SINERGY ] No patriotic pairs found.")
        return

    return df_pairs


def _calculate_winrate_plus_minus(
    df_players: pd.DataFrame, df_appearances: pd.DataFrame, df_games: pd.DataFrame
) -> pd.DataFrame:
    """
    This function is responsible for calculating the win rate and plus-minus for each player based on their
    appearances and the game results.

    The logic is based on the idea that a player's contribution to a match can be measured by whether their team won or lost
    and not only based on the player's contribution in terms of goals and assists, but also by the overall team performance
    when they were on the field.
    """
    logger.info("[ REFINE DATA | CALCULATE WINRATE PLUS MINUS ] Calculating results (Win Rate / Plus-Minus) per appearance...")

    # cross join the appearances with the games
    app_games = df_appearances.merge(df_games, on="game_id", how="inner")

    # discover if the player was at home or away
    is_home = app_games["player_club_id"] == app_games["home_club_id"]
    is_away = app_games["player_club_id"] == app_games["away_club_id"]

    # get the goals for and against based on the player's team position
    app_games.loc[is_home, "team_goals"] = app_games.loc[is_home, "home_club_goals"]
    app_games.loc[is_home, "opponent_goals"] = app_games.loc[is_home, "away_club_goals"]

    app_games.loc[is_away, "team_goals"] = app_games.loc[is_away, "away_club_goals"]
    app_games.loc[is_away, "opponent_goals"] = app_games.loc[is_away, "home_club_goals"]

    # remove any anomalies
    app_games = app_games.dropna(subset=["team_goals", "opponent_goals"])

    # the mathematics of victory and plus/minus
    app_games["plus_minus"] = app_games["team_goals"] - app_games["opponent_goals"]
    app_games["is_win"] = (app_games["plus_minus"] > 0).astype(int)
    app_games["is_draw"] = (app_games["plus_minus"] == 0).astype(int)

    # bring the player's data (Name, Country, Confederation Weight)
    app_enriched = app_games.merge(
        df_players[["player_id", "name", "country_of_citizenship", "confederation_weight"]], on="player_id", how="inner"
    )

    return app_enriched


def _calculate_individual_efficiency(app_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the individual efficiency of each player based on universal metrics
    (Win Rate and Plus/Minus), weighted by the difficulty of their confederation.
    """
    logger.info("[ REFINE DATA | CALCULATE INDIVIDUAL EFFICIENCY ] Calculating individual efficiency weighted...")

    # applies the confederation weight at each game individually
    app_enriched["weighted_win"] = app_enriched["is_win"] * app_enriched["confederation_weight"]
    app_enriched["weighted_plus_minus"] = app_enriched["plus_minus"] * app_enriched["confederation_weight"]

    # group by player
    ind_matrix = (
        app_enriched.groupby(["player_id", "name", "country_of_citizenship", "confederation_weight"])
        .agg(
            total_matches=("game_id", "count"),
            total_wins=("is_win", "sum"),
            total_weighted_wins=("weighted_win", "sum"),
            raw_plus_minus=("plus_minus", "sum"),
            total_weighted_plus_minus=("weighted_plus_minus", "sum"),
        )
        .reset_index()
    )

    # creation of final metrics
    ind_matrix["win_rate_percentage"] = (ind_matrix["total_wins"] / ind_matrix["total_matches"]) * 100

    # renames columns to standardize with the synergy table
    ind_matrix = ind_matrix.rename(columns={"country_of_citizenship": "national_team"})

    # relevance filter: remove players with very few games (noise)
    # e.g., a player enters for 1 minute and their team wins, they would have 100% win rate.
    ind_matrix = ind_matrix[ind_matrix["total_matches"] >= 10]

    # sort by the best in the world according to their mathematics
    ind_matrix = ind_matrix.sort_values(by="total_weighted_wins", ascending=False)

    return ind_matrix


def refine_data():
    """
    Transforms enriched player data into final Machine Learning features:
    Focuses on National Team Synergy (Chemistry between compatriots at the club level).
    """
    logger.info("[ REFINE DATA ] Starting Feature Engineering for Machine Learning...")

    players_file = PROCESSED_DATA_PATH / "processed_players.csv"
    appearances_file = PROCESSED_DATA_PATH / "processed_appearances.csv"
    games_file = PROCESSED_DATA_PATH / "processed_games.csv"

    if not all(p.exists() for p in [players_file, appearances_file, games_file]):
        logger.error("[ REFINE DATA ] Processed files not found! Run process_data.py first.")

        return

    df_players = pd.read_csv(players_file)
    df_appearances = pd.read_csv(appearances_file)
    df_games = pd.read_csv(games_file)

    # forceing types for merge
    df_appearances["game_id"] = df_appearances["game_id"].astype(str)
    df_games["game_id"] = df_games["game_id"].astype(str)
    df_appearances["player_club_id"] = df_appearances["player_club_id"].astype(str).str.split(".").str[0]
    df_games["home_club_id"] = df_games["home_club_id"].astype(str).str.split(".").str[0]
    df_games["away_club_id"] = df_games["away_club_id"].astype(str).str.split(".").str[0]

    #### =========================================================
    # 1. ENRICHEING THE GAME WITH VICTORY AND PLUS/MINUS
    #### =========================================================
    app_enriched = _calculate_winrate_plus_minus(df_players, df_appearances, df_games)

    #### =========================================================
    # 2. INDIVIDUAL EFFICIENCY
    #### =========================================================
    df_individual = _calculate_individual_efficiency(app_enriched)

    out_individual_file = REFINED_DATA_PATH / "ml_individual_features.csv"
    df_individual.to_csv(out_individual_file, index=False)
    logger.success(f"[ REFINE DATA ] Individual efficiency saved at: {out_individual_file}")

    #### =========================================================
    # 3. THE SINERGY CROSS JOIN (THE SYNERGY JOIN A AND B)
    #### =========================================================
    df_pairs = _cross_join_sinergy(app_enriched)

    #### =========================================================
    # 4. FINAL AGGREGATION (TEAM SINERGY)
    #### =========================================================
    synergy_matrix = _sinergy_aggregation(df_pairs)

    output_file = REFINED_DATA_PATH / "ml_national_synergy_features.csv"
    synergy_matrix.to_csv(output_file, index=False)

    logger.success(f"[ REFINE DATA ] Sinergy calculated with success. {len(synergy_matrix)} partnerships saved.")
    logger.info(f"[ REFINE DATA ] File saved at: {output_file}")


if __name__ == "__main__":
    refine_data()
