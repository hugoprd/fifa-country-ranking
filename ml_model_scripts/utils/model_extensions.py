from loguru import logger

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import itertools

"""
File that it's going to be used to create custom layers or attention mechanisms that can be used in the model.
"""


class SynergyAttention(nn.Module):
    """
    A custom Multi-Head Attention layer that accepts a Synergy Matrix
    as an attention bias.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, synergy_mask, key_padding_mask=None):
        # x shape: (Batch, Num_Players, Embed_Dim)
        # synergy_mask shape: (Batch, Num_Players, Num_Players)
        # key_padding_mask shape: (Batch, Num_Players), True = ignore this player (padding)

        combined_mask = synergy_mask
        if key_padding_mask is not None:
            # merge the bool padding mask into the float synergy bias ourselves, instead of
            # passing both to nn.MultiheadAttention. mixing a float attn_mask with a bool
            # key_padding_mask goes through PyTorch's deprecated merge path, which produced
            # NaN losses in practice. using one unified float mask avoids that path entirely
            pad_bias = torch.zeros_like(synergy_mask).masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
            combined_mask = synergy_mask + pad_bias

        # PyTorch requires the 3D mask to be shaped as (Batch * Num_Heads, Seq_Len, Seq_Len)
        # it's needed to repeat the mask along the first dimension for each attention head
        # .repeat_interleave copies the mask for each head sequentially
        adjusted_mask = combined_mask.repeat_interleave(self.num_heads, dim=0)

        attn_output, _ = self.attention(query=x, key=x, value=x, attn_mask=adjusted_mask, is_causal=False)

        return attn_output


class FIFANationalTeamDataset(Dataset):
    """
    Construct the tensors for training from the refined CSV files.
    For each country, select the top N players and build the synergy matrix between them.
    """

    def __init__(self, individual_path, pairs_path, ranking_path, top_k_players=11):
        self.df_ind = pd.read_csv(individual_path)
        self.df_pairs = pd.read_csv(pairs_path)
        self.df_target = pd.read_csv(ranking_path)

        self.top_k = top_k_players
        self.feature_cols = ["total_matches", "total_weighted_wins", "total_weighted_plus_minus", "win_rate_percentage"]

        # global synergy normalization
        global_std = self.df_pairs["total_weighted_wins"].std()
        self.synergy_std = float(global_std) if global_std > 0 else 1.0

        # build the list of available countries
        all_target_countries = self.df_target["national_team"].unique()

        # global normalization of the individual features
        feat_df = self.df_ind[self.feature_cols]
        self.feat_mean = feat_df.mean().values.astype("float32")  # shape (4,)
        self.feat_std = feat_df.std().values.astype("float32")
        self.feat_std[self.feat_std == 0] = 1.0

        # a country with ZERO rows in df_ind would get a 100%-padding mask, i.e. an
        # attention row that is -inf in every column. softmax of an all -inf row is
        # undefined (NaN), and that single NaN poisons every model weight forever via
        # the optimizer step - this is what was causing the permanent NaN loss.
        # this mismatch usually means the country name differs between data sources
        # (e.g. "USA" vs "United States"); fixing the source data is the real fix,
        # this just keeps training from breaking until you do.
        countries_with_players = set(self.df_ind["national_team"].unique())
        missing_countries = [c for c in all_target_countries if c not in countries_with_players]
        if missing_countries:
            logger.warning(
                "[ MODEL EXTENSIONS | FIFA NATIONAL TEAM DATASET ] Excluding countries with no "
                f"individual player data: {missing_countries}"
            )
        self.countries = [c for c in all_target_countries if c in countries_with_players]

        # pre-build every country's tensors ONCE. This data is static (it doesn't depend on
        # model weights), so recomputing it via DataFrame filtering + .iterrows() inside
        # __getitem__ meant redoing the same slow pandas work on every single epoch
        # (112 countries x 150 epochs) - that's was causing a multi-minute hang
        self._cache = [self._build_sample(country) for country in self.countries]

    def __len__(self):
        return len(self.countries)

    def __getitem__(self, idx):
        return self._cache[idx]

    def _build_sample(self, country):
        #### ==========================================
        # 1. TAKE PLAYERS (INDIVIDUAL TENSOR)
        #### ==========================================
        country_players = self.df_ind[self.df_ind["national_team"] == country].copy()

        # ordain by best player in a country (based on total_weighted_wins)
        country_players = country_players.sort_values(by="total_weighted_wins", ascending=False)

        # gets the "Top 11". if the country has less than 11 players, "padding" is needed (fill with zeros)
        top_players = country_players.head(self.top_k)
        num_real_players = len(top_players)

        # True = padded slot (no real player). Used to exclude "ghost" players from
        # attention (key_padding_mask) and from the final pooling average.
        padding_mask = torch.ones(self.top_k, dtype=torch.bool)
        padding_mask[:num_real_players] = False

        # prepair matix [11, Num_Features]
        num_features = len(self.feature_cols)
        player_tensor = torch.zeros((self.top_k, num_features), dtype=torch.float32)

        # maps to know who is in which index (0 to 10) for building the synergy matrix later
        player_id_to_idx = {}

        for i, (_, row) in enumerate(top_players.iterrows()):
            raw = torch.tensor([row[col] for col in self.feature_cols], dtype=torch.float32)
            player_tensor[i] = (raw - torch.tensor(self.feat_mean)) / torch.tensor(self.feat_std)

        #### ==========================================
        # 2. BUILDING THE SYNERGY MATRIX (ATTENTION)
        #### ==========================================
        synergy_tensor = torch.zeros((self.top_k, self.top_k), dtype=torch.float32)

        # filter the pairs only for the current country
        country_pairs = self.df_pairs[self.df_pairs["national_team"] == country]

        for _, row in country_pairs.iterrows():
            id_a = row["player_id_A"]
            id_b = row["player_id_B"]

            # just add the synergy if BOTH players are in the "Top 11" selected
            if id_a in player_id_to_idx and id_b in player_id_to_idx:
                idx_a = player_id_to_idx[id_a]
                idx_b = player_id_to_idx[id_b]

                # add a little value to normalize/scale the synergy points
                # (so the model doesn't get too crazy with high values)
                score = row["total_weighted_wins"] / (self.synergy_std + 1e-6)

                # fills the symmetric matrix 11x11
                synergy_tensor[idx_a, idx_b] = score
                synergy_tensor[idx_b, idx_a] = score

        #### ==========================================
        # 3. GET THE TARGET (score to predict)
        #### ==========================================
        target_row = self.df_target[self.df_target["national_team"] == country].iloc[0]
        # here it's used the 'avg_synergy_score' that was calculated as the objective for the model to learn
        # in the future, can change this column
        target = torch.tensor([target_row["avg_synergy_score"]], dtype=torch.float32)

        return player_tensor, synergy_tensor, padding_mask, target


class TacticalOptimizer:
    """
    Simulates the tatical search and perfect escalation of a national team,
    respecting the football position rules
    """

    def __init__(self, trained_model):
        self.model = trained_model

        # Dicionário definindo a quantidade exata de posições por tática
        self.tactics_map = {
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "ATT": 3},
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "ATT": 2},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "ATT": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "ATT": 1},
        }

    def find_best_lineup(self, df_country_squad: pd.DataFrame):
        """
        Receives a DataFrame withh all players of a unique national team (e.g.: 23 players).
        The table have to have the column 'position' (GK, DEF, MID, ATT) and the features/ID column
        """
        best_score = -float("inf")
        best_tactic = None
        best_11 = None

        # groups the available players by position
        gk_pool = df_country_squad[df_country_squad["position"] == "GK"]
        def_pool = df_country_squad[df_country_squad["position"] == "DEF"]
        mid_pool = df_country_squad[df_country_squad["position"] == "MID"]
        att_pool = df_country_squad[df_country_squad["position"] == "ATT"]

        # tests each possible tatic
        for tactic_name, req in self.tactics_map.items():
            # generates all possible combinations for this tatic
            gk_combs = list(itertools.combinations(gk_pool.index, req["GK"]))
            def_combs = list(itertools.combinations(def_pool.index, req["DEF"]))
            mid_combs = list(itertools.combinations(mid_pool.index, req["MID"]))
            att_combs = list(itertools.combinations(att_pool.index, req["ATT"]))

            # cartesian product (tests all the possible defenses with all possible means
            for g, d, m, a in itertools.product(gk_combs, def_combs, mid_combs, att_combs):
                # 11 players of this especific combination
                lineup_indices = list(g + d + m + a)
                df_lineup = df_country_squad.loc[lineup_indices]

                # 1. converts the features of the 11 players in Tensor
                # 2. rebuild the synergy matrix to only these 11
                players_tensor, synergy_matrix = self._prepare_tensors(df_lineup)

                with torch.no_grad():
                    # Batch=1, no padding
                    score = self.model(players_tensor.unsqueeze(0), synergy_matrix.unsqueeze(0))
                    current_score = score.item()

                # if the model point for this team be bigger than the best point saved until now
                if current_score > best_score:
                    best_score = current_score
                    best_tactic = tactic_name
                    best_11 = df_lineup["player_name"].tolist()

        return best_tactic, best_score, best_11

    def _prepare_tensors(self, df_lineup):
        """Função fictícia: Você precisa transformar o DataFrame filtrado de volta em Tensor"""
        # players_tensor = torch.tensor(df_lineup[['feat1', 'feat2'...]].values, dtype=torch.float32)
        # synergy_matrix = create_synergy_graph(df_lineup)
        # return players_tensor, synergy_matrix
        pass
