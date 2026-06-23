from loguru import logger

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

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
        ### LOAD DATA ###
        self.df_ind = pd.read_csv(individual_path)
        self.df_pairs = pd.read_csv(pairs_path)
        self.df_target = pd.read_csv(ranking_path)

        self.top_k = top_k_players

        # the model will receive these attributes of the player (can adjust if added more columns)
        self.feature_cols = ["total_matches", "total_weighted_wins", "total_weighted_plus_minus", "win_rate_percentage"]

        # build the list of available countries
        all_target_countries = self.df_target["national_team"].unique()

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
            player_tensor[i] = torch.tensor([row[col] for col in self.feature_cols], dtype=torch.float32)
            player_id_to_idx[row["player_id"]] = i

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
                score = row["total_weighted_wins"] / 10.0

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
