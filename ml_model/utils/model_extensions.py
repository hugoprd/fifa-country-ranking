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

    def forward(self, x, synergy_mask):
        # x shape: (Batch, Num_Players, Embed_Dim)
        # synergy_mask shape: (Batch, Num_Players, Num_Players)

        # PyTorch requires the 3D mask to be shaped as (Batch * Num_Heads, Seq_Len, Seq_Len)
        # it's needed to repeat the mask along the first dimension for each attention head
        # .repeat_interleave copies the mask for each head sequentially
        adjusted_mask = synergy_mask.repeat_interleave(self.num_heads, dim=0)

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
        self.countries = self.df_target["national_team"].unique()

    def __len__(self):
        return len(self.countries)

    def __getitem__(self, idx):
        country = self.countries[idx]

        #### ==========================================
        # 1. TAKE PLAYERS (INDIVIDUAL TENSOR)
        #### ==========================================
        country_players = self.df_ind[self.df_ind["national_team"] == country].copy()

        # ordain by best player in a country (based on total_weighted_wins)
        country_players = country_players.sort_values(by="total_weighted_wins", ascending=False)

        # gets the "Top 11". if the country has less than 11 players, "padding" is needed (fill with zeros)
        top_players = country_players.head(self.top_k)

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

        return player_tensor, synergy_tensor, target
