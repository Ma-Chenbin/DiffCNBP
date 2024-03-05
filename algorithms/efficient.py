import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, sequence_len, num_patches, dim, transformer):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (q p) -> b q p', p = sequence_len),
            nn.Linear(sequence_len, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = transformer
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, sequence_len),
            Rearrange('b q p -> b (q p)', p = sequence_len) #reshape
        )

    def forward(self, data):
        x = self.to_patch_embedding(data)
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)  # transformer

        x = self.to_latent(x)
        return self.mlp_head(x[:,1:n+1])  # 线性输出