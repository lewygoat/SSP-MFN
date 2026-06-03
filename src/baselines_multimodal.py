"""多模态融合 SOTA 基线模型

MulT  — Multimodal Transformer (cross-modal attention)
TFN   — Tensor Fusion Network (outer product)
LMF   — Low-rank Multimodal Fusion (低秩张量分解)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# MulT: Multimodal Transformer
# ============================================================
class CrossModalAttention(nn.Module):
    """单向 cross-modal attention: query 来自 modality A, key/value 来自 modality B"""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        # query, key_value: [B, 1, D] (单 token)
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MulT(nn.Module):
    """Multimodal Transformer (simplified for tabular data)

    参考: Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences", ACL 2019
    适配: 每个模态投影为单 token，使用 cross-modal attention 融合
    """
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_out: int = 6, dropout: float = 0.2):
        super().__init__()
        self.proj_audio = nn.Sequential(nn.Linear(d_audio, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_meta = nn.Sequential(nn.Linear(d_meta, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.proj_part = nn.Sequential(nn.Linear(d_part, d_model), nn.LayerNorm(d_model), nn.GELU())

        # Cross-modal attention layers (6 directions)
        self.cm_a2m = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cm_a2p = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cm_m2a = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cm_m2p = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cm_p2a = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cm_p2m = nn.ModuleList([CrossModalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        # 融合后回归头
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_out),
        )

    def forward(self, x_audio, x_meta, x_part, ethnic_id=None):
        # 投影 → [B, 1, D]
        za = self.proj_audio(x_audio).unsqueeze(1)
        zm = self.proj_meta(x_meta).unsqueeze(1)
        zp = self.proj_part(x_part).unsqueeze(1)

        # Cross-modal attention (逐层)
        for layer_idx in range(len(self.cm_a2m)):
            za_new = za + self.cm_a2m[layer_idx](za, zm) + self.cm_a2p[layer_idx](za, zp)
            zm_new = zm + self.cm_m2a[layer_idx](zm, za) + self.cm_m2p[layer_idx](zm, zp)
            zp_new = zp + self.cm_p2a[layer_idx](zp, za) + self.cm_p2m[layer_idx](zp, zm)
            za, zm, zp = za_new, zm_new, zp_new

        # 拼接 → 回归
        fused = torch.cat([za.squeeze(1), zm.squeeze(1), zp.squeeze(1)], dim=-1)
        return self.head(fused)


# ============================================================
# TFN: Tensor Fusion Network
# ============================================================
class TFN(nn.Module):
    """Tensor Fusion Network

    参考: Zadeh et al., "Tensor Fusion Network for Multimodal Sentiment Analysis", EMNLP 2017
    外积融合: (z_a ⊕ 1) ⊗ (z_m ⊕ 1) ⊗ (z_p ⊕ 1)
    由于维度爆炸 (d+1)^3，先投影到低维
    """
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_hidden: int = 32, n_out: int = 6, dropout: float = 0.2):
        super().__init__()
        self.proj_audio = nn.Sequential(nn.Linear(d_audio, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_meta = nn.Sequential(nn.Linear(d_meta, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_part = nn.Sequential(nn.Linear(d_part, d_hidden), nn.ReLU(), nn.Dropout(dropout))

        # 外积后维度: (d_hidden+1)^3 太大，用两两外积 + 拼接
        # audio⊗meta: (d_hidden+1)^2, audio⊗part: (d_hidden+1)^2, meta⊗part: (d_hidden+1)^2
        fusion_dim = 3 * (d_hidden + 1) ** 2
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, n_out),
        )

    def forward(self, x_audio, x_meta, x_part, ethnic_id=None):
        za = self.proj_audio(x_audio)
        zm = self.proj_meta(x_meta)
        zp = self.proj_part(x_part)

        # 追加 bias 维度
        B = za.size(0)
        ones = torch.ones(B, 1, device=za.device)
        za1 = torch.cat([za, ones], dim=-1)  # [B, d+1]
        zm1 = torch.cat([zm, ones], dim=-1)
        zp1 = torch.cat([zp, ones], dim=-1)

        # 两两外积
        am = torch.bmm(za1.unsqueeze(2), zm1.unsqueeze(1)).view(B, -1)  # [B, (d+1)^2]
        ap = torch.bmm(za1.unsqueeze(2), zp1.unsqueeze(1)).view(B, -1)
        mp = torch.bmm(zm1.unsqueeze(2), zp1.unsqueeze(1)).view(B, -1)

        fused = torch.cat([am, ap, mp], dim=-1)
        return self.head(fused)


# ============================================================
# LMF: Low-rank Multimodal Fusion
# ============================================================
class LMF(nn.Module):
    """Low-rank Multimodal Fusion

    参考: Liu et al., "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors", ACL 2018
    用低秩因子分解近似张量融合，避免维度爆炸
    """
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_hidden: int = 64, rank: int = 4, n_out: int = 6, dropout: float = 0.2):
        super().__init__()
        self.rank = rank
        self.n_out = n_out

        self.proj_audio = nn.Sequential(nn.Linear(d_audio, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_meta = nn.Sequential(nn.Linear(d_meta, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.proj_part = nn.Sequential(nn.Linear(d_part, d_hidden), nn.ReLU(), nn.Dropout(dropout))

        # 低秩因子: 每个模态一个 [d_hidden+1, rank*n_out]
        self.factor_audio = nn.Parameter(torch.randn(rank * n_out, d_hidden + 1) * 0.02)
        self.factor_meta = nn.Parameter(torch.randn(rank * n_out, d_hidden + 1) * 0.02)
        self.factor_part = nn.Parameter(torch.randn(rank * n_out, d_hidden + 1) * 0.02)
        self.fusion_bias = nn.Parameter(torch.zeros(n_out))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_audio, x_meta, x_part, ethnic_id=None):
        za = self.proj_audio(x_audio)
        zm = self.proj_meta(x_meta)
        zp = self.proj_part(x_part)

        B = za.size(0)
        ones = torch.ones(B, 1, device=za.device)
        za1 = torch.cat([za, ones], dim=-1)  # [B, d+1]
        zm1 = torch.cat([zm, ones], dim=-1)
        zp1 = torch.cat([zp, ones], dim=-1)

        # 低秩融合: output_k = sum_r (fa_r^T za) * (fm_r^T zm) * (fp_r^T zp)
        # factor: [rank*n_out, d+1] → reshape [n_out, rank, d+1]
        fa = self.factor_audio.view(self.n_out, self.rank, -1)  # [K, R, d+1]
        fm = self.factor_meta.view(self.n_out, self.rank, -1)
        fp = self.factor_part.view(self.n_out, self.rank, -1)

        # [B, d+1] x [K, R, d+1]^T → [B, K, R]
        ha = torch.einsum('bd,krd->bkr', za1, fa)
        hm = torch.einsum('bd,krd->bkr', zm1, fm)
        hp = torch.einsum('bd,krd->bkr', zp1, fp)

        # element-wise product + sum over rank
        fusion = (ha * hm * hp).sum(dim=-1)  # [B, K]
        output = self.dropout(fusion) + self.fusion_bias
        return output
