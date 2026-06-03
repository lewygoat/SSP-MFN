"""CMAF-Net · Cross-cultural Modal Attention Fusion Network

Plan-A 四创新点的 PyTorch 主模型实现
  1) 三路模态 Linear 投影 (Audio / Text / Tabular)
  2) 单向跨模态注意力 (Q=tabular, K=V=concat(all))
  3) Cultural Adaptive Instance Normalization (C-AdaIN)
  4) 六维量表 post 回归头

约定的输入形状
  x_audio    [B, D_a]   默认 D_a = 16
  x_text     [B, D_t]   默认 D_t = 32
  x_tab      [B, D_s]   默认 D_s = 17
  ethnic_id  [B]        LongTensor, 0/1/2 对应 Dong/Tibetan/Mongolian
返回
  y_hat      [B, 6]     ICS IRI CSAS SSCS IOS SCI2 post 总分
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalProjector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, p_drop: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.proj(x)


class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, p_drop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, q, kv, return_attn: bool = False):
        B = q.size(0)
        Nq = q.size(1) if q.dim() == 3 else 1
        Nk = kv.size(1)

        if q.dim() == 2:
            q = q.unsqueeze(1)

        Q = self.wq(q).view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.wk(kv).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.wv(kv).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        out = self.wo(out)

        if Nq == 1:
            out = out.squeeze(1)
        if return_attn:
            return out, attn
        return out


class CulturalAdaIN(nn.Module):
    """γ/β 由 ethnic embedding 动态生成, 对融合特征按样本内统计量做条件归一化。"""

    def __init__(self, feat_dim: int, n_ethnic: int = 3, emb_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_ethnic, emb_dim)
        self.to_affine = nn.Sequential(
            nn.Linear(emb_dim, feat_dim * 2),
        )

    def forward(self, h, ethnic_id):
        mu = h.mean(dim=-1, keepdim=True)
        sigma = h.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        h_norm = (h - mu) / sigma

        e = self.embed(ethnic_id)
        gamma_beta = self.to_affine(e)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * h_norm + beta


class CMAFNet(nn.Module):
    def __init__(
        self,
        d_audio: int = 16,
        d_text: int = 32,
        d_tab: int = 17,
        d_model: int = 64,
        n_heads: int = 4,
        n_ethnic: int = 3,
        n_scales: int = 6,
        p_drop: float = 0.1,
        use_adain: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_adain = use_adain
        self.use_attention = use_attention

        self.proj_a = ModalProjector(d_audio, d_model, p_drop)
        self.proj_t = ModalProjector(d_text, d_model, p_drop)
        self.proj_s = ModalProjector(d_tab, d_model, p_drop)

        self.attn1 = CrossModalAttention(d_model, n_heads, p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.attn2 = CrossModalAttention(d_model, n_heads, p_drop)
        self.norm3 = nn.LayerNorm(d_model)

        self.adain = CulturalAdaIN(d_model, n_ethnic=n_ethnic, emb_dim=8)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, n_scales),
        )

    def forward(self, x_audio, x_text, x_tab, ethnic_id, return_attn: bool = False):
        ha = self.proj_a(x_audio)
        ht = self.proj_t(x_text)
        hs = self.proj_s(x_tab)

        kv = torch.stack([ha, ht, hs], dim=1)
        q = hs

        if self.use_attention:
            fused, attn_w = self.attn1(q, kv, return_attn=True)
            fused = self.norm1(fused + q)
            fused = self.norm2(fused + self.ffn1(fused))
            fused2 = self.attn2(fused, kv)
            fused = self.norm3(fused + fused2)
        else:
            fused = (ha + ht + hs) / 3.0
            attn_w = None

        if self.use_adain:
            fused = self.adain(fused, ethnic_id)

        y_hat = self.head(fused)
        if return_attn:
            return y_hat, attn_w
        return y_hat


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    B = 8
    model = CMAFNet()
    xa = torch.randn(B, 16)
    xt = torch.randn(B, 32)
    xs = torch.randn(B, 17)
    eid = torch.randint(0, 3, (B,))
    y, attn = model(xa, xt, xs, eid, return_attn=True)
    print(f"output shape = {tuple(y.shape)}")
    print(f"attn shape   = {tuple(attn.shape)}")
    print(f"params       = {count_params(model):,}")
