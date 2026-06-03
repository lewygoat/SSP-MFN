"""SSP-MFN · Social Skill Prediction via Modal-gated Fusion Network

大纲 §4.3 的 PyTorch 实现（策略1: 3路模态）

架构:
  1) 三路模态投影: Audio(MERT+手工→128) / Meta(文化元多热→128) / Part(参与者+pre→128)
  2) 维度级门控注意力融合: 每个社会技能维度 k 独立计算 α_{k,j} 和 g_{k,j}
  3) Cultural AdaIN: 民族 ID 条件归一化
  4) 六维回归头: Δy{adj} 协方差调整残差

门控融合公式 (大纲 §4.3):
  z_j = Proj(h_j)                              投影到统一维度
  a_{k,j} = w_k^T tanh(W_k z_j + b_k)         注意力得分
  α_{k,j} = softmax_j(a_{k,j})                 概率分布
  g_{k,j} = sigmoid(u_k^T z_j + c_k)           逐模态通断门控
  f_k = Σ_j α_{k,j} · g_{k,j} · z_j           维度 k 的融合表示

输入:
  x_audio    [B, D_a]   MERT 768 + 手工 30 → 投影后 128
  x_meta     [B, D_m]   文化元多热编码
  x_part     [B, D_p]   参与者人口学 + 6维 pre 量表
  ethnic_id  [B]        LongTensor, 0/1/2

输出:
  y_hat      [B, 6]     六维 Δy{adj} 预测
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalProjector(nn.Module):
    """两层 MLP 投影到统一隐藏维度"""
    def __init__(self, in_dim: int, hidden: int, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionLayer(nn.Module):
    """大纲 §4.3 维度级门控注意力融合

    对每个社会技能维度 k (共 n_scales 个):
      a_{k,j} = w_k^T tanh(W_k z_j + b_k)
      α_{k,j} = softmax_j(a_{k,j})
      g_{k,j} = sigmoid(u_k^T z_j + c_k)
      f_k = Σ_j α_{k,j} · g_{k,j} · z_j
    """
    def __init__(self, d_model: int, n_modalities: int, n_scales: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.n_scales = n_scales

        # 注意力参数: W_k [n_scales, d_model, d_model], w_k [n_scales, d_model]
        self.W_attn = nn.Parameter(torch.randn(n_scales, d_model, d_model) * 0.02)
        self.b_attn = nn.Parameter(torch.zeros(n_scales, d_model))
        self.w_attn = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)

        # 门控参数: u_k [n_scales, d_model], c_k [n_scales]
        self.u_gate = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.c_gate = nn.Parameter(torch.zeros(n_scales))

    def forward(self, z_list: list[torch.Tensor], return_weights: bool = False):
        """
        z_list: list of [B, d_model], length = n_modalities
        returns: [B, n_scales, d_model] 融合表示
        """
        B = z_list[0].size(0)
        # Stack modalities: [B, J, D]
        Z = torch.stack(z_list, dim=1)  # [B, J, D]

        # 计算注意力得分 a_{k,j}
        # Z: [B, J, D] -> expand for n_scales
        # W_k z_j + b_k: [K, D, D] @ [B, J, D]^T -> need einsum
        # a_{k,j} = w_k^T tanh(W_k z_j + b_k)
        # [B, J, D] x [K, D, D] -> [B, K, J, D]
        Wz = torch.einsum('bjd,kdo->bkjo', Z, self.W_attn)  # [B, K, J, D]
        Wz = Wz + self.b_attn[None, :, None, :]  # broadcast b_k
        Wz = torch.tanh(Wz)
        # w_k^T (tanh result): [B, K, J, D] x [K, D] -> [B, K, J]
        a = torch.einsum('bkjd,kd->bkj', Wz, self.w_attn)  # [B, K, J]

        # α_{k,j} = softmax over j
        alpha = F.softmax(a, dim=-1)  # [B, K, J]

        # 门控 g_{k,j} = sigmoid(u_k^T z_j + c_k)
        # [B, J, D] x [K, D] -> [B, K, J]
        g_score = torch.einsum('bjd,kd->bkj', Z, self.u_gate)  # [B, K, J]
        g_score = g_score + self.c_gate[None, :, None]  # [B, K, J]
        g = torch.sigmoid(g_score)  # [B, K, J]

        # f_k = Σ_j α_{k,j} · g_{k,j} · z_j
        # weights: [B, K, J] * [B, K, J] = [B, K, J]
        weights = alpha * g  # [B, K, J]
        # [B, K, J] x [B, J, D] -> [B, K, D]
        f = torch.einsum('bkj,bjd->bkd', weights, Z)  # [B, K, D]

        if return_weights:
            return f, alpha, g
        return f


class CulturalAdaIN(nn.Module):
    """民族 ID 条件归一化: γ/β 由 ethnic embedding 动态生成"""
    def __init__(self, feat_dim: int, n_ethnic: int = 3, emb_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_ethnic, emb_dim)
        self.to_affine = nn.Linear(emb_dim, feat_dim * 2)

    def forward(self, h, ethnic_id):
        mu = h.mean(dim=-1, keepdim=True)
        sigma = h.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        h_norm = (h - mu) / sigma
        e = self.embed(ethnic_id)
        gamma_beta = self.to_affine(e)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * h_norm + beta


class SSPMFN(nn.Module):
    """SSP-MFN: 3路门控融合 + Cultural AdaIN + 六维回归头

    大纲 §4.3 完整实现
    """
    def __init__(
        self,
        d_audio: int = 30,
        d_meta: int = 12,
        d_part: int = 17,
        d_model: int = 128,
        n_ethnic: int = 3,
        n_scales: int = 6,
        p_drop: float = 0.2,
        use_adain: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.use_adain = use_adain
        self.use_gate = use_gate

        # 三路投影编码器
        self.proj_audio = ModalProjector(d_audio, d_model, p_drop)
        self.proj_meta = ModalProjector(d_meta, d_model, p_drop)
        self.proj_part = ModalProjector(d_part, d_model, p_drop)

        # 门控融合层
        self.fusion = GatedFusionLayer(d_model, n_modalities=3, n_scales=n_scales)

        # Cultural AdaIN (对每个维度的融合表示)
        self.adain = CulturalAdaIN(d_model, n_ethnic=n_ethnic, emb_dim=8)

        # 六维回归头: 每个维度 k 从 f_k (d_model维) 映射到 1 维
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_scales)
        ])

    def forward(self, x_audio, x_meta, x_part, ethnic_id,
                return_weights: bool = False, mask: dict | None = None):
        """
        mask: 可选，用于消融实验。如 {'audio': False, 'meta': True, 'part': True}
              False 表示该模态被 zero-out
        """
        # 投影
        z_audio = self.proj_audio(x_audio)
        z_meta = self.proj_meta(x_meta)
        z_part = self.proj_part(x_part)

        # 消融 mask
        if mask is not None:
            if not mask.get('audio', True):
                z_audio = torch.zeros_like(z_audio)
            if not mask.get('meta', True):
                z_meta = torch.zeros_like(z_meta)
            if not mask.get('part', True):
                z_part = torch.zeros_like(z_part)

        z_list = [z_audio, z_meta, z_part]

        # 门控融合
        if self.use_gate:
            if return_weights:
                f, alpha, g = self.fusion(z_list, return_weights=True)
            else:
                f = self.fusion(z_list)
                alpha, g = None, None
        else:
            # 消融: 简单均值融合
            z_mean = torch.stack(z_list, dim=1).mean(dim=1)  # [B, D]
            f = z_mean.unsqueeze(1).expand(-1, self.n_scales, -1)  # [B, K, D]
            alpha, g = None, None

        # Cultural AdaIN (对每个维度)
        # f: [B, K, D]
        preds = []
        for k in range(self.n_scales):
            fk = f[:, k, :]  # [B, D]
            if self.use_adain:
                fk = self.adain(fk, ethnic_id)
            preds.append(self.heads[k](fk))  # [B, 1]

        y_hat = torch.cat(preds, dim=-1)  # [B, 6]

        if return_weights:
            return y_hat, alpha, g
        return y_hat


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    B = 8
    model = SSPMFN(d_audio=30, d_meta=12, d_part=17)
    xa = torch.randn(B, 30)
    xm = torch.randn(B, 12)
    xp = torch.randn(B, 17)
    eid = torch.randint(0, 3, (B,))

    y, alpha, g = model(xa, xm, xp, eid, return_weights=True)
    print(f"output shape = {tuple(y.shape)}")        # [8, 6]
    print(f"alpha shape  = {tuple(alpha.shape)}")     # [8, 6, 3]
    print(f"gate shape   = {tuple(g.shape)}")         # [8, 6, 3]
    print(f"params       = {count_params(model):,}")

    # 消融测试
    y_no_audio = model(xa, xm, xp, eid, mask={'audio': False, 'meta': True, 'part': True})
    print(f"ablation (no audio) shape = {tuple(y_no_audio.shape)}")

    # 无门控消融
    model_no_gate = SSPMFN(d_audio=30, d_meta=12, d_part=17, use_gate=False)
    y_ng = model_no_gate(xa, xm, xp, eid)
    print(f"no-gate output shape = {tuple(y_ng.shape)}")
    print(f"no-gate params = {count_params(model_no_gate):,}")
