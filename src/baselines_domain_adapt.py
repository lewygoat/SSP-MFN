"""领域自适应基线模型

MLP+MMD   — 最大均值差异对齐
MLP+CORAL — 协方差对齐 (CORrelation ALignment)

用于对比 SSP-MFN 的 Cultural AdaIN 是否优于传统领域自适应方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mmd(x, y, kernel='rbf', bandwidth=1.0):
    """计算 MMD (Maximum Mean Discrepancy) between two distributions
    使用 Gaussian RBF kernel, 支持不等样本量
    """
    # ||x_i - x_j||^2
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    XX = torch.exp(-0.5 * xx / bandwidth)
    YY = torch.exp(-0.5 * yy / bandwidth)
    XY = torch.exp(-0.5 * xy / bandwidth)

    return XX.mean() + YY.mean() - 2.0 * XY.mean()


def compute_coral(source, target):
    """计算 CORAL loss (协方差矩阵差异)"""
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # 协方差矩阵
    source_centered = source - source.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)

    cs = (source_centered.t() @ source_centered) / (ns - 1 + 1e-8)
    ct = (target_centered.t() @ target_centered) / (nt - 1 + 1e-8)

    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss


class MLPEncoder(nn.Module):
    """共享 MLP 编码器 (用于 MMD 和 CORAL)"""
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        d_in = d_audio + d_meta + d_part
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_audio, x_meta, x_part):
        x = torch.cat([x_audio, x_meta, x_part], dim=-1)
        return self.encoder(x)


class MLPWithMMD(nn.Module):
    """MLP + MMD 领域自适应

    训练时: loss = MSE + λ * MMD(feat_ethnic_i, feat_ethnic_j)
    推理时: 正常前向传播
    """
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_hidden: int = 64, n_out: int = 6, dropout: float = 0.2,
                 lambda_mmd: float = 0.1):
        super().__init__()
        self.encoder = MLPEncoder(d_audio, d_meta, d_part, d_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_out),
        )
        self.lambda_mmd = lambda_mmd

    def forward(self, x_audio, x_meta, x_part, ethnic_id=None):
        feat = self.encoder(x_audio, x_meta, x_part)
        pred = self.head(feat)
        return pred

    def compute_loss(self, x_audio, x_meta, x_part, y, ethnic_id, criterion):
        """训练时调用，包含 MMD 正则"""
        feat = self.encoder(x_audio, x_meta, x_part)
        pred = self.head(feat)
        mse_loss = criterion(pred, y)

        # MMD: 对齐不同民族的特征分布
        mmd_loss = torch.tensor(0.0, device=feat.device)
        unique_eth = ethnic_id.unique()
        n_pairs = 0
        for i in range(len(unique_eth)):
            for j in range(i + 1, len(unique_eth)):
                mask_i = ethnic_id == unique_eth[i]
                mask_j = ethnic_id == unique_eth[j]
                if mask_i.sum() > 1 and mask_j.sum() > 1:
                    mmd_loss = mmd_loss + compute_mmd(feat[mask_i], feat[mask_j])
                    n_pairs += 1
        if n_pairs > 0:
            mmd_loss = mmd_loss / n_pairs

        return mse_loss + self.lambda_mmd * mmd_loss


class MLPWithCORAL(nn.Module):
    """MLP + CORAL 领域自适应

    训练时: loss = MSE + λ * CORAL(feat_ethnic_i, feat_ethnic_j)
    """
    def __init__(self, d_audio: int, d_meta: int, d_part: int,
                 d_hidden: int = 64, n_out: int = 6, dropout: float = 0.2,
                 lambda_coral: float = 0.1):
        super().__init__()
        self.encoder = MLPEncoder(d_audio, d_meta, d_part, d_hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_out),
        )
        self.lambda_coral = lambda_coral

    def forward(self, x_audio, x_meta, x_part, ethnic_id=None):
        feat = self.encoder(x_audio, x_meta, x_part)
        pred = self.head(feat)
        return pred

    def compute_loss(self, x_audio, x_meta, x_part, y, ethnic_id, criterion):
        """训练时调用，包含 CORAL 正则"""
        feat = self.encoder(x_audio, x_meta, x_part)
        pred = self.head(feat)
        mse_loss = criterion(pred, y)

        # CORAL: 对齐不同民族的协方差矩阵
        coral_loss = torch.tensor(0.0, device=feat.device)
        unique_eth = ethnic_id.unique()
        n_pairs = 0
        for i in range(len(unique_eth)):
            for j in range(i + 1, len(unique_eth)):
                mask_i = ethnic_id == unique_eth[i]
                mask_j = ethnic_id == unique_eth[j]
                if mask_i.sum() > 1 and mask_j.sum() > 1:
                    coral_loss = coral_loss + compute_coral(feat[mask_i], feat[mask_j])
                    n_pairs += 1
        if n_pairs > 0:
            coral_loss = coral_loss / n_pairs

        return mse_loss + self.lambda_coral * coral_loss
