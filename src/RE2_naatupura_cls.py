"""RE2 · Indian Folk Naatupura 真实分类基线

输入  real_naatupura_mel.npz  (7788 段 × 128 × 130 mel)
任务  Song(16) / Artist(4) / Gender(2) 三类分类, GroupKFold 5 折 (按 song 分组防泄漏)
模型  轻量 CNN (Conv-Conv-Pool×3 + FC), pytorch, MPS 后端
输出  实验/results/RE2_naatupura_cls.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path("/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/"
            "基于⺠族⾳乐交流的社会技能提升预测")
OUT = ROOT / "数据" / "真实数据集成" / "output"
RES = ROOT / "实验" / "results"
RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class MelDS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        m = self.X.mean(axis=(1, 2), keepdims=True)
        s = self.X.std(axis=(1, 2), keepdims=True) + 1e-6
        self.X = (self.X - m) / s

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).unsqueeze(0), int(self.y[i])


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 16, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x): return self.body(x)


def train_one(X_tr, y_tr, X_te, y_te, n_classes, epochs=8, lr=2e-3, bs=128):
    tr = DataLoader(MelDS(X_tr, y_tr), batch_size=bs, shuffle=True, num_workers=0)
    te = DataLoader(MelDS(X_te, y_te), batch_size=bs, shuffle=False, num_workers=0)
    model = SmallCNN(n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in te:
            xb = xb.to(DEVICE)
            p = model(xb).argmax(1).cpu().numpy()
            preds.append(p)
            ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return preds, ys


def cv_task(X, labels_str, groups_str, task_name, n_splits=5):
    uniq = sorted(set(labels_str))
    lab2id = {l: i for i, l in enumerate(uniq)}
    y = np.array([lab2id[l] for l in labels_str], dtype=np.int64)
    g_uniq = sorted(set(groups_str))
    g2id = {l: i for i, l in enumerate(g_uniq)}
    groups = np.array([g2id[l] for l in groups_str], dtype=np.int64)
    n_classes = len(uniq)

    if len(g_uniq) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split = list(splitter.split(X, y, groups))
        kind = "GroupKFold(by_song)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split = list(splitter.split(X, y))
        kind = "StratifiedKFold"

    fold_acc = []
    fold_f1 = []
    for fold, (tr, te) in enumerate(split):
        preds, ys = train_one(X[tr], y[tr], X[te], y[te], n_classes)
        acc = accuracy_score(ys, preds)
        f1m = f1_score(ys, preds, average="macro", zero_division=0)
        fold_acc.append(acc)
        fold_f1.append(f1m)
        print(f"  {task_name} fold {fold+1}: acc={acc:.4f} f1_macro={f1m:.4f} ntr={len(tr)} nte={len(te)}")

    return {
        "task": task_name,
        "n_classes": int(n_classes),
        "class_names": uniq,
        "cv_kind": kind,
        "n_splits": n_splits,
        "n_samples": int(len(y)),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "f1_macro_mean": float(np.mean(fold_f1)),
        "f1_macro_std": float(np.std(fold_f1)),
        "per_fold_acc": [float(x) for x in fold_acc],
        "per_fold_f1": [float(x) for x in fold_f1],
        "chance_acc": float(1.0 / n_classes),
    }


def main():
    print(f"[RE2] device = {DEVICE}")
    npz = np.load(OUT / "real_naatupura_mel.npz", allow_pickle=True)
    X = npz["mel_spec"]
    song = npz["song"]
    artist = npz["artist"]
    gender = npz["gender"]
    print(f"     X shape={X.shape} songs={len(set(song))} artists={len(set(artist))} genders={set(gender.tolist())}")

    if X.shape[0] > 2000:
        idx = np.random.default_rng(42).permutation(X.shape[0])[:2000]
        X_s = X[idx]
        song_s = song[idx]
        artist_s = artist[idx]
        gender_s = gender[idx]
        print(f"     subsample n=2000 for speed")
    else:
        X_s, song_s, artist_s, gender_s = X, song, artist, gender

    results = {
        "data_source": "Zenodo 6584021 Indian Folk Naatupura, CC-BY-4.0",
        "device": str(DEVICE),
        "tasks": {},
    }

    print("\n>>> task1: gender (binary)")
    results["tasks"]["gender_binary"] = cv_task(
        X_s, gender_s.tolist(), song_s.tolist(),
        task_name="gender", n_splits=5
    )

    print("\n>>> task2: artist (4 classes)")
    results["tasks"]["artist_4cls"] = cv_task(
        X_s, artist_s.tolist(), song_s.tolist(),
        task_name="artist", n_splits=5
    )

    print("\n>>> task3: song (16 classes, StratifiedKFold, song 标签即目标故不分组)")
    from sklearn.model_selection import StratifiedKFold
    y_song = np.array([s for s in song_s])
    uniq = sorted(set(y_song))
    lab2id = {l: i for i, l in enumerate(uniq)}
    y = np.array([lab2id[l] for l in y_song], dtype=np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc, fold_f1 = [], []
    for fold, (tr, te) in enumerate(skf.split(X_s, y)):
        preds, ys = train_one(X_s[tr], y[tr], X_s[te], y[te], len(uniq), epochs=8)
        acc = accuracy_score(ys, preds)
        f1m = f1_score(ys, preds, average="macro", zero_division=0)
        fold_acc.append(acc); fold_f1.append(f1m)
        print(f"  song fold {fold+1}: acc={acc:.4f} f1_macro={f1m:.4f}")
    results["tasks"]["song_16cls"] = {
        "task": "song", "n_classes": int(len(uniq)),
        "class_names": uniq, "cv_kind": "StratifiedKFold",
        "n_splits": 5, "n_samples": int(len(y)),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std": float(np.std(fold_acc)),
        "f1_macro_mean": float(np.mean(fold_f1)),
        "f1_macro_std": float(np.std(fold_f1)),
        "per_fold_acc": [float(x) for x in fold_acc],
        "per_fold_f1": [float(x) for x in fold_f1],
        "chance_acc": float(1.0 / len(uniq)),
    }

    (RES / "RE2_naatupura_cls.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float)
    )

    print("\n=== SUMMARY ===")
    for k, v in results["tasks"].items():
        print(f"  {k:<20} acc={v['acc_mean']:.4f}±{v['acc_std']:.4f}  "
              f"f1={v['f1_macro_mean']:.4f}  chance={v['chance_acc']:.4f}")
    print(f"saved → {RES / 'RE2_naatupura_cls.json'}")


if __name__ == "__main__":
    main()
