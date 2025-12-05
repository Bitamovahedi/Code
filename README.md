# Code
# =========================
# 📦 Section 1: Imports & Global Params (AMP / Robust Backbones / Safe Load)
# =========================
import os, glob, random, json, numpy as np, pandas as pd, cv2
from glob import glob as gglob
from contextlib import contextmanager

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast  # ✅ new-style AMP (PyTorch >= 2.0)
from torch.cuda.amp import GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

import timm
from timm.data import Mixup
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ----------------- Reproducibility -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Paths / Config -----------------
# ✅ Dataset path
isic_base_dir = "/kaggle/input/isic-2019"

# Backbones
ACC_BACKBONE = "tf_efficientnet_b5_ns"  # upgraded accuracy model
BAL_BACKBONE = "resnext50_32x4d"        # balanced model

# Image sizes
TARGET_IMAGE_SIZE_ACC = 456             # upgraded from 384 -> 456
TARGET_IMAGE_SIZE_BAL = 384

# Batch size (even + drop_last=True in train loaders)
BATCH_SIZE = 32

BEST_CKPT_PATHS = {"acc": "best_acc_model.pth", "bal": "best_balanced_model.pth"}

# Training schedule (epochs per stage): Set-and-forget
EPOCHS_ACC = (10, 40, 20)   # Stage1/2/3
EPOCHS_BAL = (10, 40, 20)

# Optim / regularization
WEIGHT_DECAY = 2.5e-6
OPTIMAL_DROPOUT = 0.45      # a bit stronger for B5

# Mixup & smoothing
LABEL_SMOOTHING = 0.03      # slightly milder for accuracy
MIXUP_ACC = (0.0, 0.08, 0.04)
MIXUP_BAL = (0.0, 0.20, 0.10)

# Loss configs
LOSS_CFG_ACC = dict(alpha=0.5, gamma=1.5)   # favor top-1 accuracy
LOSS_CFG_BAL = dict(alpha=0.6, gamma=2.0)   # favor rare-class recall

# LRs per stage (backbone/head)
LRS_ACC = dict(head1=3.5e-4, back2=6e-5, head2=2.5e-4, back3=2.5e-5, head3=8e-5)
LRS_BAL = dict(head1=3e-4,   back2=8e-5,  head2=3e-4,  back3=4e-5,  head3=1.2e-4)

# EMA / AMP
USE_EMA = True
EMA_DECAY = 0.999
scaler = GradScaler(enabled=torch.cuda.is_available())

# TTA
TTA_MODES = ("orig","hflip","rot90","rot180","rot270")

# ----------------- Robust backbone builder -----------------
def create_backbone_safe(backbone_name: str, pretrained: bool = True):
    """
    Build a timm backbone safely:
      - map common aliases to canonical names that have pretrained weights
      - if pretrained weights are unavailable in this env, fall back to pretrained=False
    """
    alias = {
        # EfficientNetV2 aliases
        "efficientnetv2_s": "tf_efficientnetv2_s",
        "efficientnetv2_m": "tf_efficientnetv2_m",
        "efficientnetv2_l": "tf_efficientnetv2_l",
        # EfficientNet TF-NAS aliases
        "efficientnet_b5": "tf_efficientnet_b5_ns",
        "efficientnet_b7": "tf_efficientnet_b7_ns",
    }
    name = alias.get(backbone_name, backbone_name)
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=0)
    except RuntimeError as e:
        if "No pretrained weights exist" in str(e) or "pretrained" in str(e):
            print(f"[warn] {e} -> falling back to pretrained=False for '{name}'")
            return timm.create_model(name, pretrained=False, num_classes=0)
        raise

# ----------------- Safe checkpoint load for PyTorch 2.6 -----------------
def load_ckpt_safe(path, device):
    """
    PyTorch 2.6 sets weights_only=True by default; here we explicitly set it to False
    because checkpoints are trusted (created by this training).
    """
    return torch.load(path, map_location=device, weights_only=False)

# =========================
# 🧬 Section 2: Load ISIC2019 + Label mapping (8 classes)
# =========================
disease_dict = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
    'mel': 4, 'nv': 5, 'vasc': 6, 'scc': 7
}
class_labels_abbr = list(disease_dict.keys())
num_classes = len(disease_dict)

df_truth = pd.read_csv(os.path.join(isic_base_dir, 'ISIC_2019_Training_GroundTruth.csv'))
df_meta  = pd.read_csv(os.path.join(isic_base_dir, 'ISIC_2019_Training_Metadata.csv'))
df = pd.merge(df_truth, df_meta, on='image')

# Map image paths
image_paths = gglob(os.path.join(isic_base_dir, "**", "*.jpg"), recursive=True)
image_paths_dict = {
    os.path.splitext(os.path.basename(p))[0].replace('_downsampled', ''): p
    for p in image_paths
}
df['path'] = df['image'].map(image_paths_dict.get)

# Cleanup / label mapping
df['age'] = df['age_approx'].fillna(df['age_approx'].mean())
df['sex'] = df['sex'].fillna('unknown')
df = df.rename(columns={'image': 'image_id'})

label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
df['dx'] = df[label_cols].idxmax(axis=1).map({
    'AK': 'akiec', 'BCC': 'bcc', 'BKL': 'bkl', 'DF': 'df',
    'MEL': 'mel', 'NV': 'nv', 'VASC': 'vasc', 'SCC': 'scc'
})
df = df.dropna(subset=['path'])
df['cell_type_idx'] = df['dx'].map(disease_dict).astype(int)

# =========================
# 🧹 Section 3: Split + Meta features
# =========================
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['cell_type_idx'], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['cell_type_idx'], random_state=SEED
)

for d in [train_df, val_df, test_df]:
    d['age'] = d['age'].clip(0, 100)
    d['sex'] = d['sex'].fillna('unknown')

# One-hot for sex + align columns
train_df = pd.get_dummies(train_df, columns=['sex'], prefix='sex', dtype=float)
val_df   = pd.get_dummies(val_df,   columns=['sex'], prefix='sex', dtype=float)
test_df  = pd.get_dummies(test_df,  columns=['sex'], prefix='sex', dtype=float)

all_cols = set(train_df.columns) | set(val_df.columns) | set(test_df.columns)
sex_cols = [c for c in all_cols if 'sex_' in c]
for d in [train_df, val_df, test_df]:
    for c in sex_cols:
        if c not in d.columns: d[c] = 0.0
    d['age_sex_interaction'] = d.get('age', 0) * d.get('sex_male', 0)

metadata_features = ['age'] + sorted(sex_cols) + ['age_sex_interaction']

# =========================
# 🧾 Section 4: Dataset & Transforms (No Hair Removal)
# =========================
def build_transforms(img_size):
    train_t = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=1, max_height=int(0.12*img_size),
                        max_width=int(0.12*img_size), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_t = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_t, val_t

train_tf_acc, val_tf_acc = build_transforms(TARGET_IMAGE_SIZE_ACC)
train_tf_bal, val_tf_bal = build_transforms(TARGET_IMAGE_SIZE_BAL)

class SkinDataset(Dataset):
    """Returns ((image_tensor, meta_tensor), label)."""
    def __init__(self, df, metadata_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.metadata_cols = metadata_cols
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image'] if self.transform else img
        meta = torch.tensor(row[self.metadata_cols].values.astype(np.float32))
        label = torch.tensor(row['cell_type_idx'], dtype=torch.long)
        return (img, meta), label

    def __len__(self): return len(self.df)

# Train loaders: drop_last=True to keep batches even for Mixup
train_loader_acc = DataLoader(SkinDataset(train_df, metadata_features, train_tf_acc),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader_acc   = DataLoader(SkinDataset(val_df,   metadata_features, val_tf_acc),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader_acc  = DataLoader(SkinDataset(test_df,  metadata_features, val_tf_acc),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

train_loader_bal = DataLoader(SkinDataset(train_df, metadata_features, train_tf_bal),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader_bal   = DataLoader(SkinDataset(val_df,   metadata_features, val_tf_bal),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader_bal  = DataLoader(SkinDataset(test_df,  metadata_features, val_tf_bal),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# =========================
# 🧠 Section 5: MetaAttention + Backbone wrapper
# =========================
class MetaAttention(nn.Module):
    """Lightweight self-attention over tabular meta features -> 32-dim vector."""
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.key = nn.Linear(input_dim, embed_dim)
        self.query = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 32)

    def forward(self, x):
        k = self.key(x).unsqueeze(1)
        q = self.query(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1)
        out = torch.matmul(attn, v).squeeze(1)
        return self.fc(self.norm(out))

class BackboneMeta(nn.Module):
    """
    Generic image backbone + MetaAttention fusion + classifier head.
    backbone_name: e.g., "tf_efficientnet_b5_ns", "resnext50_32x4d".
    """
    def __init__(self, backbone_name, num_classes, meta_dim, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = create_backbone_safe(backbone_name, pretrained=pretrained)
        feat_dim = self.backbone.num_features
        self.meta = MetaAttention(meta_dim, embed_dim=128)   # -> 32
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim + 32),
            nn.Linear(feat_dim + 32, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_meta):
        img_feat = self.backbone(x_img)        # [B, F]
        meta_feat = self.meta(x_meta)          # [B, 32]
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(x)

# =========================
# 💥 Section 6: Class Weights (Effective Number) + Mixup + ComboLoss
# =========================
def effective_number_weights(labels_series, num_classes, beta=0.999):
    """Mild class weighting to help rare-class recall without killing accuracy."""
    counts = labels_series.value_counts().sort_index().values.astype(np.float64)
    en = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / (en + 1e-8)
    w = w / (w.sum() + 1e-8)
    return torch.tensor(w, dtype=torch.float32, device=device)

# Switches for the two models
USE_CLASS_WEIGHTS_ACC = False
USE_CLASS_WEIGHTS_BAL = True
EFFECTIVE_BETA_ACC = 0.999
EFFECTIVE_BETA_BAL = 0.9995

WEIGHTS_ACC = effective_number_weights(train_df['cell_type_idx'], num_classes, beta=EFFECTIVE_BETA_ACC) \
              if USE_CLASS_WEIGHTS_ACC else None
WEIGHTS_BAL = effective_number_weights(train_df['cell_type_idx'], num_classes, beta=EFFECTIVE_BETA_BAL) \
              if USE_CLASS_WEIGHTS_BAL else None

class ComboLoss(nn.Module):
    """
    ComboLoss = alpha * Focal(CE) + (1 - alpha) * CE
    - Supports hard labels ([B]) and soft labels ([B, C]) from Mixup.
    - Applies class weights in both branches.
    """
    def __init__(self, alpha=0.6, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        # CE term
        if targets.dtype in (torch.float16, torch.float32, torch.float64) and targets.dim() == 2:
            per_class = -targets * logp
            if self.weight is not None:
                per_class = per_class * self.weight.view(1, -1)
            ce = per_class.sum(dim=1)
        else:
            ce = F.nll_loss(logp, targets, weight=self.weight, reduction='none')
        # Focal term on CE
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        loss = self.alpha * focal + (1 - self.alpha) * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

def build_mixup(alpha):
    return Mixup(mixup_alpha=alpha, cutmix_alpha=0.0,
                 label_smoothing=LABEL_SMOOTHING, num_classes=num_classes)

# =========================
# 🔁 Section 7: Train/Val loops with AMP + EMA + Cosine (Mixup-safe)
# =========================
def ema_update(ema_state, model, decay):
    """In-place EMA update of state dict."""
    with torch.no_grad():
        if ema_state is None:
            return {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_((decay)).add_(v.detach(), alpha=(1.0 - decay))
            else:
                ema_state[k] = v
        return ema_state

@contextmanager
def use_ema_weights(model, ema_state):
    if (ema_state is None) or (not USE_EMA):
        yield; return
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(backup, strict=False)

def train_one_epoch(model, loader, opt, mixup, criterion, ema_state):
    """
    Training with:
      - AMP (torch.amp.autocast)
      - EMA update after optimizer step
      - Mixup-safe fallback if a batch happens to be odd-sized
    """
    model.train()
    total_loss = 0.0
    for (imgs, metas), labels in tqdm(loader, desc="Training"):
        imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)

        # ✅ Mixup-safe fallback: ensure even batch-size if mixup is active
        if mixup.mixup_alpha > 0 and (imgs.size(0) % 2 == 1):
            imgs, metas, labels = imgs[:-1], metas[:-1], labels[:-1]

        # Apply mixup if enabled
        if mixup.mixup_alpha > 0:
            imgs, targets_soft = mixup(imgs, labels)
            lbl = targets_soft
        else:
            lbl = labels

        opt.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs, metas)
            loss = criterion(logits, lbl)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if USE_EMA:
            ema_state = ema_update(ema_state, model, EMA_DECAY)
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset), ema_state

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, ema_state):
    model.eval()
    total_loss, preds, gts = 0.0, [], []
    with use_ema_weights(model, ema_state):
        for (imgs, metas), labels in tqdm(loader, desc="Validating"):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            gts.extend(labels.cpu().numpy())
    acc = accuracy_score(gts, preds)
    bacc = recall_score(gts, preds, average='macro')
    f1   = f1_score(gts, preds, average='macro')
    return total_loss / len(loader.dataset), acc, bacc, f1

def run_stages(model, loaders, epochs_3tuple, lrs, loss_cfg, weights, ckpt_path, mixup_alphas):
    """Run 3-stage schedule; save best by Val BACC and keep last-3 snapshots in Stage3."""
    best_bacc, ema_state = -1.0, None
    criterion = ComboLoss(weight=weights, **loss_cfg).to(device)

    # Stage 1: freeze backbone, train meta+head (warmup)
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in list(model.meta.parameters()) + list(model.classifier.parameters()): p.requires_grad = True
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lrs["head1"], weight_decay=WEIGHT_DECAY)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_3tuple[0])
    mix1 = build_mixup(mixup_alphas[0])

    for ep in range(epochs_3tuple[0]):
        print(f"\n[Stage1] Epoch {ep+1}/{sum(epochs_3tuple)}")
        tr_loss, ema_state = train_one_epoch(model, loaders['train'], opt1, mix1, criterion, ema_state)
        va_loss, va_acc, va_bacc, va_f1 = validate_one_epoch(model, loaders['val'], criterion, ema_state)
        sch1.step()
        print(f"TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | Acc {va_acc:.4f} | BACC {va_bacc:.4f} | F1 {va_f1:.4f}")
        if va_bacc > best_bacc:
            best_bacc = va_bacc
            torch.save({"model": model.state_dict(), "ema": ema_state,
                        "val_bacc": float(va_bacc), "val_acc": float(va_acc), "val_f1": float(va_f1)}, ckpt_path)
            print(f"💾 New BEST (BACC={va_bacc:.4f}) -> {ckpt_path}")

    # Stage 2: unfreeze; separate LR for backbone/head
    for p in model.parameters(): p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lrs["back2"]},
        {"params": list(model.meta.parameters()) + list(model.classifier.parameters()), "lr": lrs["head2"]}
    ], weight_decay=WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_3tuple[1])
    mix2 = build_mixup(mixup_alphas[1])

    for ep in range(epochs_3tuple[1]):
        print(f"\n[Stage2] Epoch {epochs_3tuple[0]+ep+1}/{sum(epochs_3tuple)}")
        tr_loss, ema_state = train_one_epoch(model, loaders['train'], opt2, mix2, criterion, ema_state)
        va_loss, va_acc, va_bacc, va_f1 = validate_one_epoch(model, loaders['val'], criterion, ema_state)
        sch2.step()
        print(f"TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | Acc {va_acc:.4f} | BACC {va_bacc:.4f} | F1 {va_f1:.4f}")
        if va_bacc > best_bacc:
            best_bacc = va_bacc
            torch.save({"model": model.state_dict(), "ema": ema_state,
                        "val_bacc": float(va_bacc), "val_acc": float(va_acc), "val_f1": float(va_f1)}, ckpt_path)
            print(f"💾 New BEST (BACC={va_bacc:.4f}) -> {ckpt_path}")

    # Stage 3: lower LRs; slight mixup + keep last-3 snapshots
    opt3 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lrs["back3"]},
        {"params": list(model.meta.parameters()) + list(model.classifier.parameters()), "lr": lrs["head3"]}
    ], weight_decay=WEIGHT_DECAY)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=epochs_3tuple[2])
    mix3 = build_mixup(mixup_alphas[2])

    # snapshot buffer
    snapshot_paths = []
    SNAPSHOT_TOPK = 3

    for ep in range(epochs_3tuple[2]):
        print(f"\n[Stage3] Epoch {sum(epochs_3tuple[:2])+ep+1}/{sum(epochs_3tuple)}")
        tr_loss, ema_state = train_one_epoch(model, loaders['train'], opt3, mix3, criterion, ema_state)
        va_loss, va_acc, va_bacc, va_f1 = validate_one_epoch(model, loaders['val'], criterion, ema_state)
        sch3.step()
        print(f"TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | Acc {va_acc:.4f} | BACC {va_bacc:.4f} | F1 {va_f1:.4f}")
        if va_bacc > best_bacc:
            best_bacc = va_bacc
            torch.save({"model": model.state_dict(), "ema": ema_state,
                        "val_bacc": float(va_bacc), "val_acc": float(va_acc), "val_f1": float(va_f1)}, ckpt_path)
            print(f"💾 New BEST (BACC={va_bacc:.4f}) -> {ckpt_path}")

        # Save snapshot every epoch (keep last K)
        snap_path = f"{ckpt_path.replace('.pth','')}_snap_ep{sum(epochs_3tuple[:2])+ep+1}.pth"
        torch.save({"model": model.state_dict(), "ema": ema_state}, snap_path)
        snapshot_paths.append(snap_path)
        if len(snapshot_paths) > SNAPSHOT_TOPK:
            try:
                os.remove(snapshot_paths.pop(0))
            except Exception as e:
                print(f"[warn] removing old snapshot failed: {e}")

# =========================
# 📊 Section 8: TTA + Single/Ensemble/Snapshot Evaluation
# =========================
@torch.no_grad()
def tta_predict(model, loader, ema_state, tta_modes=TTA_MODES):
    model.eval()
    probs_all, labels_all = [], []
    with use_ema_weights(model, ema_state):
        for (imgs, metas), labels in tqdm(loader, desc="TTA"):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            acc_probs = None
            for m in tta_modes:
                if   m=="orig":   it = imgs
                elif m=="hflip":  it = torch.flip(imgs, dims=[3])
                elif m=="rot90":  it = imgs.transpose(2,3).flip(2)
                elif m=="rot180": it = torch.flip(imgs, dims=[2,3])
                elif m=="rot270": it = imgs.transpose(2,3).flip(3)
                else: it = imgs
                logits = model(it, metas)
                probs  = F.softmax(logits, dim=1)
                acc_probs = probs if acc_probs is None else (acc_probs + probs)
            acc_probs = acc_probs / float(len(tta_modes))
            probs_all.append(acc_probs.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)

def evaluate_and_plot(probs, targets, tag=""):
    preds = np.argmax(probs, axis=1)
    acc  = accuracy_score(targets, preds)
    bacc = recall_score(targets, preds, average='macro')
    f1   = f1_score(targets, preds, average='macro')

    print(f"\n✅ {tag} Results:")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Balanced Acc  : {bacc:.4f}")
    print(f"  Macro F1      : {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(targets, preds, target_names=class_labels_abbr, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels_abbr, yticklabels=class_labels_abbr)
    plt.title(f"Confusion Matrix {tag}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(f"cm_{tag}.png", dpi=300); plt.close()
    print(f"🖼️ Saved: cm_{tag}.png")

    # ROC
    onehot = label_binarize(targets, classes=np.arange(num_classes))
    plt.figure(figsize=(10,8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(onehot[:, i], probs[:, i])
        auc = roc_auc_score(onehot[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_labels_abbr[i]} (AUC={auc:.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.title(f"ROC {tag}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"roc_{tag}.png", dpi=300); plt.close()
    print(f"🖼️ Saved: roc_{tag}.png")

    # PR
    plt.figure(figsize=(10,8))
    aps = []
    for i in range(num_classes):
        pr, rc, _ = precision_recall_curve(onehot[:, i], probs[:, i])
        ap = average_precision_score(onehot[:, i], probs[:, i])
        aps.append(ap); plt.plot(rc, pr, label=f"{class_labels_abbr[i]} (AP={ap:.2f})")
    plt.title(f"PR {tag}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc="lower left"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"pr_{tag}.png", dpi=300); plt.close()
    print(f"🖼️ Saved: pr_{tag}.png")

    with open(f"metrics_{tag}.json","w") as f:
        json.dump({"acc": float(acc), "bacc": float(bacc), "f1": float(f1),
                   "per_class_AP": {cls: float(aps[i]) for i, cls in enumerate(class_labels_abbr)}}, f, indent=2)
    print(f"📝 Saved: metrics_{tag}.json")

@torch.no_grad()
def snapshot_ensemble_eval(snapshot_glob_pattern, backbone_name, test_loader):
    """Average probs from K snapshots of the SAME model (last K matched by glob)."""
    snaps = sorted(glob.glob(snapshot_glob_pattern))
    if len(snaps) == 0:
        print(f"[warn] No snapshots found for pattern: {snapshot_glob_pattern}")
        return
    model = BackboneMeta(backbone_name, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)
    probs_accum, targets_ref, n = None, None, 0
    for sp in snaps:
        c = load_ckpt_safe(sp, device)
        model.load_state_dict(c["model"])
        ema = c.get("ema", None)
        p, t = tta_predict(model, test_loader, ema, tta_modes=TTA_MODES)
        probs_accum = p if probs_accum is None else (probs_accum + p)
        targets_ref = t; n += 1
    probs = probs_accum / n
    evaluate_and_plot(probs, targets_ref, tag=f"SNAPSHOTx{n}_{os.path.basename(snapshot_glob_pattern)[:-4]}")

@torch.no_grad()
def ensemble_eval_weighted(model_a_name, ckpt_a, model_b_name, ckpt_b, val_loader, test_loader):
    """Grid-search the best weight on validation, then evaluate on test."""
    # Build models
    ma = BackboneMeta(model_a_name, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)
    mb = BackboneMeta(model_b_name, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)
    ca = load_ckpt_safe(ckpt_a, device); ma.load_state_dict(ca["model"]); ema_a = ca.get("ema", None)
    cb = load_ckpt_safe(ckpt_b, device); mb.load_state_dict(cb["model"]); ema_b = cb.get("ema", None)

    # TTA on validation
    pa, y = tta_predict(ma, val_loader, ema_a, tta_modes=("orig","hflip","rot180"))
    pb, _ = tta_predict(mb, val_loader, ema_b, tta_modes=("orig","hflip","rot180"))

    best_w, best_acc = 0.5, -1
    for w in np.linspace(0.0, 1.0, 21):
        p = w*pa + (1-w)*pb
        acc = accuracy_score(y, np.argmax(p, axis=1))
        if acc > best_acc: best_acc, best_w = acc, w
    print(f"[grid] best w={best_w:.2f} on validation (ACC={best_acc:.4f})")

    # Evaluate on test with the best weight
    pa_test, t_test = tta_predict(ma, test_loader, ema_a, tta_modes=TTA_MODES)
    pb_test, _      = tta_predict(mb, test_loader, ema_b, tta_modes=TTA_MODES)
    probs = best_w*pa_test + (1-best_w)*pb_test
    evaluate_and_plot(probs, t_test, tag=f"ENSEMBLE_w{best_w:.2f}")

# =========================
# 🚀 Section 9: Train both models then (Snapshot + Weighted) Ensemble on test
# =========================
def main():
    # Build models
    model_acc = BackboneMeta(ACC_BACKBONE, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)
    model_bal = BackboneMeta(BAL_BACKBONE, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)

    # Accuracy-focused model
    loaders_acc = {"train": train_loader_acc, "val": val_loader_acc}
    run_stages(model_acc, loaders_acc, EPOCHS_ACC, LRS_ACC, LOSS_CFG_ACC,
               WEIGHTS_ACC, BEST_CKPT_PATHS["acc"], MIXUP_ACC)

    # Balanced-focused model
    loaders_bal = {"train": train_loader_bal, "val": val_loader_bal}
    run_stages(model_bal, loaders_bal, EPOCHS_BAL, LRS_BAL, LOSS_CFG_BAL,
               WEIGHTS_BAL, BEST_CKPT_PATHS["bal"], MIXUP_BAL)

    # Single-model evals (TTA + EMA)
    for tag, bname, ckpt, test_loader in [
        ("ACC", ACC_BACKBONE, BEST_CKPT_PATHS["acc"], test_loader_acc),
        ("BAL", BAL_BACKBONE, BEST_CKPT_PATHS["bal"], test_loader_bal)
    ]:
        model = BackboneMeta(bname, num_classes, len(metadata_features), dropout=OPTIMAL_DROPOUT).to(device)
        ema = None
        if os.path.exists(ckpt):
            c = load_ckpt_safe(ckpt, device)
            model.load_state_dict(c["model"]); ema = c.get("ema", None)
            print(f"[info] Loaded {tag} ckpt: BACC={c.get('val_bacc', None)}")
        probs, targets = tta_predict(model, test_loader, ema, tta_modes=TTA_MODES)
        evaluate_and_plot(probs, targets, tag=tag)

    # Snapshot-ensemble for each model (if snapshots exist)
    acc_snap_glob = BEST_CKPT_PATHS["acc"].replace(".pth","") + "_snap_ep*.pth"
    bal_snap_glob = BEST_CKPT_PATHS["bal"].replace(".pth","") + "_snap_ep*.pth"
    snapshot_ensemble_eval(acc_snap_glob, ACC_BACKBONE, test_loader_acc)
    snapshot_ensemble_eval(bal_snap_glob, BAL_BACKBONE, test_loader_bal)

    # Weighted ensemble (w from validation), evaluated on test
    ensemble_eval_weighted(ACC_BACKBONE, BEST_CKPT_PATHS["acc"], BAL_BACKBONE, BEST_CKPT_PATHS["bal"],
                           val_loader_acc, test_loader_acc)

if __name__ == "__main__":
    main()
