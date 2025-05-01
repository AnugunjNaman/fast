import argparse
import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from config import ADIMA_LANGUAGES, HOP_LENGTH, FastConfig
from fast_model import FastAST
from loader import AudioDataset, pad_time

CHECKPOINT_DIR = Path("saves/fast")
MULTI_PRETRAIN_PATH = CHECKPOINT_DIR / "multi_pretrain.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def checkpoint_path(language: str) -> Path:
    return CHECKPOINT_DIR / f"{language}.pth"


def class_weights_from_dataset(dataset, device):
    labels = np.asarray(dataset.labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)


def train_epoch(model, ema_model, dataloader, optimizer, criterion, expected_width, device, scaler, ema_decay):
    model.train()
    total_loss = 0.0
    amp_enabled = scaler is not None

    for batch in tqdm(dataloader, leave=False):
        inputs, labels = batch[0], batch[1]
        inputs = pad_time(inputs.to(device, non_blocking=True), expected_width)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(model(inputs.float()), labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        update_ema(ema_model, model, ema_decay)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def test_model(model, dataloader, criterion, expected_width, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    dataset = dataloader.dataset

    for _, targets, filenames in dataloader:
        targets = targets.to(device, non_blocking=True)
        for i, fname in enumerate(filenames):
            windows = pad_time(dataset.generate_eval_windows(fname).to(device), expected_width)
            clip_logits = model(windows.float()).mean(dim=0, keepdim=True)
            total_loss += criterion(clip_logits, targets[i].unsqueeze(0)).item()
            all_preds.append(int(torch.argmax(clip_logits, dim=1).item()))
            all_targets.append(int(targets[i].item()))

    n = max(len(all_targets), 1)
    acc = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / n
    return (
        acc,
        total_loss / n,
        f1_score(all_targets, all_preds, zero_division=0),
        precision_score(all_targets, all_preds, zero_division=0),
        recall_score(all_targets, all_preds, zero_division=0),
    )


def build_scheduler(optimizer, num_epochs, warmup_epochs, min_factor=0.02):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_audio_dataset(language, train, args, config, augment=False, return_filename=False):
    split = "train" if train else "test"
    return AudioDataset(
        f"data/audio/SC_audio_{language}/",
        f"data/annotations/{language}_{split}.csv",
        hop_size=args.hop_size,
        augment=augment,
        segment_frames=config.image_size[1],
        return_filename=return_filename,
        cache=True,
        eval_stride=0.5,
    )


class LabeledConcat(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.labels = []
        for dataset in datasets:
            self.labels.extend(dataset.labels)


def _make_ema(model):
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def pretrain_multilingual(args, languages, device, config):
    print("=" * 72)
    print(f"Multilingual pretrain on {languages}")
    print("=" * 72)
    set_seed(args.seed)
    model = FastAST(config=config).to(device)
    if args.init_ckpt:
        print(f"Warm-starting pretrain from {args.init_ckpt}")
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
    ema_model = _make_ema(model)
    expected_width = config.image_size[1]
    train_dataset = LabeledConcat(
        [make_audio_dataset(lang, True, args, config, augment=True) for lang in languages]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    weights = class_weights_from_dataset(train_dataset, device)
    print(f"Pretrain counts [No, Yes]: {np.bincount(train_dataset.labels, minlength=2).tolist()}")
    print(f"Pretrain weights [No, Yes]: {weights.tolist()}")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    scheduler = build_scheduler(optimizer, args.pretrain_epochs, min(2, args.pretrain_epochs))
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
    for epoch in range(args.pretrain_epochs):
        train_loss = train_epoch(
            model, ema_model, train_dataloader, optimizer, criterion,
            expected_width, device, scaler, args.ema_decay,
        )
        scheduler.step()
        print(
            f"Pretrain {epoch + 1}/{args.pretrain_epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} loss={train_loss:.4f}"
        )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ema_model.state_dict(), MULTI_PRETRAIN_PATH)
    print(f"Saved multilingual checkpoint to {MULTI_PRETRAIN_PATH}")
    del model, ema_model, train_dataloader, train_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return MULTI_PRETRAIN_PATH


def train_language(args, language, device, config, init_ckpt=None):
    print("=" * 72)
    print(f"Training {language}")
    print("=" * 72)
    set_seed(args.seed)

    model = FastAST(config=config).to(device)
    if init_ckpt is not None:
        print(f"Loading init checkpoint {init_ckpt}")
        model.load_state_dict(torch.load(init_ckpt, map_location=device))
    ema_model = _make_ema(model)
    expected_width = config.image_size[1]

    train_dataset = make_audio_dataset(language, True, args, config, augment=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    test_dataset = make_audio_dataset(language, False, args, config, return_filename=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    weights = class_weights_from_dataset(train_dataset, device)
    print(f"Class counts [No, Yes]: {np.bincount(train_dataset.labels, minlength=2).tolist()}")
    print(f"Class weights [No, Yes]: {weights.tolist()}")

    ft_lr = args.learning_rate * (0.4 if init_ckpt is not None else 1.0)
    optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    scheduler = build_scheduler(optimizer, args.num_epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    best_score = -1.0
    best_stats = (0.0, 0.0, 0.0, 0.0)
    ckpt = checkpoint_path(language)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(
            model, ema_model, train_dataloader, optimizer, criterion,
            expected_width, device, scaler, args.ema_decay,
        )
        acc, test_loss, f1, precision, recall = test_model(
            ema_model, test_dataloader, eval_criterion, expected_width, device,
        )
        scheduler.step()
        score = acc + f1
        if score > best_score:
            best_score = score
            best_stats = (acc, f1, precision, recall)
            torch.save(ema_model.state_dict(), ckpt)
            print(f"Saved {ckpt} (acc={acc:.4f} f1={f1:.4f})")
        print(
            f"Epoch {epoch + 1}/{args.num_epochs} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"acc={acc:.4f} f1={f1:.4f} p={precision:.4f} r={recall:.4f}"
        )

    del model, ema_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = FastAST(config=config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    acc, _, f1, precision, recall = test_model(
        model, test_dataloader, eval_criterion, expected_width, device,
    )
    print(
        f"Final Acc/F1: {acc:.4f} {f1:.4f} (P={precision:.4f} R={recall:.4f})  "
        f"best during train: acc={best_stats[0]:.4f} f1={best_stats[1]:.4f}"
    )
    del model, train_dataloader, test_dataloader, train_dataset, test_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acc, f1, precision, recall


def parse_args():
    parser = argparse.ArgumentParser(description="Train FAST on ADIMA (ICASSP 2025 recipe)")
    parser.add_argument(
        "--language",
        type=str,
        default="Hindi",
        help="Language, comma-separated list, or 'all'",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--hop_size", type=int, default=HOP_LENGTH)
    parser.add_argument("--pretrain_epochs", type=int, default=0)
    parser.add_argument(
        "--pretrain_languages",
        type=str,
        default="all",
        help="Comma-separated languages for multilingual pretrain, or 'all'",
    )
    parser.add_argument("--init_ckpt", type=str, default="", help="Checkpoint to fine-tune from")
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Run the ICASSP table recipe: 15-epoch multilingual pretrain + 20-epoch fine-tune on all 10 languages",
    )
    return parser.parse_args()


def parse_language_list(value: str):
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if not raw or raw[0].lower() == "all":
        return list(ADIMA_LANGUAGES)
    unknown = [lang for lang in raw if lang not in ADIMA_LANGUAGES]
    if unknown:
        raise SystemExit(f"Unknown language(s): {unknown}. Choose from {list(ADIMA_LANGUAGES)}")
    return raw


def main():
    args = parse_args()
    if args.reproduce:
        args.pretrain_epochs = 15
        args.num_epochs = 20
        args.language = "all"
        args.learning_rate = 4e-4
        args.batch_size = 16
        args.seed = 42
        args.hop_size = HOP_LENGTH
        args.weight_decay = 1e-4
        args.label_smoothing = 0.05
        args.warmup_epochs = 2
        args.ema_decay = 0.995
        args.pretrain_languages = "all"
        args.init_ckpt = ""

    languages = parse_language_list(args.language)
    print("Languages:", languages)
    print(
        f"epochs={args.num_epochs} pretrain_epochs={args.pretrain_epochs} "
        f"lr={args.learning_rate} batch={args.batch_size} seed={args.seed} gpu={args.gpu_id}"
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    config = FastConfig.icassp()
    print(config)

    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else None
    if args.pretrain_epochs > 0:
        pretrain_langs = parse_language_list(args.pretrain_languages)
        init_ckpt = pretrain_multilingual(args, pretrain_langs, device, config)

    results = {}
    for language in languages:
        results[language] = train_language(args, language, device, config, init_ckpt=init_ckpt)

    print("\n===== Summary =====")
    print(f"{'Language':<12} {'Acc':>7} {'F1':>7} {'P':>7} {'R':>7}  Checkpoint")
    for language, (acc, f1, precision, recall) in results.items():
        print(
            f"{language:<12} {acc:7.4f} {f1:7.4f} {precision:7.4f} {recall:7.4f}  "
            f"{checkpoint_path(language)}"
        )


if __name__ == "__main__":
    main()
