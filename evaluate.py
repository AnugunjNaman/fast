import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from config import ADIMA_LANGUAGES, HOP_LENGTH, FastConfig
from fast_model import FastAST
from loader import AudioDataset, pad_time

CHECKPOINT_DIR = Path("saves/fast")


def resolve_weights(language: str, weights: Path | None) -> Path:
    if weights is not None:
        return weights
    candidates = [
        CHECKPOINT_DIR / f"{language}.pth",
        CHECKPOINT_DIR / f"{language}_1_ast_weights_epoch.pth",
        CHECKPOINT_DIR / f"{language.lower()}-1.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint for {language}. Looked for: {', '.join(str(p) for p in candidates)}"
    )


def build_model(weights_path: Path, device: torch.device) -> FastAST:
    config = FastConfig.icassp()
    model = FastAST(config=config)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_dataloader(language: str, audio_root: Path, annotations_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = AudioDataset(
        f"{audio_root}/SC_audio_{language}/",
        f"{annotations_dir}/{language}_test.csv",
        hop_size=HOP_LENGTH,
        augment=False,
        segment_frames=FastConfig.icassp().image_size[1],
        return_filename=True,
        cache=True,
        eval_stride=0.5,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_loader(model: FastAST, dataloader: DataLoader, device: torch.device) -> dict:
    expected_width = model.config.image_size[1]
    all_preds, all_targets = [], []
    dataset = dataloader.dataset
    for _, labels, filenames in dataloader:
        labels = labels.to(device)
        for idx, fname in enumerate(filenames):
            windows = pad_time(dataset.generate_eval_windows(fname).to(device), expected_width)
            logits = model(windows.float()).mean(dim=0, keepdim=True)
            all_preds.append(int(torch.argmax(logits, dim=1).item()))
            all_targets.append(int(labels[idx].item()))
    return {
        "acc": accuracy_score(all_targets, all_preds),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "n": len(all_targets),
    }


def format_row(language: str, metrics: dict, weights: Path) -> str:
    return (
        f"{language:<12} {metrics['acc']:.4f} {metrics['f1']:.4f} "
        f"{metrics['precision']:.4f} {metrics['recall']:.4f}  {weights}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAST checkpoints on the ADIMA test split")
    parser.add_argument("--language", type=str, default="Hindi", help="Language or 'all'")
    parser.add_argument("--audio_root", type=Path, default=Path("data/audio"))
    parser.add_argument("--annotations_dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--weights", type=Path, default=None, help="Checkpoint file (single language)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    if args.language.strip().lower() == "all":
        languages = list(ADIMA_LANGUAGES)
        if args.weights is not None:
            raise SystemExit("--weights cannot be combined with --language all")
    else:
        languages = [args.language.strip()]

    print(f"{'Language':<12} {'Acc':>6} {'F1':>6} {'P':>6} {'R':>6}  Checkpoint")
    rows = []
    for language in languages:
        weights = resolve_weights(language, args.weights)
        dataloader = build_dataloader(
            language, args.audio_root, args.annotations_dir, args.batch_size, args.num_workers,
        )
        model = build_model(weights, device)
        metrics = evaluate_loader(model, dataloader, device)
        print(format_row(language, metrics, weights))
        rows.append((language, metrics, weights))
        del model, dataloader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(rows) > 1:
        mean_acc = sum(m["acc"] for _, m, _ in rows) / len(rows)
        mean_f1 = sum(m["f1"] for _, m, _ in rows) / len(rows)
        print(f"{'Macro':<12} {mean_acc:.4f} {mean_f1:.4f}")


if __name__ == "__main__":
    main()
