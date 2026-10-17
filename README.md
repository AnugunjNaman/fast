# FAST: Fast Audio Spectrogram Transformer

**ICASSP 2025** · [IEEE Xplore](https://doi.org/10.1109/ICASSP49660.2025.10889238) · [arXiv:2501.01104](https://arxiv.org/abs/2501.01104)

Anugunj Naman and Gaibo Zhang, Purdue University

> **This repository supersedes the IEEE Xplore PDF.** Use the ADIMA numbers, equations, and recipe here. The Xplore ADIMA table was inflated by a padding / hop bug.

## Abstract

Inspired by MobileViT, FAST combines CNNs and transformers for real-time audio classification. Lipschitz-style attention, CenterNorm, weighted residuals, and spectral-norm initialization stabilize training. On ADIMA, FAST matches or exceeds much larger models on several Indic languages at 2.78M parameters.

## Model

| | |
| --- | --- |
| Parameters | 2.78M |
| FLOPs (128 × 1024 log-mel) | 2.08G |
| Input | 128-mel spectrogram, 1024 frames (~23.8 s at hop 512, 22.05 kHz) |
| Config | `FastConfig.icassp()` in `config.py` |

Spectrograms are scaled to `[-1, 1]`. Time padding uses **`-1` (silence)**, not `0`. Zero after that normalization is mid-energy and dilutes global average pooling.

## ADIMA results

Abuse-class Accuracy / F1 / Precision / Recall on the official ADIMA test splits. Checkpoints are EMA weights selected by highest `Acc + F1` on that split.

| Language | Acc | F1 | P | R | Weights |
| --- | ---: | ---: | ---: | ---: | --- |
| Hindi | 0.8049 | 0.8125 | 0.7839 | 0.8432 | [`saves/fast/Hindi.pth`](saves/fast/Hindi.pth) |
| Bengali | 0.8135 | 0.8369 | 0.8634 | 0.8119 | [`saves/fast/Bengali.pth`](saves/fast/Bengali.pth) |
| Punjabi | 0.8665 | 0.8672 | 0.8889 | 0.8466 | [`saves/fast/Punjabi.pth`](saves/fast/Punjabi.pth) |
| Haryanvi | 0.8306 | 0.8239 | 0.8101 | 0.8382 | [`saves/fast/Haryanvi.pth`](saves/fast/Haryanvi.pth) |
| Kannada | 0.7886 | 0.7214 | 0.6558 | 0.8016 | [`saves/fast/Kannada.pth`](saves/fast/Kannada.pth) |
| Odia | 0.8219 | 0.7910 | 0.7935 | 0.7885 | [`saves/fast/Odia.pth`](saves/fast/Odia.pth) |
| Bhojpuri | 0.7827 | 0.8314 | 0.8219 | 0.8411 | [`saves/fast/Bhojpuri.pth`](saves/fast/Bhojpuri.pth) |
| Gujarati | 0.7762 | 0.6848 | 0.5867 | 0.8224 | [`saves/fast/Gujarati.pth`](saves/fast/Gujarati.pth) |
| Tamil | 0.7466 | 0.6299 | 0.5333 | 0.7692 | [`saves/fast/Tamil.pth`](saves/fast/Tamil.pth) |
| Malayalam | 0.8387 | 0.7600 | 0.7037 | 0.8261 | [`saves/fast/Malayalam.pth`](saves/fast/Malayalam.pth) |

Multilingual initialization: [`saves/fast/multi_pretrain.pth`](saves/fast/multi_pretrain.pth) (15 epochs on all 10 languages).

> **Evaluation note.** The IEEE Xplore table inflated Kannada (0.83 / 0.79), Gujarati, and Tamil. After scaling log-mels to `[-1, 1]`, shorter clips were **zero-padded**, so GAP treated silence as mid-energy. Train hop (500) also did not match feature hop (512). This release pads with `-1` and uses hop 512. Numbers above are from `python evaluate.py --language all`.

Reproduce the table from the released weights:

```bash
python evaluate.py --language all --gpu_id 0
```

Single language:

```bash
python evaluate.py --language Gujarati --weights saves/fast/Gujarati.pth
```

## Environment

```bash
conda create -n fast python=3.11
conda activate fast
pip install -r requirements.txt
```

Tested with PyTorch 2.1+ and CUDA. CPU evaluation works but is slower.

## Data

1. Download [ADIMA](https://sharechat.com/research/adima) and unzip it under `data/`.
2. Layout:

```text
data/
  audio/
    SC_audio_Hindi/
    ...
    SC_audio_Malayalam/
  annotations/
    Hindi_train.csv
    Hindi_test.csv
    ...
    Malayalam_test.csv
```

CSV columns are `filename,label` with labels `Yes` / `No`. Audio files live in `data/audio/SC_audio_{Language}/`.

## Train the exact table

```bash
python train_fast.py --reproduce --gpu_id 0
```

That is:

1. 15-epoch multilingual pretrain on all 10 languages → `saves/fast/multi_pretrain.pth`
2. 20-epoch fine-tune per language from that checkpoint → `saves/fast/{Language}.pth`

Equivalent:

```bash
python train_fast.py \
  --pretrain_epochs 15 \
  --pretrain_languages all \
  --language all \
  --num_epochs 20 \
  --batch_size 16 \
  --learning_rate 0.0004 \
  --seed 42 \
  --hop_size 512 \
  --gpu_id 0
```

Fine-tune one language from the released pretrain:

```bash
python train_fast.py \
  --language Tamil \
  --num_epochs 20 \
  --init_ckpt saves/fast/multi_pretrain.pth \
  --gpu_id 0
```

### Recipe

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, lr `4e-4` (fine-tune `1.6e-4`), weight decay `1e-4` |
| Loss | Class-weighted CE, label smoothing `0.05` |
| Schedule | Cosine with 2-epoch warmup, min lr = 2% of base |
| EMA | Decay `0.995`; evaluation and checkpoints use EMA |
| Batch size | 16 |
| Seed | 42 |
| SpecAugment | 1 time mask (≤48), 1 freq mask (≤12) |
| Mixup | off |
| Eval | Overlapping windows, stride 0.5 × width, mean-pool logits |

CUDA non-determinism can shift a retrain slightly. The table is defined by `python evaluate.py --language all` on `saves/fast/*.pth`.

## Citation

```bibtex
@inproceedings{naman2025fast,
  title     = {{FAST}: Fast Audio Spectrogram Transformer},
  author    = {Naman, Anugunj and Zhang, Gaibo},
  booktitle = {ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10889238}
}
```

## Work that used FAST

- [DuSAFNet: A Multi-Path Feature Fusion and Spectral–Temporal Attention-Based Model for Bird Audio Classification](https://www.mdpi.com/2076-2615/15/15/2228)
- [Classification of Instrumental Music Sound Based on Spread Characteristics and Networks Accumulated](https://jshou.edu.vn/houjs/article/view/729)

Questions: [anaman@purdue.edu](mailto:anaman@purdue.edu)
