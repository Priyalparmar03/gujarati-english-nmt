# Gujarati-English NMT (from scratch)

An English-to-Gujarati neural machine translation system. The transformer
encoder-decoder is built and trained from scratch — no pretrained weights,
no fine-tuning of an existing MT model.

## Data

| Split | Source | Link |
|---|---|---|
| Train | Samanantar (AI4Bharat), `gu` config, ~3.07M pairs | https://huggingface.co/datasets/ai4bharat/samanantar |
| Dev | 2,000 pairs held out from Samanantar train | — |
| Test | FLORES-200 `devtest`, `guj_Gujr-eng_Latn`, 1,012 pairs | https://huggingface.co/datasets/facebook/flores |

FLORES-200 is a gated dataset on HuggingFace — accept the terms on the
dataset page once, then `huggingface-cli login` before running preprocessing.

Alternative/supplementary sources if you want more training volume later:
OPUS (https://opus.nlpl.eu, search "gu"), which aggregates several smaller
En-Gu parallel corpora.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # needed once, for FLORES-200 access
```

## Pipeline

```bash
# 1. Download + clean data (also filters train/test leakage against FLORES-200)
python -m src.data.preprocess --out_dir data/

# 2. Train shared BPE tokenizer (32k vocab, English + Gujarati)
python -m src.model.tokenizer --train data/train.en data/train.gu \
    --model_prefix data/spm_shared --vocab_size 32000

# 3. Train the transformer (English -> Gujarati by default; pass --direction gu2en to reverse)
python -m src.train --data_dir data/ --spm_model data/spm_shared.model \
    --out_dir checkpoints/ --epochs 20 --max_tokens 25000

# resume an interrupted run:
python -m src.train --data_dir data/ --spm_model data/spm_shared.model \
    --out_dir checkpoints/ --epochs 20 --resume checkpoints/last.pt

# 4. Evaluate on FLORES-200 test set
python -m src.evaluate --ckpt checkpoints/best.pt --spm_model data/spm_shared.model \
    --src data/test.en --ref data/test.gu --out results/test_hyp.gu \
    --beam_size 4 --use_comet
```

### Before committing to a full run

Run a smoke test on a small slice first (e.g. 50k-100k pairs, 2-3 epochs) to
confirm the pipeline produces sane loss curves before spending days of
compute on the full 3M-pair run:

```bash
head -n 50000 data/train.en > data/train_small.en
head -n 50000 data/train.gu > data/train_small.gu
# point --data_dir at a folder with train_small.{en,gu} renamed to train.{en,gu}, plus dev/test
```

Unit tests cover mask correctness (causal masking, padding masking) and
tokenizer roundtrip:

```bash
pip install pytest
python -m pytest tests/ -v
```

### Batching, precision, and reproducibility

- Training batches by **token budget** (`--max_tokens`, default 25,000 —
  the Vaswani et al. standard), not fixed sentence count, with length
  bucketing (`TokenBudgetBatchSampler` in `src/data/dataset.py`) so batches
  stay length-homogeneous and GPU time per step is roughly constant.
- Mixed precision (`torch.amp`) is on by default on CUDA (`--no_amp` to
  disable); roughly halves memory and time on modern GPUs.
- `--seed` (default 42) sets `torch`/`numpy`/`random` seeds for
  reproducibility.
- Only `checkpoints/best.pt` and `checkpoints/last.pt` are kept (no
  per-epoch checkpoint files); `--resume checkpoints/last.pt` continues an
  interrupted run from the right epoch, optimizer, and scheduler state.
- `checkpoints/training_log.csv` logs epoch/train_loss/dev_loss/lr/seconds
  for plotting the loss curve later.

## Architecture

Encoder-decoder transformer, hand-written in PyTorch (`src/model/transformer.py`):
6 encoder + 6 decoder layers, d_model=512, 8 heads, feedforward dim 2048,
tied input/output embeddings, sinusoidal positional encoding, label smoothing
(0.1), Noam learning-rate schedule (warmup 4000 steps). No HuggingFace
`transformers` model classes are used for the model itself — only `datasets`
and `sentencepiece` are used as data/tokenizer utilities.

## Translation direction

`--direction en2gu` (default) trains English source -> Gujarati target,
matching the stated goal of an app that converts English content into
Gujarati. `--direction gu2en` trains the reverse. Pick one and report that
one in the paper — if comparing against Panchal et al. (arXiv 2506.21566),
match whichever direction they report for a fair comparison.

## Status

This is a scaffold: the pipeline runs end to end, but nothing has been
trained yet. Once a checkpoint exists, next steps are:
1. Baseline comparison against Panchal et al. (arXiv 2506.21566) and
   Google Translate / IndicTrans2 on the same FLORES-200 test set
   (`baselines/`)
2. Linguistic error taxonomy — define categories, annotate a sample of
   test-set translations, compute error-frequency breakdown
   (`src/error_taxonomy/`)
3. Write up results and taxonomy findings (`paper/`)

## Repo layout

```
src/
  data/          preprocessing, PyTorch Dataset
  model/         tokenizer, transformer architecture
  train.py        training loop
  evaluate.py      BLEU / chrF / COMET
  error_taxonomy/  annotation tooling + taxonomy definition (TODO)
baselines/         comparison runs against Panchal et al. and existing systems (TODO)
results/           metric tables, error taxonomy breakdown (TODO)
paper/             arXiv draft (TODO)
```
