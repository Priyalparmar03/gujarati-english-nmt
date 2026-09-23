"""
Download and clean the English-Gujarati parallel corpus.

Sources:
  - Training data: Samanantar (AI4Bharat) via HuggingFace `ai4bharat/samanantar`, config "gu"
    https://huggingface.co/datasets/ai4bharat/samanantar
  - Test/val data: FLORES-200 via HuggingFace `facebook/flores`, config "guj_Gujr-eng_Latn"
    https://huggingface.co/datasets/facebook/flores
    (gated dataset — accept the terms on the HF page once, logged in via `huggingface-cli login`)

Also removes train/test leakage: Samanantar is mined from the web and
FLORES-200 draws partly from Wikipedia, so near-duplicate sentences can leak
across the two and inflate BLEU/chrF. This filters train against test both
by exact normalized match and by 5-gram shingle overlap before writing
train.en/train.gu.

Usage:
    python -m src.data.preprocess --out_dir data/
"""
import argparse
import os
import re
import string
import unicodedata
from datasets import load_dataset


GUJARATI_RANGE = re.compile(r"[\u0A80-\u0AFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization used only for leakage comparison, not for training text."""
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingles(text: str, n: int = 5):
    words = text.split()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def is_valid_pair(en: str, gu: str, min_len=3, max_len=400, max_ratio=3.0) -> bool:
    if not en or not gu:
        return False
    if not (min_len <= len(en) <= max_len and min_len <= len(gu) <= max_len):
        return False
    if not GUJARATI_RANGE.search(gu):
        return False
    if not LATIN_RANGE.search(en):
        return False
    ratio = max(len(en), len(gu)) / max(1, min(len(en), len(gu)))
    if ratio > max_ratio:
        return False
    return True


def dedup(pairs):
    seen = set()
    out = []
    for en, gu in pairs:
        key = (en, gu)
        if key in seen:
            continue
        seen.add(key)
        out.append((en, gu))
    return out


def build_leakage_filter(test_pairs, shingle_overlap_thresh=0.7):
    """Returns a function(en, gu) -> True if this train pair looks like it
    leaked from the test set (exact or near-duplicate on the English side)."""
    test_exact = {normalize_for_dedup(en) for en, _ in test_pairs}
    test_shingle_sets = [shingles(normalize_for_dedup(en)) for en, _ in test_pairs]
    test_shingle_union = set().union(*test_shingle_sets) if test_shingle_sets else set()

    def is_leaked(en: str, gu: str) -> bool:
        norm = normalize_for_dedup(en)
        if norm in test_exact:
            return True
        sh = shingles(norm)
        if not sh:
            return False
        overlap = len(sh & test_shingle_union) / len(sh)
        return overlap >= shingle_overlap_thresh

    return is_leaked


def load_samanantar():
    ds = load_dataset("ai4bharat/samanantar", "gu", split="train")
    return [(normalize(r["src"]), normalize(r["tgt"])) for r in ds]


def load_flores_split(split: str):
    ds = load_dataset("facebook/flores", "guj_Gujr-eng_Latn", split=split)
    return [
        (normalize(r["sentence_eng_Latn"]), normalize(r["sentence_guj_Gujr"]))
        for r in ds
    ]


def write_pairs(pairs, en_path, gu_path):
    with open(en_path, "w", encoding="utf-8") as fe, open(gu_path, "w", encoding="utf-8") as fg:
        for en, gu in pairs:
            fe.write(en + "\n")
            fg.write(gu + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_holdout", type=int, default=2000,
                     help="sentences carved out of Samanantar train for a dev set")
    ap.add_argument("--shingle_overlap_thresh", type=float, default=0.7,
                     help="fraction of a train sentence's 5-grams that must overlap test 5-grams to count as leakage")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading FLORES-200 devtest as held-out test set ...")
    test_pairs = load_flores_split("devtest")
    write_pairs(test_pairs, f"{args.out_dir}/test.en", f"{args.out_dir}/test.gu")
    print(f"  test pairs: {len(test_pairs)}")

    print("Loading Samanantar en-gu ...")
    train_pairs = load_samanantar()
    print(f"  raw pairs: {len(train_pairs)}")

    train_pairs = [(en, gu) for en, gu in train_pairs if is_valid_pair(en, gu)]
    train_pairs = dedup(train_pairs)
    print(f"  after clean+dedup: {len(train_pairs)}")

    print("Filtering train/test leakage against FLORES-200 ...")
    is_leaked = build_leakage_filter(test_pairs, args.shingle_overlap_thresh)
    before = len(train_pairs)
    train_pairs = [(en, gu) for en, gu in train_pairs if not is_leaked(en, gu)]
    print(f"  removed {before - len(train_pairs)} leaked/near-duplicate pairs ({before} -> {len(train_pairs)})")

    dev_pairs = train_pairs[: args.val_holdout]
    train_pairs = train_pairs[args.val_holdout :]

    write_pairs(train_pairs, f"{args.out_dir}/train.en", f"{args.out_dir}/train.gu")
    write_pairs(dev_pairs, f"{args.out_dir}/dev.en", f"{args.out_dir}/dev.gu")

    print("Done. Files written to", args.out_dir)
    print(f"  train: {len(train_pairs)} | dev: {len(dev_pairs)} | test: {len(test_pairs)}")


if __name__ == "__main__":
    main()
