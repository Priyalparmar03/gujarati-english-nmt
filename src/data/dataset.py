import random
import torch
from torch.utils.data import Dataset, Sampler
from src.model.tokenizer import PAD_ID


class ParallelTextDataset(Dataset):
    """Reads aligned src/tgt line files and tokenizes on the fly."""

    def __init__(self, src_path, tgt_path, tokenizer, max_len=128):
        with open(src_path, encoding="utf-8") as f:
            self.src_lines = [l.rstrip("\n") for l in f]
        with open(tgt_path, encoding="utf-8") as f:
            self.tgt_lines = [l.rstrip("\n") for l in f]
        assert len(self.src_lines) == len(self.tgt_lines), "src/tgt line count mismatch"
        self.tokenizer = tokenizer
        self.max_len = max_len
        # pre-tokenize once so the length-bucketing sampler doesn't re-tokenize
        # every epoch just to know sentence lengths
        self._lengths = None

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_ids = self.tokenizer.encode(self.src_lines[idx])[: self.max_len]
        tgt_ids = self.tokenizer.encode(self.tgt_lines[idx])[: self.max_len]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

    def lengths(self):
        """Actual subword token length (via the tokenizer) for bucketing, cached."""
        if self._lengths is None:
            self._lengths = [
                max(len(self.tokenizer.encode(s)), len(self.tokenizer.encode(t)))
                for s, t in zip(self.src_lines, self.tgt_lines)
            ]
        return self._lengths


class TokenBudgetBatchSampler(Sampler):
    """
    Groups examples into length buckets, then greedily packs each batch up to
    a target token budget (src+tgt tokens), following the "batch by tokens,
    not by sentence count" practice from Vaswani et al. (~25k tokens/batch).
    This keeps GPU memory and step time roughly constant across batches
    instead of wasting compute padding short sentences up to the longest one
    in a fixed-size batch.
    """

    def __init__(self, dataset: ParallelTextDataset, max_tokens=25000, shuffle=True, bucket_mult=100):
        self.lengths = dataset.lengths()
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.bucket_mult = bucket_mult
        self._build_batches()

    def _build_batches(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(idx)
        # coarse length sort within shuffled mega-chunks, so batches are
        # length-homogeneous but epoch order still varies
        chunk = self.bucket_mult * max(1, self.max_tokens // 32)
        batches = []
        for start in range(0, len(idx), chunk):
            piece = idx[start : start + chunk]
            piece.sort(key=lambda i: self.lengths[i])
            cur_batch, cur_max_len = [], 0
            for i in piece:
                new_max_len = max(cur_max_len, self.lengths[i])
                if cur_batch and new_max_len * (len(cur_batch) + 1) > self.max_tokens:
                    batches.append(cur_batch)
                    cur_batch, cur_max_len = [i], self.lengths[i]
                else:
                    cur_batch.append(i)
                    cur_max_len = new_max_len
            if cur_batch:
                batches.append(cur_batch)
        if self.shuffle:
            random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        self._build_batches()  # re-bucket each epoch for fresh shuffling
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : len(s)] = s
        tgt_padded[i, : len(t)] = t

    return src_padded, tgt_padded