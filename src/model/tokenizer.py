"""
Trains a single shared SentencePiece BPE tokenizer over English + Gujarati
training text, and wraps it for encode/decode use in the model.

Usage:
    python -m src.model.tokenizer --train data/train.en data/train.gu \
        --vocab_size 32000 --model_prefix data/spm_shared
"""
import argparse
import sentencepiece as spm


SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train(files, model_prefix: str, vocab_size: int = 32000):
    spm.SentencePieceTrainer.train(
        input=",".join(files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9998,  # high coverage needed for Gujarati script
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )


class SharedTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text: str, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--model_prefix", default="data/spm_shared")
    ap.add_argument("--vocab_size", type=int, default=32000)
    args = ap.parse_args()
    train(args.train, args.model_prefix, args.vocab_size)
    print(f"Tokenizer written to {args.model_prefix}.model / .vocab")
