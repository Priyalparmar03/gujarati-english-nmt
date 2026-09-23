"""Unit test: tokenizer encode -> decode recovers the original string
(modulo BPE-normalized whitespace)."""
import os
import tempfile
import sentencepiece as spm
from src.model.tokenizer import train, SharedTokenizer


def test_encode_decode_roundtrip():
    sentences = [
        "This is a simple English sentence.",
        "Machine translation needs a lot of data.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = os.path.join(tmp, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))

        prefix = os.path.join(tmp, "spm_test")
        train([corpus_path], prefix, vocab_size=100)

        tok = SharedTokenizer(prefix + ".model")
        for s in sentences:
            ids = tok.encode(s)
            decoded = tok.decode(ids)
            # sentencepiece BPE roundtrip should recover the sentence exactly
            # (case and punctuation preserved) once special tokens are stripped
            assert decoded == s, f"roundtrip mismatch: {s!r} -> {decoded!r}"
