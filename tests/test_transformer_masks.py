"""Unit tests for mask correctness in the from-scratch transformer."""
import torch
from src.model.transformer import Seq2SeqTransformer
from src.model.tokenizer import PAD_ID


def make_model():
    return Seq2SeqTransformer(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, pad_id=PAD_ID)


def test_src_mask_blocks_padding():
    model = make_model()
    # batch of 1, seq len 5, last 2 tokens are padding
    src = torch.tensor([[4, 5, 6, PAD_ID, PAD_ID]])
    mask = model.make_src_mask(src)  # (B, 1, 1, T)
    assert mask.shape == (1, 1, 1, 5)
    assert mask[0, 0, 0, :3].all()
    assert not mask[0, 0, 0, 3:].any()


def test_tgt_mask_is_causal_and_blocks_padding():
    model = make_model()
    tgt = torch.tensor([[2, 7, 8, PAD_ID]])  # bos, tok, tok, pad
    mask = model.make_tgt_mask(tgt)  # (B, 1, T, T)
    assert mask.shape == (1, 1, 4, 4)

    # explicit causal check on a clean (no padding) sequence
    clean_tgt = torch.tensor([[2, 7, 8, 9]])
    clean_mask = model.make_tgt_mask(clean_tgt)
    for i in range(4):
        assert clean_mask[0, 0, i, : i + 1].all()
        if i < 3:
            assert not clean_mask[0, 0, i, i + 1 :].any()

    # padding column should be masked out for every query position
    assert not mask[0, 0, :, 3].any()


def test_forward_pass_shapes():
    model = make_model()
    src = torch.tensor([[4, 5, 6, PAD_ID]])
    tgt_in = torch.tensor([[2, 7, 8]])
    logits = model(src, tgt_in)
    assert logits.shape == (1, 3, 50)
