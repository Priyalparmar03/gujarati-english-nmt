"""
Evaluates a trained checkpoint on a test set with BLEU, chrF, and (optionally) COMET.

Usage:
    python -m src.evaluate --ckpt checkpoints/best.pt --spm_model data/spm_shared.model \
        --src data/test.en --ref data/test.gu --out results/test_hyp.gu \
        --beam_size 4 --use_comet
"""
import argparse
import torch
import sacrebleu

from src.model.transformer import Seq2SeqTransformer
from src.model.tokenizer import SharedTokenizer, PAD_ID, BOS_ID, EOS_ID


def translate_file(model, tok, src_path, device, beam_size=4, max_len=128, batch_greedy=32):
    with open(src_path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    hyps = []
    if beam_size and beam_size > 1:
        for line in lines:
            ids = torch.tensor([tok.encode(line)[:max_len]], device=device)
            out_ids = model.beam_translate(ids, BOS_ID, EOS_ID, beam_size=beam_size, max_len=max_len)
            hyps.append(tok.decode(out_ids[0].tolist()))
    else:
        for i in range(0, len(lines), batch_greedy):
            batch = lines[i : i + batch_greedy]
            enc = [tok.encode(l)[:max_len] for l in batch]
            max_l = max(len(e) for e in enc)
            src = torch.full((len(enc), max_l), PAD_ID, dtype=torch.long)
            for j, e in enumerate(enc):
                src[j, : len(e)] = torch.tensor(e)
            src = src.to(device)
            out_ids = model.greedy_translate(src, BOS_ID, EOS_ID, max_len=max_len)
            hyps.extend(tok.decode(seq.tolist()) for seq in out_ids)
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="results/test_hyp.gu")
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--use_comet", action="store_true")
    ap.add_argument("--direction", choices=["en2gu", "gu2en"], default=None,
                     help="defaults to whatever direction the checkpoint was trained with")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = SharedTokenizer(args.spm_model)

    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    model = Seq2SeqTransformer(
        vocab_size=tok.vocab_size,
        d_model=margs["d_model"],
        n_heads=margs["n_heads"],
        n_layers=margs["n_layers"],
        d_ff=margs["d_ff"],
        dropout=0.0,
        max_len=margs["max_len"] + 2,
        pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    direction = args.direction or margs.get("direction", "en2gu")
    if direction != margs.get("direction", "en2gu"):
        print(f"warning: evaluating in {direction} but checkpoint was trained {margs.get('direction')}")

    hyps = translate_file(model, tok, args.src, device, beam_size=args.beam_size, max_len=args.max_len)

    with open(args.ref, encoding="utf-8") as f:
        refs = [l.rstrip("\n") for l in f]
    with open(args.src, encoding="utf-8") as f:
        srcs = [l.rstrip("\n") for l in f]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    print(f"BLEU:  {bleu.score:.2f}")
    print(f"chrF:  {chrf.score:.2f}")

    if args.use_comet:
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
            comet_out = comet_model.predict(data, batch_size=16, gpus=1 if device.type == "cuda" else 0)
            print(f"COMET: {comet_out.system_score:.4f}")
        except ImportError:
            print("COMET not installed — `pip install unbabel-comet` to enable --use_comet")


if __name__ == "__main__":
    main()
