#!/usr/bin/env python
"""Translate one sign-language video to English with a trained checkpoint.

    python scripts/translate_video.py demo/videos/8840_en.mp4 \
        --config configs/proposed_lid_selfcond.yaml --model-dir demo/ckpt/proposed/seed3

Same pipeline as training: DINOv2 ViT-S/14 CLS features (every 2nd frame,
224x224), then the model with greedy decoding. Only works on SP-10-style clips.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from extract_features_dinov2 import encode_frames, load_dinov2, read_frames
from src.evaluation.decode import greedy_decode
from src.models.build import build_model
from src.models.layers import key_padding_mask_from_lengths
from src.training.checkpoint import load_checkpoint
from src.training.setup import TXT_VOCAB_FILE, build_tokenizers, language_filter
from src.utils.config import load_config


def load(config_path: str, model_dir: str, device: str) -> dict:
    """Load the translation model, its vocabularies and DINOv2 once."""
    config = load_config(config_path)
    tok, lid_tok = build_tokenizers(config, config["data"]["splits_dir"], language_filter(config),
                                    vocab_path=os.path.join(model_dir, TXT_VOCAB_FILE))
    model, _ = build_model(config, len(tok), tok.pad_id, len(lid_tok) if lid_tok else None)
    load_checkpoint(os.path.join(model_dir, "best.ckpt"), model)
    model.to(device).eval()
    return {"model": model, "tok": tok, "lid_tok": lid_tok, "dino": load_dinov2("dinov2_vits14", device), "device": device}


@torch.no_grad()
def translate(m: dict, path: str) -> dict:
    """Returns {"translation", "sign_language" (code or None), "frames"}."""
    device, model = m["device"], m["model"]
    frames = read_frames(path, stride=2, size=224)
    # features were cached as fp16 for training; round-trip to match
    feats = encode_frames(m["dino"], frames, device, 64).half().float().unsqueeze(0).to(device)
    lengths = torch.tensor([feats.size(1)], device=device)
    ids = greedy_decode(model, feats, lengths, m["tok"].bos_id, m["tok"].eos_id, max_len=40)[0]
    lang = None
    if getattr(model, "use_lid_branch", False):
        _, lid = model.encode(feats, key_padding_mask_from_lengths(lengths, feats.size(1)))
        votes = [i for i in lid.argmax(-1)[0].tolist() if i != 0]  # 0 = CTC blank
        if votes:
            lang = m["lid_tok"].itos[Counter(votes).most_common(1)[0][0]].strip("<>")
    return {"translation": m["tok"].decode(ids), "sign_language": lang, "frames": int(feats.size(1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--config", required=True)
    p.add_argument("--model-dir", required=True, help="holds best.ckpt and txt_vocab.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    m = load(args.config, args.model_dir, args.device)
    for path in args.videos:
        r = translate(m, path)
        print(f"\n{path}  ({r['frames']} frames)")
        print(f"  translation: {r['translation']}")
        if r["sign_language"]:
            print(f"  sign language guess: {r['sign_language']}")


if __name__ == "__main__":
    main()
