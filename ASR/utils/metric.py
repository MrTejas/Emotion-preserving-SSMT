# utils/metrics.py

import Levenshtein

def wer(refs: list[str], hyps: list[str]) -> float:
    total_words = 0
    total_edits = 0
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_words += len(ref_words)
        total_edits += Levenshtein.distance(" ".join(ref_words), " ".join(hyp_words))
    return total_edits / total_words if total_words > 0 else 0

def cer(refs: list[str], hyps: list[str]) -> float:
    total_chars = 0
    total_edits = 0
    for ref, hyp in zip(refs, hyps):
        total_chars += len(ref.strip())
        total_edits += Levenshtein.distance(ref.strip(), hyp.strip())
    return total_edits / total_chars if total_chars > 0 else 0

def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    wer_score = wer(refs, preds)
    cer_score = cer(refs, preds)
    return {
        "WER": round(wer_score * 100, 2),
        "CER": round(cer_score * 100, 2),
        "Samples Used": len(refs),
    }
