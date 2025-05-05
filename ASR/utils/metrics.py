# utils/metrics.py

from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip, RemoveWhiteSpace

def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    # Ensure preds and refs are clean flat lists of strings
    assert isinstance(preds, list) and isinstance(refs, list)
    assert all(isinstance(p, str) for p in preds)
    assert all(isinstance(r, str) for r in refs)

    # Transformations
    transformation = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        Strip(),
        RemoveWhiteSpace(replace_by_space=True),
    ])

    # Jiwer expects a flat list of strings (not lists of words)
    wer_score = wer(refs, preds, truth_transform=transformation, hypothesis_transform=transformation)
    cer_score = cer(refs, preds, truth_transform=transformation, hypothesis_transform=transformation)

    return {
        "WER": round(wer_score * 100, 2),
        "CER": round(cer_score * 100, 2),
        "Samples Used": len(refs),
    }
