"""Bootstrap GHÉP CẶP cho hiệu F1 giữa hai setup trên cùng tập ảnh test (dùng lại logic chấm của score.py)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score import evaluate  # noqa: E402


def per_image_counts(entries, preds, coord="norm1000"):
    """(tp, fp, fn) của từng ảnh, tính bằng chính scripts.score.evaluate trên một ảnh."""
    out = []
    for e in entries:
        r = evaluate([e], {e["image_id"]: preds[e["image_id"]]} if e["image_id"] in preds else {},
                     coord=coord, n_boot=1)["micro"]
        out.append((r["tp"], r["fp"], r["fn"]))
    return out


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def paired_bootstrap(rows_a, rows_b, n_boot=1000, seed=0):
    """Hiệu F1 (A − B) và CI95 khi lấy mẫu lại CÙNG chỉ số ảnh cho cả hai setup."""
    assert len(rows_a) == len(rows_b)
    n, rng = len(rows_a), random.Random(seed)

    def f1_of(rows, idx):
        return _f1(sum(rows[i][0] for i in idx), sum(rows[i][1] for i in idx), sum(rows[i][2] for i in idx))

    full = range(n)
    diff = f1_of(rows_a, full) - f1_of(rows_b, full)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(f1_of(rows_a, idx) - f1_of(rows_b, idx))
    vals.sort()
    lo, hi = vals[int(0.025 * n_boot)], vals[max(int(0.975 * n_boot) - 1, 0)]
    return {"diff": diff, "ci95": [lo, hi], "p_a_better": sum(v > 0 for v in vals) / n_boot}
