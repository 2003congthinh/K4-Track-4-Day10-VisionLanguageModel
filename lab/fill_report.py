"""Điền số liệu từ report/evidence/*.json vào REPORT.md (thay các token {{...}}).

Token:  {{<stem>.<trường>}}           vd {{e1.f1}}, {{e4_coord_scale_pixel.ci}}
        {{<stemA>-<stemB>.<trường>}}  hiệu số có dấu, vd {{e2-e1.f1}}
        {{paired:<stemA>:<stemB>}}     hiệu F1 A−B kèm CI95 bootstrap GHÉP CẶP (cần work/<stem>.jsonl)
Trường: f1 ci ciw lo hi fp valid p r iou tp fpn fn n params vram min
        f1_<lớp> p_<lớp> r_<lớp> tp_<lớp> fn_<lớp> fpn_<lớp>   (lớp: pothole, crack, manhole)
Chạy:   python lab/fill_report.py            → ghi REPORT.md, in các token chưa điền được
        python lab/fill_report.py --dry-run  → chỉ in, không ghi
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PAIRED = re.compile(r"\{\{paired:([A-Za-z0-9_.]+):([A-Za-z0-9_.]+)\}\}")
TOKEN = re.compile(r"\{\{([A-Za-z0-9_.]+?)(?:-([A-Za-z0-9_.]+?))?\.([a-z][a-z0-9_]*)\}\}")


def field(d, name):
    m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
    simple = {
        "f1": m["f1"], "p": m["precision"], "r": m["recall"], "tp": m["tp"], "fpn": m["fp"], "fn": m["fn"],
        "lo": d["f1_ci95"][0], "hi": d["f1_ci95"][1], "ciw": d["f1_ci95"][1] - d["f1_ci95"][0],
        "fp": d["fp_per_image"], "valid": d["parse_valid_rate"], "iou": d["mean_iou"], "n": d["n_images"],
        "params": meta.get("trainable_params"), "vram": meta.get("peak_vram_gb"), "min": meta.get("train_minutes"),
    }
    if name in simple:
        return simple[name]
    if name == "ci":
        return f"[{d['f1_ci95'][0]:.3f}, {d['f1_ci95'][1]:.3f}]"
    key, _, cls = name.partition("_")
    if cls in pc:
        return {"f1": pc[cls]["f1"], "p": pc[cls]["precision"], "r": pc[cls]["recall"],
                "tp": pc[cls]["tp"], "fn": pc[cls]["fn"], "fpn": pc[cls]["fp"]}.get(key)
    return None


def fmt(v, name):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if name in ("params",):
        return f"{int(v):,}"
    if name in ("tp", "fpn", "fn", "n") or name.startswith(("tp_", "fn_", "fpn_")):
        return str(int(v))
    if name in ("vram", "min"):
        return f"{v:.2f}"
    if name in ("fp",):
        return f"{v:.2f}"
    return f"{v:.3f}"


def fill(text, ev):
    missing = []

    def sub(mt):
        a, b, name = mt.groups()
        if a not in ev or (b and b not in ev):
            missing.append(mt.group(0))
            return mt.group(0)
        va = field(ev[a], name)
        if b:
            vb = field(ev[b], name)
            if va is None or vb is None:
                missing.append(mt.group(0))
                return mt.group(0)
            return f"{va - vb:+.3f}"
        out = fmt(va, name)
        if out is None:
            missing.append(mt.group(0))
            return mt.group(0)
        return out

    text = TOKEN.sub(sub, text)

    def sub_paired(mt):
        a, b = mt.groups()
        res = paired(a, b, ev)
        if res is None:
            missing.append(mt.group(0))
            return mt.group(0)
        return f"{res['diff']:+.3f} (CI95 ghép cặp [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}])"

    text = PAIRED.sub(sub_paired, text)
    missing += [t for t in re.findall(r"\{\{[^{}]*\}\}", text) if t not in missing]   # không để sót token nào
    return text, missing


def paired(a, b, ev, work=None, splits=ROOT / "data" / "splits"):
    work = Path(work or WORK)
    sys.path.insert(0, str(ROOT))
    from lab.stats import paired_bootstrap, per_image_counts
    from scripts.labcommon import read_jsonl
    pa, pb = work / f"{a}.jsonl", work / f"{b}.jsonl"
    if a not in ev or b not in ev or not pa.exists() or not pb.exists():
        return None
    entries = read_jsonl(splits / "test.jsonl")
    rows = [per_image_counts(entries, {r["image_id"]: r["raw"] for r in read_jsonl(p)}, ev[k].get("coord", "norm1000"))
            for k, p in ((a, pa), (b, pb))]
    return paired_bootstrap(*rows)


def main(argv=None):
    global WORK
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # console Windows (cp1252)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(ROOT / "REPORT.md"))
    ap.add_argument("--evidence", default=str(ROOT / "report" / "evidence"))
    ap.add_argument("--work", default=str(WORK), help="thư mục chứa <stem>.jsonl cho token paired")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    WORK = Path(a.work)
    ev = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.evidence).glob("e*.json"))}
    text = Path(a.report).read_text(encoding="utf-8")
    new, missing = fill(text, ev)
    print(f"bằng chứng: {sorted(ev)}")
    if missing:
        print(f"{len(missing)} token chưa điền được:", sorted(set(missing)))
    if not a.dry_run:
        Path(a.report).write_text(new, encoding="utf-8")
        print("đã ghi", a.report)
    left = re.findall(r"\[\[TODO[^\]]*\]\]", new)
    if left:
        print(f"còn {len(left)} chỗ [[TODO ...]] cần tự viết sau khi xem kết quả")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
