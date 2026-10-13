"""Reward IoU cho GRPO (E5, GUIDE bước 9). Python thuần: test được không cần GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score import IOU_THR, iou_xyxy, parse_output  # noqa: E402

VALID_BONUS = 0.2   # thưởng cho JSON hợp lệ
IOU_WEIGHT = 0.8    # phần còn lại theo tổng IoU các cặp khớp


def iou_reward(completion, gt_boxes, iou_thr=IOU_THR):
    """Reward của một output thô so với gt_boxes (list[{"label","bbox"}], cùng định dạng parse_output)."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0                                   # JSON hỏng
    if not gt_boxes and not pred:
        return 1.0
    # ghép đôi tham lam theo IoU giảm dần, chỉ cùng lớp, chỉ nhận IoU >= ngưỡng
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total_iou = set(), set(), 0.0
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total_iou += v
    # mẫu số max(số gt, số pred): box bịa hoặc box bỏ sót đều kéo reward xuống
    return VALID_BONUS + IOU_WEIGHT * total_iou / max(len(gt_boxes), len(pred), 1)


def completion_text(completion):
    """GRPOTrainer trả completion dạng chuỗi hoặc hội thoại [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def grpo_reward_func(completions, gt, **kwargs):
    """Hàm reward theo chữ ký của trl.GRPOTrainer; cột `gt` của dataset là JSON đáp án thang 0–1000."""
    return [iou_reward(completion_text(c), parse_output(g) or []) for c, g in zip(completions, gt)]
