"""Test reward IoU (E5) — viết và chạy TRƯỚC khi chạy GRPO: python -m pytest -q lab"""
import json

import pytest

from lab.reward import completion_text, grpo_reward_func, iou_reward
from scripts.score import parse_output


def _raw(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


GT = parse_output(_raw(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))))


def test_perfect_prediction_gets_one():
    assert iou_reward(_raw(("crack", (500, 500, 700, 600)), ("pothole", (100, 100, 200, 200))), GT) \
        == pytest.approx(1.0)


def test_broken_json_gets_zero():
    assert iou_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert iou_reward("I see a pothole.", GT) == 0.0
    assert iou_reward(None, GT) == 0.0


def test_degenerate_box_is_broken_json():
    # parse_output coi box x2 <= x1 là hỏng → reward 0
    assert iou_reward(_raw(("pothole", (200, 100, 100, 200))), GT) == 0.0


def test_fenced_json_still_parses():
    raw = "```json\n" + _raw(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))) + "\n```"
    assert iou_reward(raw, GT) == pytest.approx(1.0)


def test_wrong_class_only_gets_valid_bonus():
    raw = _raw(("manhole", (100, 100, 200, 200)), ("manhole", (500, 500, 700, 600)))
    assert iou_reward(raw, GT) == pytest.approx(0.2)


def test_extra_hallucinated_box_is_penalised_by_denominator():
    raw = _raw(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600)),
               ("manhole", (0, 0, 50, 50)))
    assert iou_reward(raw, GT) == pytest.approx(0.2 + 0.8 * 2 / 3)


def test_missing_box_is_penalised():
    assert iou_reward(_raw(("pothole", (100, 100, 200, 200))), GT) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_empty_prediction_with_gt_gets_only_bonus():
    assert iou_reward("[]", GT) == pytest.approx(0.2)


def test_empty_prediction_without_gt_gets_one():
    assert iou_reward("[]", []) == 1.0


def test_prediction_without_gt_is_penalised():
    assert iou_reward(_raw(("crack", (1, 1, 9, 9))), []) == pytest.approx(0.2)


def test_giant_box_does_not_hack_the_reward():
    assert iou_reward(_raw(("pothole", (0, 0, 1000, 1000))), GT) == pytest.approx(0.2)


def test_partial_iou_below_threshold_not_matched_above_threshold_counted():
    gt = parse_output(_raw(("pothole", (0, 0, 100, 100))))
    # IoU = 0.6 → khớp, đóng góp đúng 0.6
    assert iou_reward(_raw(("pothole", (0, 0, 100, 60))), gt) == pytest.approx(0.2 + 0.8 * 0.6)
    # IoU = 0.4 → không khớp
    assert iou_reward(_raw(("pothole", (0, 0, 100, 40))), gt) == pytest.approx(0.2)


def test_duplicate_boxes_match_once():
    gt = parse_output(_raw(("pothole", (0, 0, 100, 100))))
    raw = _raw(("pothole", (0, 0, 100, 100)), ("pothole", (0, 0, 100, 100)))
    assert iou_reward(raw, gt) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_reward_is_bounded():
    for raw in ("[]", _raw(("crack", (500, 500, 700, 600))), "garbage"):
        assert 0.0 <= iou_reward(raw, GT) <= 1.0


def test_completion_text_handles_conversational_formats():
    assert completion_text("[]") == "[]"
    assert completion_text([{"role": "assistant", "content": "[]"}]) == "[]"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "[]"}]}]) == "[]"


def test_grpo_reward_func_signature_and_values():
    gt_json = _raw(("pothole", (100, 100, 200, 200)))
    completions = [[{"role": "assistant", "content": gt_json}],
                   [{"role": "assistant", "content": "oops"}]]
    out = grpo_reward_func(prompts=None, completions=completions, gt=[gt_json, gt_json], trainer_state=None)
    assert out == [pytest.approx(1.0), 0.0]
