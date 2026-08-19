from __future__ import annotations

import json
from pathlib import Path
import unittest

from PIL import Image

from question_bbox_monitor import (
    QUESTION_BBOX_CATEGORIES,
    extract_question_bbox,
    has_complete_tool_call,
    parse_tool_call,
    synthetic_tool_call,
)


class QuestionBboxMonitorTest(unittest.TestCase):
    def test_real_and_synthetic_tool_calls(self) -> None:
        response = (
            '<tool_call>{"type":"tool_call","tool_name":"zoom_in",'
            '"arguments":{"bbox_2d":[1,2,30,40]}}</tool_call>'
        )
        self.assertTrue(has_complete_tool_call(response))
        self.assertEqual(parse_tool_call(response)["arguments"]["bbox_2d"], [1, 2, 30, 40])

        annotated_bbox = [5667, 2341, 6107, 3063]
        generated = synthetic_tool_call(annotated_bbox)
        self.assertEqual(parse_tool_call(generated)["arguments"]["bbox_2d"], annotated_bbox)

    def test_all_question_bboxes_fit_their_source_images(self) -> None:
        dataset_root = Path(__file__).resolve().parents[1] / "xlrsbench650"
        checked = 0
        for annotation_path in dataset_root.glob("*.json"):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("category") not in QUESTION_BBOX_CATEGORIES:
                continue
            bbox = extract_question_bbox(annotation["question"])
            with Image.open(dataset_root / annotation["image_path"]) as image:
                self.assertLessEqual(bbox[2], image.width)
                self.assertLessEqual(bbox[3], image.height)
            checked += 1
        self.assertEqual(checked, 150)


if __name__ == "__main__":
    unittest.main()
