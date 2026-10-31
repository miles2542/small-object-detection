"""Contract checks independent of model weights or pycocotools."""

import math
import unittest
from copy import deepcopy

from src.eval.protocol import area_memberships, evaluation_protocol, prepare_ground_truth


class EvaluationProtocolTests(unittest.TestCase):
    def test_closed_boundaries_and_neighbors(self):
        for boundary, left, right in [
            (64, "verytiny", "tiny"),
            (256, "tiny", "small"),
            (1024, "small", "medium"),
        ]:
            self.assertEqual(area_memberships(boundary), ["all", left, right])
            self.assertEqual(area_memberships(math.nextafter(boundary, -math.inf)), ["all", left])
            self.assertEqual(area_memberships(math.nextafter(boundary, math.inf)), ["all", right])
        self.assertEqual(area_memberships(4), ["all", "verytiny"])
        self.assertEqual(area_memberships(math.nextafter(4, -math.inf)), ["all"])
        self.assertEqual(area_memberships(4096), ["all", "medium"])
        self.assertEqual(area_memberships(math.nextafter(4096, math.inf)), ["all"])
        self.assertEqual(area_memberships(1, "upstream"), ["all", "verytiny"])
        self.assertEqual(area_memberships(10000, "upstream"), ["all", "medium"])
        self.assertEqual(area_memberships(float("nan")), [])

    def test_protocol_is_fresh_and_rejects_unknown_name(self):
        first = evaluation_protocol()
        first["max_dets"].append(9999)
        self.assertEqual(evaluation_protocol()["max_dets"], [1, 100, 1500])
        with self.assertRaises(ValueError):
            evaluation_protocol("unknown")

    def test_ground_truth_preserves_ids_geometry_and_source(self):
        source = {
            "images": [{"id": 0}, {"id": 71}],
            "categories": [{"id": 9}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 9, "bbox": [-1, 2, 4, 4], "area": 16},
                {"id": 1, "image_id": 0, "category_id": 9, "bbox": [2, 2, 0, 4], "area": 0},
                {"id": 2, "image_id": 71, "category_id": 9, "bbox": [1, 1, 4, 4], "area": 16},
            ],
        }
        original = deepcopy(source)
        selected = prepare_ground_truth(source, [0])
        self.assertEqual(source, original)
        self.assertEqual(selected["images"], [{"id": 0}])
        self.assertEqual(len(selected["annotations"]), 1)
        ann = selected["annotations"][0]
        self.assertEqual(
            (ann["id"], ann["source_annotation_id"], ann["image_id"], ann["category_id"]),
            (1, 0, 0, 9),
        )
        self.assertEqual(ann["bbox"], [-1, 2, 4, 4])
        self.assertEqual(prepare_ground_truth(selected), selected)
        for selection in [[0, 0], [99]]:
            with self.assertRaises(ValueError):
                prepare_ground_truth(source, selection)
        source["annotations"][0]["bbox"][0] = float("inf")
        with self.assertRaises(ValueError):
            prepare_ground_truth(source)


if __name__ == "__main__":
    unittest.main()
