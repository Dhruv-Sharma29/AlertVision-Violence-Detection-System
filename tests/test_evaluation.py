import tempfile
import unittest
from pathlib import Path
from evaluation import export_metrics

class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix_and_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "synthetic-test-model"
            model.write_bytes(b"fixture only, not model weights")
            output = Path(directory) / "result.json"
            result = export_metrics([0, 0, 1, 1], [.1, .7, .4, .9], output, dataset="test fixture", model_path=model)
            self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
            for key in ("accuracy", "precision", "recall", "f1"):
                self.assertEqual(result[key], .5)
            self.assertEqual(len(result["model_sha256"]), 64)
            self.assertTrue(output.is_file())

    def test_invalid_inputs_cannot_publish_metrics(self):
        for labels, probabilities in [([], []), ([1], []), ([2], [.5]), ([.9], [.5]), ([1], [float("nan")]), ([0], [1.1])]:
            with self.assertRaises(ValueError):
                export_metrics(labels, probabilities, "unused.json", dataset="test", model_path="missing")

if __name__ == "__main__":
    unittest.main()
