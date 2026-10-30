from pathlib import Path
import unittest
from tracking_demo.demo_delivery import fresh_output


class DemoEntryTests(unittest.TestCase):
    def test_protected_experiment_and_review_outputs_rejected(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('E001_c1_baseline','E002_multicam','E003_improvement','E004_identity','E005_temporal_assignment','PHASE5_review'):
            with self.assertRaises(ValueError):fresh_output(root,f'outputs/{name}/new_demo')

    def test_explicit_new_output_required(self):
        root=Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):fresh_output(root,None)
        with self.assertRaises(ValueError):fresh_output(root,'data/new_demo')
        with self.assertRaises(FileExistsError):fresh_output(root,'outputs')
