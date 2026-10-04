import unittest

import torch

from irrm_codec.multitask_losses import IRRMCodecMultiTaskLoss


class TCRemPGeometryLossTest(unittest.TestCase):
    def test_pairwise_loss_is_zero_for_exact_prediction(self):
        target = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
        loss = IRRMCodecMultiTaskLoss._pairwise_log_cosine_distance(target, target)
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_pairwise_loss_detects_collapsed_geometry(self):
        target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        prediction = torch.ones_like(target)
        loss = IRRMCodecMultiTaskLoss._pairwise_log_cosine_distance(prediction, target)
        self.assertGreater(float(loss), 1.0)

    def test_geometry_weights_are_validated(self):
        with self.assertRaises(ValueError):
            IRRMCodecMultiTaskLoss(
                tcremp_centered_cosine_weight=0.6,
                tcremp_pairwise_log_distance_weight=0.5,
            )


if __name__ == "__main__":
    unittest.main()