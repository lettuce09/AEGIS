"""Small numerical tests for the scientific objectives, without checkpoints."""

import math
import unittest

import torch

from aegis.losses import (
    AttributeConditionalAlignment,
    HistoryAwareWeighting,
    per_sample_segmentation_loss,
)


class SegmentationLossTests(unittest.TestCase):
    def test_per_sample_values_and_single_sample_shape(self):
        logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
        masks = torch.stack([torch.ones(1, 2, 2), torch.zeros(1, 2, 2)])
        actual = per_sample_segmentation_loss(logits, masks)
        expected = torch.tensor([
            1 - (4 + 1e-5) / (5 + 1e-5) + math.log(2),
            1 - 1e-5 / (1 + 1e-5) + math.log(2),
        ])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(per_sample_segmentation_loss(logits[:1], masks[:1]).shape, (1,))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_reject_nonbinary_labels(self):
        with self.assertRaises(ValueError):
            per_sample_segmentation_loss(torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2))


class HistoryAwareWeightingTests(unittest.TestCase):
    def test_ema_weights_and_detached_weight_gradient(self):
        module = HistoryAwareWeighting(2)
        losses = torch.tensor([1.0, 3.0, 4.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1])
        result = module(losses, groups)
        torch.testing.assert_close(module.history, torch.tensor([0.2, 0.4]))
        weights = torch.softmax(torch.tensor([2.0, 4.0]), dim=0)
        torch.testing.assert_close(result, 2 * weights[0] + 4 * weights[1])
        result.backward()
        torch.testing.assert_close(losses.grad, torch.stack([weights[0] / 2, weights[0] / 2, weights[1]]))
        self.assertFalse(module.history.requires_grad)
        self.assertFalse(module.weights.requires_grad)

    def test_missing_group_keeps_history_and_full_softmax_denominator(self):
        module = HistoryAwareWeighting(3)
        module.history.copy_(torch.tensor([0.2, 0.4, 0.6]))
        losses = torch.tensor([2.0], requires_grad=True)
        actual = module(losses, torch.tensor([1]))
        torch.testing.assert_close(module.history, torch.tensor([0.2, 0.56, 0.6]))
        torch.testing.assert_close(actual, module.weights[1] * 2)
        self.assertLess(float(module.weights[1]), 1.0)

    def test_no_update_and_state_restore(self):
        module = HistoryAwareWeighting(2)
        module(torch.tensor([2.0, 3.0]), torch.tensor([0, 1]))
        saved = module.history.clone()
        module(torch.tensor([5.0]), torch.tensor([1]), update=False)
        torch.testing.assert_close(module.history, saved)
        restored = HistoryAwareWeighting(2)
        restored.load_state_dict(module.state_dict())
        torch.testing.assert_close(restored.weights, module.weights)

    def test_invalid_groups_fail_instead_of_silently_mapping(self):
        module = HistoryAwareWeighting(2)
        for groups in (torch.tensor([-1]), torch.tensor([2]), torch.tensor([0.0])):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                module(torch.tensor([1.0]), groups)


class AlignmentTests(unittest.TestCase):
    def test_paper_ordered_sum_and_legacy_scale(self):
        features = torch.tensor([[[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
                                 [[[3.0, 3.0, 3.0]], [[5.0, 5.0, 5.0]]]], requires_grad=True)
        masks = torch.ones(2, 1, 1, 3)
        groups = torch.tensor([0, 1])
        paper = AttributeConditionalAlignment()(features, masks, groups)
        legacy = AttributeConditionalAlignment(pair_reduction="legacy_mse")(features, masks, groups)
        # ||[1,2]-[3,5]||^2 = 13, counted twice in the paper's ordered sum.
        torch.testing.assert_close(paper, torch.tensor(26.0), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(legacy, paper / 4)
        paper.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_biological_background_is_preserved_but_site_is_aligned(self):
        features = torch.tensor([[[[1.0, 1.0, 1.0, 2.0]]],
                                 [[[1.0, 1.0, 1.0, 4.0]]]], requires_grad=True)
        masks = torch.tensor([[[[1, 1, 1, 0]]], [[[1, 1, 1, 0]]]])
        groups = torch.tensor([0, 1])
        module = AttributeConditionalAlignment()
        bio = module(features, masks, groups)
        site = module(features, masks, groups, is_style_attribute=True)
        torch.testing.assert_close(bio, torch.tensor(0.0))
        torch.testing.assert_close(site, torch.tensor(0.4), rtol=1e-5, atol=1e-5)
        bio.backward(retain_graph=True)
        torch.testing.assert_close(features.grad[..., -1], torch.zeros(2, 1, 1))
        site.backward()
        self.assertGreater(float(features.grad[..., -1].abs().sum()), 0)

    def test_threshold_is_strict_and_invalid_lesions_excluded_from_group_mean(self):
        features = torch.tensor([[[[100.0, 100.0, 100.0]]],
                                 [[[2.0, 2.0, 2.0]]],
                                 [[[5.0, 5.0, 5.0]]]], requires_grad=True)
        masks = torch.tensor([[[[1, 1, 0]]], [[[1, 1, 1]]], [[[1, 1, 1]]]])
        result = AttributeConditionalAlignment()(features, masks, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(result, torch.tensor(18.0), rtol=1e-5, atol=1e-5)
        result.backward()
        torch.testing.assert_close(features.grad[0], torch.zeros_like(features.grad[0]))

    def test_one_group_or_no_valid_lesions_has_differentiable_zero(self):
        for groups, mask_value in (([0], 1), ([0, 1], 0)):
            features = torch.randn(len(groups), 3, 2, 2, requires_grad=True)
            masks = torch.full((len(groups), 1, 4, 4), mask_value)
            result = AttributeConditionalAlignment()(features, masks, torch.tensor(groups))
            torch.testing.assert_close(result, torch.tensor(0.0))
            result.backward()
            torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_different_mask_resolution_and_batch_size_one(self):
        features = torch.ones(1, 2, 2, 3, requires_grad=True)
        masks = torch.ones(1, 1, 4, 6)
        result = AttributeConditionalAlignment()(features, masks, torch.tensor([0]), True)
        self.assertEqual(result.ndim, 0)
        result.backward()
        self.assertIsNotNone(features.grad)

    def test_negative_group_rejected(self):
        with self.assertRaises(ValueError):
            AttributeConditionalAlignment()(torch.ones(1, 2, 2, 2), torch.ones(1, 1, 2, 2), torch.tensor([-1]))


if __name__ == "__main__":
    unittest.main()
