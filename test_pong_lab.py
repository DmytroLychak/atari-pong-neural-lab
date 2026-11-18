"""Checks for observation isolation and evaluation integrity."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import pong_lab as lab


class PongTests(unittest.TestCase):
    def test_no_ball_and_finite_features(self):
        f=lab.Features()(np.zeros((210,160,3),dtype=np.uint8))
        self.assertEqual(f.shape,(7,))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f[5],0)

    def test_scoreboard_does_not_change_features(self):
        frame=np.zeros((210,160,3),dtype=np.uint8)
        frame[110:126,140:144]=[92,186,92]
        frame[90:94,80:82]=236
        other=frame.copy()
        other[:34]=236
        np.testing.assert_array_equal(lab.Features()(frame),lab.Features()(other))

    def test_neural_evaluation_does_not_call_teacher(self):
        # A complete real Atari episode must run with the teacher disabled.
        torch.set_num_threads(1)
        model=lab.load_model(lab.ROOT/'models/policy.pt')
        with patch.object(lab,'teacher',side_effect=AssertionError('teacher called')):
            row, x, y=lab.episode(20000,'neural',model)
        self.assertEqual(len(x),0)
        self.assertEqual(len(y),0)
        self.assertTrue(row['terminated'] or row['truncated'])
        self.assertEqual(row['return_'],row['player_score']-row['opponent_score'])


if __name__=='__main__':
    unittest.main()
