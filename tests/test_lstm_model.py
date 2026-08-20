from __future__ import annotations

import unittest

import torch

from src.modeling.lstm_pipeline import DailyLoadLSTM


class LSTMModelTests(unittest.TestCase):
    def test_lstm_output_shape_is_24_hours(self) -> None:
        model = DailyLoadLSTM(context_size=5, horizon=24, hidden_size=8)
        output = model(
            history=torch.zeros((2, 336, 1)),
            context=torch.zeros((2, 5)),
        )
        self.assertEqual(tuple(output.shape), (2, 24))


if __name__ == "__main__":
    unittest.main()
