"""Check that model selection can expose a neglected class despite high accuracy."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import evaluate


def test_macro_recall_exposes_missing_rose_predictions():
    # Nine easy tulips can conceal a completely missed rose in overall accuracy.
    logits = torch.tensor([[0.0, 4.0]] * 10)
    labels = torch.tensor([0] + [1] * 9)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=3)
    result = evaluate(nn.Identity(), loader, ["rose", "tulip"])
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["per_class_recall"] == {"rose": 0.0, "tulip": 1.0}
    assert result["confusion_matrix"] == [[0, 1], [0, 9]]


def test_evaluation_uses_class_order_and_all_batches():
    loader = DataLoader(TensorDataset(torch.eye(3) * 5, torch.arange(3)), batch_size=2)
    result = evaluate(nn.Identity(), loader, ["tulip", "rose", "daisy"])
    assert result["count"] == 3
    assert result["macro_recall"] == 1.0
    assert result["per_class_recall"] == {"tulip": 1.0, "rose": 1.0, "daisy": 1.0}
