import unittest.mock as mock

import torch

from vqa.engine import (
    calc_loss_batch,
    calc_loss_loader,
    prepare_model,
    save_checkpoint,
    save_model,
)


def test_prepare_model():
    with (
        mock.patch("vqa.engine.BlipForQuestionAnswering.from_pretrained") as mock_model,
        mock.patch("vqa.engine.AutoProcessor.from_pretrained") as mock_processor,
    ):
        mock_model.return_value = mock.MagicMock()
        mock_processor.return_value = mock.MagicMock()

        model, processor, device = prepare_model("dummy_model", device="cpu")

        assert model == mock_model.return_value
        assert processor == mock_processor.return_value
        assert device == "cpu"
        mock_model.assert_called_once_with("dummy_model")
        mock_processor.assert_called_once_with("dummy_model")


def test_calc_loss_batch():
    mock_model = mock.MagicMock()
    mock_output = mock.MagicMock()
    mock_output.loss = torch.tensor(0.5)
    mock_model.return_value = mock_output

    batch = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}
    device = "cpu"

    loss = calc_loss_batch(batch, mock_model, device)

    assert loss == mock_output.loss
    mock_model.assert_called_once()


def test_calc_loss_loader():
    mock_model = mock.MagicMock()
    mock_output = mock.MagicMock()
    mock_output.loss = torch.tensor(0.5)
    mock_model.return_value = mock_output

    dataloader = [
        {"input_ids": torch.tensor([1]), "labels": torch.tensor([1])},
        {"input_ids": torch.tensor([2]), "labels": torch.tensor([2])},
    ]

    loss = calc_loss_loader(dataloader, mock_model, "cpu")

    assert loss == 0.5
    assert mock_model.call_count == 2


def test_save_model(tmp_path):
    mock_model = mock.MagicMock()
    mock_processor = mock.MagicMock()

    save_dir = tmp_path / "checkpoints"

    save_path = save_model(mock_model, mock_processor, save_dir, epoch=1)

    assert "checkpoint-epoch-1" in save_path
    mock_model.save_pretrained.assert_called_once()
    mock_processor.save_pretrained.assert_called_once()


def test_save_checkpoint(tmp_path):
    mock_model = mock.MagicMock()
    mock_optimizer = mock.MagicMock()

    save_dir = tmp_path / "checkpoints"

    with mock.patch("torch.save") as mock_torch_save:
        checkpoint_path = save_checkpoint(mock_model, mock_optimizer, 1, 0.5, 0.6, save_dir)

        assert "checkpoint-epoch-1.pt" in checkpoint_path
        mock_torch_save.assert_called_once()
