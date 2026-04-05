import unittest.mock as mock

import torch
from PIL import Image

from vqa.inference import load_model, predict_answer


def test_load_model():
    with (
        mock.patch("vqa.inference.BlipForQuestionAnswering.from_pretrained") as mock_model,
        mock.patch("vqa.inference.AutoProcessor.from_pretrained") as mock_processor,
    ):
        mock_model.return_value = mock.MagicMock()
        mock_processor.return_value = mock.MagicMock()

        model, processor, device = load_model("dummy_path", device="cpu")

        assert model == mock_model.return_value
        assert processor == mock_processor.return_value
        assert device == "cpu"
        mock_model.assert_called_once_with("dummy_path")
        mock_processor.assert_called_once_with("dummy_path")


def test_predict_answer():
    mock_model = mock.MagicMock()
    mock_processor = mock.MagicMock()
    mock_image = mock.MagicMock(spec=Image.Image)

    mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_model.generate.return_value = torch.tensor([[4, 5, 6]])
    mock_processor.decode.return_value = "blue"

    answer = predict_answer(mock_image, "What color is the car?", mock_model, mock_processor, "cpu")

    assert answer == "blue"
    mock_processor.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_processor.decode.assert_called_once_with(mock.ANY, skip_special_tokens=True)
