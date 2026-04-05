import unittest.mock as mock

from vqa.train import main


def test_train_main():
    with (
        mock.patch("vqa.train.train_model") as mock_train,
        mock.patch("sys.argv", ["vqa-train", "--num_epochs", "1", "--batch_size", "2"]),
    ):
        mock_train.return_value = (mock.MagicMock(), mock.MagicMock(), [], [], [])

        main()

        # Verify that train_model was called with the expected arguments from CLI
        # We need to check some of the args
        args, kwargs = mock_train.call_args
        assert kwargs["num_epochs"] == 1
        assert kwargs["batch_size"] == 2
        mock_train.assert_called_once()
