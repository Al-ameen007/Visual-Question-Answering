import unittest.mock as mock

from vqa.data import (
    add_most_common_answer,
    filter_unanswerable,
    update_filename_column,
)


def test_update_filename_column():
    dataset = mock.MagicMock()
    path = "/data/images/"

    # Mocking dataset.map
    dataset.map.side_effect = lambda f: [{"filename": f({"filename": "image1.jpg"})["filename"]}]

    modified_dataset = update_filename_column(dataset, path)

    assert modified_dataset[0]["filename"] == "/data/images/image1.jpg"
    dataset.map.assert_called_once()


def test_filter_unanswerable():
    dataset = mock.MagicMock()

    # Sample data
    data = [{"answer_type": "other"}, {"answer_type": "unanswerable"}, {"answer_type": "yes/no"}]

    # Mocking dataset.filter
    dataset.filter.side_effect = lambda f: [item for item in data if f(item)]

    filtered_dataset = filter_unanswerable(dataset)

    assert len(filtered_dataset) == 2
    assert all(item["answer_type"] != "unanswerable" for item in filtered_dataset)
    dataset.filter.assert_called_once()


def test_add_most_common_answer():
    dataset = [
        {"answers": ["yes", "yes", "no"]},
        {"answers": ["red", "blue", "red", "green"]},
        {"answers": []},
    ]

    # Mocking dataset.add_column since it's a list here for simplicity in test
    # In real datasets, it returns a new dataset.

    with mock.patch("vqa.data.tqdm", side_effect=lambda x, **kwargs: x):
        # We need to mock add_column on a mock dataset object
        mock_dataset = mock.MagicMock()
        mock_dataset.__iter__.return_value = iter(dataset)

        def mock_add_column(name, column):
            # Just verify the logic of add_most_common_answer
            assert name == "max_answer"
            assert column == ["yes", "red", ""]
            return "modified_dataset"

        mock_dataset.add_column.side_effect = mock_add_column

        result = add_most_common_answer(mock_dataset)
        assert result == "modified_dataset"
