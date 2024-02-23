import unittest
from unittest import mock

from dataset import SpeechCommandsDataset, transform


class TestDatasetLoading(unittest.TestCase):
    def test_speech_commands_dataset_initialization(self):
        mock_root = "mock/dataset/directory"
        mock_transform = mock.MagicMock()
        with mock.patch("dataset.SpeechCommandsDataset.__init__", return_value=None) as mock_init:
            SpeechCommandsDataset(root=mock_root, transform=mock_transform)
            mock_init.assert_called_with(root=mock_root, transform=mock_transform)

    def test_transform_function_application(self):
        mock_data = {"audio": "mockAudioData", "sample_rate": 16000}
        expected_transformed_data = {"transformed_audio": "transformedMockAudioData"}
        mock_transform = mock.MagicMock(return_value=expected_transformed_data)
        actual_transformed_data = transform(mock_data, desired_samples=16000, labels={"yes": 1, "no": 2})
        self.assertEqual(actual_transformed_data, expected_transformed_data)

    def test_dataset_splitting(self):
        mock_dataset_size = 100
        with mock.patch("torch.utils.data.Dataset.__len__", return_value=mock_dataset_size):
            dataset = SpeechCommandsDataset(root="mock/dataset/directory", transform=transform)
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            self.assertEqual(train_size, 80)
            self.assertEqual(val_size, 20)

if __name__ == '__main__':
    unittest.main()
