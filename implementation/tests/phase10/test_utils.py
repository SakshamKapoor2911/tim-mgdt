import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib.util

# Mock torch BEFORE importing anything else
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Import utils.py directly without going through the package __init__
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../code/phases/phase10/utils.py'))
spec = importlib.util.spec_from_file_location("utils_direct", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

class TestGetDevice(unittest.TestCase):
    def test_get_device_cuda(self):
        # Reset mocks
        mock_torch.cuda.is_available.reset_mock()
        mock_torch.backends.mps.is_available.reset_mock()

        # Setup: CUDA is available
        mock_torch.cuda.is_available.return_value = True

        # Action
        device = utils_module.get_device()

        # Assert
        self.assertEqual(device, "cuda")
        mock_torch.cuda.is_available.assert_called_once()

    def test_get_device_mps(self):
        # Reset mocks
        mock_torch.cuda.is_available.reset_mock()
        mock_torch.backends.mps.is_available.reset_mock()

        # Setup: CUDA is NOT available, but MPS is
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        # Action
        device = utils_module.get_device()

        # Assert
        self.assertEqual(device, "mps")
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.backends.mps.is_available.assert_called_once()

    def test_get_device_cpu(self):
        # Reset mocks
        mock_torch.cuda.is_available.reset_mock()
        mock_torch.backends.mps.is_available.reset_mock()

        # Setup: Neither CUDA nor MPS is available
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Action
        device = utils_module.get_device()

        # Assert
        self.assertEqual(device, "cpu")
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.backends.mps.is_available.assert_called_once()

if __name__ == '__main__':
    unittest.main()
