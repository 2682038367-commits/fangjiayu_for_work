import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ts_mllm.spectrum_cache import CachedSpectrumDataset


@pytest.mark.parametrize("dtype,value,scale", [(np.float32, 0.5, 1), (np.uint8, 128, 255)])
def test_float32_spectra_are_not_divided_by255(tmp_path, dtype, value, scale):
    np.save(tmp_path / "train.npy", np.full((1, 3, 114, 114), value, dtype=dtype))
    (tmp_path / "manifest.json").write_text(json.dumps({"dtype": np.dtype(dtype).name, "scale": scale}))
    cached = CachedSpectrumDataset([{"x": torch.zeros(40, 14)}], tmp_path / "train.npy")
    torch.testing.assert_close(cached[0]["image"], torch.full((3, 114, 114), float(value / scale)))
