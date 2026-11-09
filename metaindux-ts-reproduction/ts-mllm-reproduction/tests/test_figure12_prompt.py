import numpy as np
import pytest

from ts_mllm.figure12_prompt import build_figure12_prompt


def test_figure12_fields_and_dynamic_statistics():
    window = np.linspace(0, 1, 40, dtype=np.float32)[:, None] * np.ones((1, 14), dtype=np.float32)
    prompt = build_figure12_prompt(window)
    for text in ("Task Describe", "Dataset Feature", "Dataset Describe", "C-MAPSS",
                 "min=0.000", "max=1.000", "median=0.500", "overall trend is upward",
                 "preceding visual feature"):
        assert text in prompt
    assert "true RUL" not in prompt
    assert "operating settings are present" in prompt


def test_figure12_invalid_input():
    with pytest.raises(ValueError):
        build_figure12_prompt(np.zeros((39, 14)))
    data = np.zeros((40, 14)); data[0, 0] = np.nan
    with pytest.raises(ValueError):
        build_figure12_prompt(data)
