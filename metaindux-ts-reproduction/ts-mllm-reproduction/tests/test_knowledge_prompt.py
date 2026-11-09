import numpy as np
import pytest
from ts_mllm.knowledge_prompt import build_knowledge_prompt, KnowledgeWindowDataset


def test_context_has_template_fields_actual_conditions_and_switches():
    x = np.zeros((40,14), dtype=np.float32)
    settings = np.array([[0,0,100]] * 20 + [[10,0.25,100]] * 20)
    text = build_knowledge_prompt(x, settings, "FD002")
    for field in ("Dataset description:", "###Domain:", "###Instruction:", "switches=1", "(0,0,100):20/40", "(10,0.25,100):20/40"):
        assert field in text
    assert "globally Min-Max" in text
    assert "true RUL" not in text


def test_short_observed_window_is_not_counted_as_40_observations():
    text = build_knowledge_prompt(np.zeros((40,14)), np.array([[0,0,100]] * 12), "FD001")
    assert "12/12" in text and "switches=0" in text
    assert "first 28 rows are padding" in text
    sequence = text.split("oldest to newest: ")[1].split(".")[0].split()
    assert sequence == ["A"]*12


def test_ordered_codes_preserve_every_cycle_and_distinguish_order():
    x = np.zeros((40,14))
    a, b = [0,0,100], [10,0.25,100]
    text = build_knowledge_prompt(x, np.array([a,b]*20), "FD002")
    sequence = text.split("oldest to newest: ")[1].split(".")[0].split()
    assert sequence == ["A","B"]*20
    assert "A=(0,0,100)" in text and "B=(10,0.25,100)" in text
    assert "first 0 rows are padding" in text
    assert text != build_knowledge_prompt(x, np.array([a]*20+[b]*20), "FD002")


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        build_knowledge_prompt(np.zeros((40,14)), np.zeros((40,2)), "FD002")


def test_condition_prompt_declares_correct_scale_and_keeps_order():
    from ts_mllm.condition_knowledge_prompt import build_condition_knowledge_prompt
    settings=np.array([[0,0,100],[10,0.25,100]]*20)
    text=build_condition_knowledge_prompt(np.zeros((40,14)),settings,"FD002")
    assert "operating-condition-wise Min-Max" in text
    assert "globally Min-Max" not in text
    assert "do not interpret these as raw absolute sensor values" in text
    assert text.split("oldest to newest: ")[1].split(".")[0].split()==["A","B"]*20
