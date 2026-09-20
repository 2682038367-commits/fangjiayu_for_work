import pytest

torch = pytest.importorskip("torch")

from rul_chronos.encoder import chronos_special_token_indices, sensor_group_ids


def test_sensor_group_ids_keep_each_multivariate_prefix_joint():
    actual = sensor_group_ids(batch_size=2, num_sensors=21, device="cpu")
    expected = torch.tensor([0] * 21 + [1] * 21)
    assert torch.equal(actual, expected)


def test_sensor_group_ids_reject_empty_dimensions():
    with pytest.raises(ValueError):
        sensor_group_ids(batch_size=0, num_sensors=21, device="cpu")


def test_chronos_special_token_indices_with_reg_and_one_future_query():
    reg, future, expected_length = chronos_special_token_indices(
        num_context_patches=8,
        use_reg_token=True,
        num_output_patches=1,
    )
    assert reg == 8
    assert future == 9
    assert expected_length == 10


def test_chronos_special_token_indices_without_reg():
    reg, future, expected_length = chronos_special_token_indices(
        num_context_patches=8,
        use_reg_token=False,
        num_output_patches=1,
    )
    assert reg is None
    assert future == 8
    assert expected_length == 9
