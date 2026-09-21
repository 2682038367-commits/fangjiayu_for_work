import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ts_mllm.qwen_multimodal import (
    SpectrumTextProjector,
    build_dynamic_prompt,
    select_cache_tokens,
    DomainKnowledgeEmbedding,
    MultimodalCacheConfig,
    QwenTextBridge,
    compose_multimodal_embeddings,
)


def test_dynamic_prompt_is_deterministic_and_has_no_target_value() -> None:
    window = np.linspace(0, 1, 40 * 14, dtype=np.float32).reshape(40, 14)
    prompt = build_dynamic_prompt(window)
    assert prompt == build_dynamic_prompt(window)
    assert "turbofan" in prompt
    assert "40-cycle" in prompt
    assert "target=" not in prompt.lower()
    assert len(prompt) < 1000


def test_projector_and_token_selection_shapes() -> None:
    projector = SpectrumTextProjector()
    assert projector(torch.rand(3, 128)).shape == (3, 1024)
    hidden = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4)
    text_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    selected, mask = select_cache_tokens(hidden, text_mask, cached_text_tokens=3)
    assert selected.shape == (2, 4, 4)
    assert mask.all()
    torch.testing.assert_close(selected[0, 1:], hidden[0, 2:5])
    torch.testing.assert_close(selected[1, 1:], hidden[1, 4:7])


def test_linear_projector_and_96_dim_text_module() -> None:
    projector = SpectrumTextProjector()
    assert isinstance(projector.network, torch.nn.Linear)
    text = DomainKnowledgeEmbedding(vocabulary_size=100)
    assert text(torch.ones(2, 512, dtype=torch.long)).shape == (2, 512, 96)
    with pytest.raises(ValueError):
        text(torch.ones(2, 513, dtype=torch.long))


def test_full_length_cache_keeps_all_text_and_masks_padding() -> None:
    config = MultimodalCacheConfig()
    assert config.max_text_tokens == 512
    hidden = torch.randn(2, 513, 8)
    text_mask = torch.zeros(2, 512, dtype=torch.long)
    text_mask[0, :210] = 1
    text_mask[1] = 1
    selected, mask = select_cache_tokens(hidden, text_mask, config.cached_text_tokens)
    assert selected.shape == (2, 513, 8)
    torch.testing.assert_close(selected[0, :211], hidden[0, :211])
    assert not mask[0, 211:].any()
    torch.testing.assert_close(selected[1], hidden[1])


def test_actual_dke_path_and_linear_projector_receive_gradients() -> None:
    from ts_mllm.alignment import alignment_losses
    adapter = QwenTextBridge(120)
    projector = SpectrumTextProjector()
    ids = torch.randint(0, 120, (2, 8))
    mask = torch.ones_like(ids)
    features = torch.randn(2, 128)
    inputs = compose_multimodal_embeddings(features, ids, projector, adapter, torch.float32)
    assert inputs.shape == (2, 9, 1024)
    assert adapter.dke(ids).shape == (2, 8, 96)
    teacher = torch.randn(2, 8, 1024)
    a, b = alignment_losses(adapter, projector, ids, mask, features, teacher)
    (a + b).backward()
    assert adapter.dke.embedding.weight.grad is not None
    assert adapter.dke.position.grad is not None
    assert adapter.bridge.weight.grad is not None
    assert projector.network.weight.grad is not None


def test_teacher_initialization_keeps96_dims_with_small_training_vocabulary() -> None:
    from ts_mllm.alignment import initialize_from_teacher
    adapter = QwenTextBridge(120)
    teacher = torch.nn.Embedding(120, 1024).requires_grad_(False)
    ids = torch.arange(12)
    initialize_from_teacher(adapter, teacher, ids)
    assert adapter.dke.embedding.weight.shape == (120, 96)
    torch.testing.assert_close(adapter(ids[None]), teacher(ids[None]), atol=1e-4, rtol=1e-4)


def test_cache_writer_uses_bridged_dke_not_native_embeddings(tmp_path) -> None:
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from ts_mllm.qwen_cache import cache_split

    class FakeQwen:
        dtype = torch.float32

        def get_input_embeddings(self):
            raise AssertionError("must not bypass the96-d DKE path")

        def __call__(self, *, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class FakeTokenizer:
        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.ones(len(prompts), 3, dtype=torch.long),
                    "attention_mask": torch.ones(len(prompts), 3, dtype=torch.long)}

    class FakeVision(torch.nn.Module):
        def forward(self, images):
            return torch.ones(len(images), 128)

    rows = [{"x": torch.zeros(40, 14), "image": torch.zeros(3, 114, 114),
             "target": torch.tensor(25.), "unit": torch.tensor(i), "cycle": torch.tensor(40)} for i in range(2)]
    report = cache_split(
        split="test", loader=DataLoader(rows, batch_size=2), dataset_length=2,
        output_dir=tmp_path, vision_encoder=FakeVision(), projector=SpectrumTextProjector(),
        qwen=FakeQwen(), tokenizer=FakeTokenizer(), config=MultimodalCacheConfig(),
        device=torch.device("cpu"), text_adapter=QwenTextBridge(4),
    )
    assert report["token_shape"] == [2, 513, 1024]
    assert np.load(tmp_path / "test_mask.npy").sum(axis=1).tolist() == [4, 4]


@pytest.mark.parametrize(
    ("input_modality", "expected_length", "needs_vision", "needs_tokenizer"),
    [("text_only", 3, False, True), ("visual_only", 1, True, False)],
)
def test_input_level_modality_ablation_excludes_removed_qwen_input(
    tmp_path, input_modality, expected_length, needs_vision, needs_tokenizer,
) -> None:
    """The ablations must happen before Qwen, not by masking its output tokens."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from ts_mllm.qwen_cache import cache_split

    class FakeQwen:
        dtype = torch.float32

        def __init__(self):
            self.input_lengths = []

        def get_input_embeddings(self):
            return torch.nn.Embedding(4, 1024)

        def __call__(self, *, inputs_embeds, **kwargs):
            self.input_lengths.append(inputs_embeds.shape[1])
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class FakeTokenizer:
        def __init__(self):
            self.called = False

        def __call__(self, prompts, **kwargs):
            self.called = True
            return {"input_ids": torch.ones(len(prompts), 3, dtype=torch.long),
                    "attention_mask": torch.ones(len(prompts), 3, dtype=torch.long)}

    class FakeVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def forward(self, images):
            self.called = True
            return torch.ones(len(images), 128)

    row = {"x": torch.zeros(40, 14), "target": torch.tensor(25.),
           "unit": torch.tensor(1), "cycle": torch.tensor(40)}
    if input_modality == "text_only":
        row["prompt"] = "only text"
    else:
        row["image"] = torch.zeros(3, 114, 114)
    qwen, tokenizer, vision = FakeQwen(), FakeTokenizer(), FakeVision()
    cache_split(
        split="test", loader=DataLoader([row], batch_size=1), dataset_length=1,
        output_dir=tmp_path, vision_encoder=vision, projector=SpectrumTextProjector(),
        qwen=qwen, tokenizer=tokenizer, config=MultimodalCacheConfig(),
        device=torch.device("cpu"), text_adapter=QwenTextBridge(4),
        input_modality=input_modality,
    )
    assert qwen.input_lengths == [expected_length]
    assert vision.called is needs_vision
    assert tokenizer.called is needs_tokenizer
    assert np.load(tmp_path / "test_mask.npy").sum(axis=1).tolist() == [expected_length]
