"""Tests for optional LoRA support in GPT2CrossAttnDecoder.

These tests intentionally do NOT download GPT-2 from Hugging Face.

Instead, ``GPT2Model.from_pretrained`` is monkeypatched to return a tiny
locally constructed GPT-2 model. This keeps the tests:

    - fast
    - deterministic
    - offline
    - suitable for normal pytest runs

The tests verify the three supported GPT-2 training modes:

1. Full fine-tuning
2. Frozen pretrained GPT-2
3. LoRA adaptation

For LoRA mode we additionally verify that:

    - pretrained GPT-2 self-attention weights are frozen
    - LoRA parameters are trainable
    - LoRA targets only causal self-attention
    - GPT-2 MLP layers remain frozen
    - newly initialized cross-attention remains fully trainable
    - PatchSGG task-specific layers remain trainable
    - forward/backward still works
"""
from __future__ import annotations

from typing import Iterable

import pytest
import torch

# ---------------------------------------------------------------------------
# Transformers is already required by the GPT-2 decoder, but using
# importorskip here makes the test suite degrade cleanly in environments
# where optional GPT-2 dependencies are not installed.
# ---------------------------------------------------------------------------
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2Model

from patchsgg.decoder.gpt2_decoder import GPT2CrossAttnDecoder
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import VG_VOCAB


# =============================================================================
# Tiny local GPT-2 checkpoint
# =============================================================================


@pytest.fixture(autouse=True)
def fake_gpt2_checkpoint(monkeypatch):
    """Replace Hugging Face checkpoint loading with a tiny local GPT-2.

    ``GPT2CrossAttnDecoder`` normally executes:

        GPT2Model.from_pretrained(...)

    which would download/load the actual GPT-2 checkpoint.

    For unit tests, we instead create a very small GPT-2:

        hidden size: 32
        layers:      2
        heads:       2
        context:     64

    This is large enough to exercise the real GPT-2/PEFT module hierarchy
    while remaining very cheap to construct and execute.
    """

    def _fake_from_pretrained(
        cls,
        *_args,
        **_kwargs,
    ):
        # Make the fake pretrained checkpoint deterministic.
        torch.manual_seed(1234)

        config = GPT2Config(
            vocab_size=128,

            # Tiny hidden dimension.
            n_embd=32,

            # Two GPT-2 transformer blocks.
            n_layer=2,

            # Two attention heads.
            n_head=2,

            # More than enough for these tests.
            n_positions=64,
            n_ctx=64,

            # Keep tests deterministic.
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,

            use_cache=True,
        )

        return GPT2Model(config)

    monkeypatch.setattr(
        GPT2Model,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )


# =============================================================================
# Helpers
# =============================================================================


def _build_decoder(
    *,
    freeze_pretrained: bool = False,
    lora_enabled: bool = False,
) -> GPT2CrossAttnDecoder:
    """Construct the tiny GPT-2 PatchSGG decoder used by the tests."""

    return GPT2CrossAttnDecoder(
        vocab=VG_VOCAB,

        # Conditioning vectors have dimension 16 in these tests.
        cond_dim=16,

        # Native fake GPT-2 context is 64, so no extension is necessary.
        max_seq_len=64,

        # This name is ignored because from_pretrained is monkeypatched.
        model_name="fake-local-gpt2",

        revision="main",
        cache_dir=None,
        local_files_only=True,

        # Disable checkpointing in tiny unit tests.
        gradient_checkpointing=False,

        freeze_pretrained=freeze_pretrained,

        # Disable tying here so token_embed and head can be tested
        # independently.
        tie_graph_embeddings=False,

        extend_positions=False,

        dropout=0.0,

        # --------------------------------------------------------------
        # LoRA
        # --------------------------------------------------------------
        lora_enabled=lora_enabled,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_bias="none",
    )


def _parameters_containing(
    module: torch.nn.Module,
    text: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return named parameters whose name contains ``text``."""

    return [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if text in name
    ]


def _assert_parameters_exist(
    parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ],
    description: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Convert to list and fail clearly when no matching parameters exist."""

    parameters = list(parameters)

    assert parameters, (
        f"Expected to find {description}, "
        "but no matching parameters were found."
    )

    return parameters


def _assert_all_trainable(
    parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ],
    description: str,
) -> None:
    """Assert every supplied parameter requires gradients."""

    parameters = _assert_parameters_exist(
        parameters,
        description,
    )

    not_trainable = [
        name
        for name, parameter in parameters
        if not parameter.requires_grad
    ]

    assert not not_trainable, (
        f"Expected all {description} to be trainable, "
        f"but these were frozen:\n"
        + "\n".join(not_trainable)
    )


def _assert_all_frozen(
    parameters: Iterable[
        tuple[str, torch.nn.Parameter]
    ],
    description: str,
) -> None:
    """Assert every supplied parameter is frozen."""

    parameters = _assert_parameters_exist(
        parameters,
        description,
    )

    still_trainable = [
        name
        for name, parameter in parameters
        if parameter.requires_grad
    ]

    assert not still_trainable, (
        f"Expected all {description} to be frozen, "
        f"but these were trainable:\n"
        + "\n".join(still_trainable)
    )


def _conditioning(
    batch_size: int = 2,
    num_tokens: int = 5,
    dim: int = 16,
) -> ConditioningSet:
    """Create fake image/text conditioning memory."""

    tokens = torch.randn(
        batch_size,
        num_tokens,
        dim,
    )

    pooled = torch.randn(
        batch_size,
        dim,
    )

    mask = torch.ones(
        batch_size,
        num_tokens,
        dtype=torch.bool,
    )

    return ConditioningSet(
        tokens=tokens,
        pooled=pooled,
        mask=mask,
    )


def _trainable_parameter_count(
    module: torch.nn.Module,
) -> int:
    """Count parameters with ``requires_grad=True``."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


# =============================================================================
# Mode 1: full GPT-2 fine-tuning
# =============================================================================


def test_gpt2_full_finetuning_mode():
    """Full mode should leave ordinary GPT-2 parameters trainable."""

    decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=False,
    )

    assert decoder.lora_enabled is False

    # ------------------------------------------------------------------
    # GPT-2 causal self-attention should be trainable.
    # ------------------------------------------------------------------
    self_attention = _parameters_containing(
        decoder,
        ".attn.c_attn",
    )

    _assert_all_trainable(
        self_attention,
        "GPT-2 self-attention c_attn parameters",
    )

    self_attention_projection = _parameters_containing(
        decoder,
        ".attn.c_proj",
    )

    _assert_all_trainable(
        self_attention_projection,
        "GPT-2 self-attention c_proj parameters",
    )

    # ------------------------------------------------------------------
    # GPT-2 MLP should also be trainable during full fine-tuning.
    # ------------------------------------------------------------------
    mlp = _parameters_containing(
        decoder,
        ".mlp.",
    )

    _assert_all_trainable(
        mlp,
        "GPT-2 MLP parameters",
    )

    # ------------------------------------------------------------------
    # Newly created cross-attention must be trainable.
    # ------------------------------------------------------------------
    cross_attention = _parameters_containing(
        decoder,
        ".crossattention.",
    )

    _assert_all_trainable(
        cross_attention,
        "GPT-2 cross-attention parameters",
    )

    cross_attention_norm = _parameters_containing(
        decoder,
        ".ln_cross_attn.",
    )

    _assert_all_trainable(
        cross_attention_norm,
        "GPT-2 cross-attention LayerNorm parameters",
    )

    # ------------------------------------------------------------------
    # PatchSGG-specific task layers must be trainable.
    # ------------------------------------------------------------------
    assert decoder.cond_proj.weight.requires_grad
    assert decoder.cond_proj.bias.requires_grad

    assert decoder.token_embed.weight.requires_grad
    assert decoder.head.weight.requires_grad

    # ------------------------------------------------------------------
    # No LoRA parameters should exist.
    # ------------------------------------------------------------------
    lora_parameters = _parameters_containing(
        decoder,
        "lora_",
    )

    assert not lora_parameters


# =============================================================================
# Mode 2: frozen pretrained GPT-2
# =============================================================================


def test_gpt2_frozen_mode():
    """Frozen mode should freeze GPT-2 but train new cross-attention."""

    decoder = _build_decoder(
        freeze_pretrained=True,
        lora_enabled=False,
    )

    assert decoder.lora_enabled is False

    # ------------------------------------------------------------------
    # Pretrained causal self-attention must be frozen.
    # ------------------------------------------------------------------
    self_attention = _parameters_containing(
        decoder,
        ".attn.c_attn",
    )

    _assert_all_frozen(
        self_attention,
        "pretrained GPT-2 self-attention parameters",
    )

    self_attention_projection = _parameters_containing(
        decoder,
        ".attn.c_proj",
    )

    _assert_all_frozen(
        self_attention_projection,
        "pretrained GPT-2 self-attention output projections",
    )

    # ------------------------------------------------------------------
    # Pretrained MLP must be frozen.
    # ------------------------------------------------------------------
    mlp = _parameters_containing(
        decoder,
        ".mlp.",
    )

    _assert_all_frozen(
        mlp,
        "pretrained GPT-2 MLP parameters",
    )

    # ------------------------------------------------------------------
    # Pretrained positional embeddings must be frozen.
    # ------------------------------------------------------------------
    positional = _parameters_containing(
        decoder,
        ".wpe.",
    )

    _assert_all_frozen(
        positional,
        "GPT-2 positional embeddings",
    )

    # ------------------------------------------------------------------
    # Newly initialized cross-attention MUST remain trainable.
    # ------------------------------------------------------------------
    cross_attention = _parameters_containing(
        decoder,
        ".crossattention.",
    )

    _assert_all_trainable(
        cross_attention,
        "new GPT-2 cross-attention parameters",
    )

    cross_attention_norm = _parameters_containing(
        decoder,
        ".ln_cross_attn.",
    )

    _assert_all_trainable(
        cross_attention_norm,
        "new cross-attention LayerNorm parameters",
    )

    # ------------------------------------------------------------------
    # PatchSGG task-specific layers remain trainable.
    # ------------------------------------------------------------------
    assert decoder.cond_proj.weight.requires_grad
    assert decoder.cond_proj.bias.requires_grad

    assert decoder.token_embed.weight.requires_grad
    assert decoder.head.weight.requires_grad

    # No LoRA should exist in frozen-only mode.
    assert not _parameters_containing(
        decoder,
        "lora_",
    )


# =============================================================================
# Mode 3: LoRA
# =============================================================================


def test_gpt2_lora_targets_only_self_attention():
    """LoRA should target only GPT-2 causal self-attention."""

    # Skip only the LoRA-specific test if PEFT was not installed.
    pytest.importorskip("peft")

    decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=True,
    )

    assert decoder.lora_enabled is True

    # ------------------------------------------------------------------
    # Our tiny GPT-2 has two layers.
    #
    # Each layer contributes:
    #
    #   attn.c_attn
    #   attn.c_proj
    #
    # therefore exactly four modules should be targeted.
    # ------------------------------------------------------------------
    expected_targets = {
        "h.0.attn.c_attn",
        "h.0.attn.c_proj",
        "h.1.attn.c_attn",
        "h.1.attn.c_proj",
    }

    assert set(
        decoder.lora_target_modules
    ) == expected_targets

    # ------------------------------------------------------------------
    # LoRA parameters must actually exist.
    # ------------------------------------------------------------------
    lora_parameters = _parameters_containing(
        decoder,
        "lora_",
    )

    lora_parameters = _assert_parameters_exist(
        lora_parameters,
        "LoRA parameters",
    )

    # Every LoRA parameter must be trainable.
    _assert_all_trainable(
        lora_parameters,
        "LoRA parameters",
    )

    # ------------------------------------------------------------------
    # Most important protection:
    #
    # LoRA must NOT have been attached to:
    #
    #   cross-attention
    #   MLP
    #
    # ------------------------------------------------------------------
    for name, _parameter in lora_parameters:

        assert ".crossattention." not in name, (
            "LoRA was accidentally attached to newly initialized "
            f"cross-attention: {name}"
        )

        assert ".mlp." not in name, (
            "LoRA was accidentally attached to GPT-2 MLP: "
            f"{name}"
        )

        assert ".attn." in name, (
            "LoRA parameter was found outside causal self-attention: "
            f"{name}"
        )


def test_gpt2_lora_freezes_base_and_keeps_cross_attention_trainable():
    """LoRA should freeze base GPT-2 while keeping new task layers trainable."""

    pytest.importorskip("peft")

    decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=True,
    )

    # ------------------------------------------------------------------
    # Base causal self-attention parameters must be frozen.
    #
    # PEFT changes names to forms similar to:
    #
    #   gpt2.base_model.model.h.0.attn.c_attn.base_layer.weight
    #
    # so we deliberately use substring matching rather than exact paths.
    # ------------------------------------------------------------------
    c_attn_parameters = [
        (name, parameter)
        for name, parameter in decoder.named_parameters()
        if ".attn.c_attn." in name
        and "lora_" not in name
    ]

    _assert_all_frozen(
        c_attn_parameters,
        "base GPT-2 c_attn parameters",
    )

    c_proj_parameters = [
        (name, parameter)
        for name, parameter in decoder.named_parameters()
        if ".attn.c_proj." in name
        and "lora_" not in name
    ]

    _assert_all_frozen(
        c_proj_parameters,
        "base GPT-2 c_proj parameters",
    )

    # ------------------------------------------------------------------
    # GPT-2 MLP remains frozen in our first LoRA experiment.
    # ------------------------------------------------------------------
    mlp_parameters = _parameters_containing(
        decoder,
        ".mlp.",
    )

    _assert_all_frozen(
        mlp_parameters,
        "GPT-2 MLP parameters in LoRA mode",
    )

    # ------------------------------------------------------------------
    # Normal pretrained GPT-2 LayerNorm remains frozen.
    # ------------------------------------------------------------------
    ln1_parameters = _parameters_containing(
        decoder,
        ".ln_1.",
    )

    _assert_all_frozen(
        ln1_parameters,
        "pretrained GPT-2 LayerNorm parameters",
    )

    # ------------------------------------------------------------------
    # Newly initialized cross-attention MUST be trainable.
    # ------------------------------------------------------------------
    cross_attention = _parameters_containing(
        decoder,
        ".crossattention.",
    )

    _assert_all_trainable(
        cross_attention,
        "new GPT-2 cross-attention parameters",
    )

    cross_attention_norm = _parameters_containing(
        decoder,
        ".ln_cross_attn.",
    )

    _assert_all_trainable(
        cross_attention_norm,
        "new GPT-2 cross-attention LayerNorm parameters",
    )

    # ------------------------------------------------------------------
    # PatchSGG layers are outside the PEFT-wrapped GPT-2 and must stay
    # fully trainable.
    # ------------------------------------------------------------------
    assert decoder.cond_proj.weight.requires_grad
    assert decoder.cond_proj.bias.requires_grad

    assert decoder.token_embed.weight.requires_grad
    assert decoder.head.weight.requires_grad

    # ------------------------------------------------------------------
    # LoRA itself must be trainable.
    # ------------------------------------------------------------------
    lora_parameters = _parameters_containing(
        decoder,
        "lora_",
    )

    _assert_all_trainable(
        lora_parameters,
        "LoRA adapter parameters",
    )


# =============================================================================
# Parameter-efficiency sanity check
# =============================================================================


def test_lora_has_fewer_trainable_parameters_than_full_finetuning():
    """LoRA mode should train substantially fewer parameters than full FT."""

    pytest.importorskip("peft")

    full_decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=False,
    )

    lora_decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=True,
    )

    full_trainable = _trainable_parameter_count(
        full_decoder
    )

    lora_trainable = _trainable_parameter_count(
        lora_decoder
    )

    assert lora_trainable < full_trainable, (
        "LoRA should have fewer trainable parameters than "
        "full GPT-2 fine-tuning, but got:\n"
        f"full fine-tuning: {full_trainable:,}\n"
        f"LoRA:             {lora_trainable:,}"
    )


# =============================================================================
# Forward/backward test
# =============================================================================


def test_gpt2_lora_forward_backward():
    """A LoRA decoder should support normal PatchSGG forward/backward."""

    pytest.importorskip("peft")

    torch.manual_seed(42)

    decoder = _build_decoder(
        freeze_pretrained=False,
        lora_enabled=True,
    )

    decoder.train()

    batch_size = 2
    sequence_length = 7

    cond = _conditioning(
        batch_size=batch_size,
        num_tokens=5,
        dim=16,
    )

    tokens = torch.randint(
        low=0,
        high=VG_VOCAB.vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    logits = decoder(
        cond,
        tokens,
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        VG_VOCAB.vocab_size,
    )

    assert torch.isfinite(
        logits
    ).all()

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    #
    # A simple synthetic loss is enough to verify gradient flow.
    # ------------------------------------------------------------------
    loss = logits.square().mean()

    loss.backward()

    # ------------------------------------------------------------------
    # LoRA must receive gradients.
    # ------------------------------------------------------------------
    lora_parameters = [
        parameter
        for name, parameter in decoder.named_parameters()
        if "lora_" in name
        and parameter.requires_grad
    ]

    assert lora_parameters

    assert any(
        parameter.grad is not None
        for parameter in lora_parameters
    ), "No LoRA parameter received a gradient."

    # ------------------------------------------------------------------
    # New cross-attention must receive gradients.
    # ------------------------------------------------------------------
    cross_attention_parameters = [
        parameter
        for name, parameter in decoder.named_parameters()
        if ".crossattention." in name
        and parameter.requires_grad
    ]

    assert cross_attention_parameters

    assert any(
        parameter.grad is not None
        for parameter in cross_attention_parameters
    ), "Cross-attention did not receive gradients."

    # ------------------------------------------------------------------
    # Conditioning projection must receive gradients.
    # ------------------------------------------------------------------
    assert decoder.cond_proj.weight.grad is not None

    # ------------------------------------------------------------------
    # Graph vocabulary layers must receive gradients.
    # ------------------------------------------------------------------
    assert decoder.token_embed.weight.grad is not None
    assert decoder.head.weight.grad is not None

    # ------------------------------------------------------------------
    # Frozen GPT-2 base self-attention must NOT accumulate gradients.
    # ------------------------------------------------------------------
    base_self_attention = [
        parameter
        for name, parameter in decoder.named_parameters()
        if ".attn.c_attn." in name
        and "lora_" not in name
    ]

    assert base_self_attention

    assert all(
        parameter.grad is None
        for parameter in base_self_attention
    )


# =============================================================================
# Configuration validation
# =============================================================================


@pytest.mark.parametrize(
    (
        "kwargs",
        "expected_message",
    ),
    [
        (
            {
                "lora_r": 0,
            },
            "decoder.lora.r",
        ),
        (
            {
                "lora_alpha": 0,
            },
            "decoder.lora.alpha",
        ),
        (
            {
                "lora_dropout": -0.1,
            },
            "decoder.lora.dropout",
        ),
        (
            {
                "lora_dropout": 1.0,
            },
            "decoder.lora.dropout",
        ),
        (
            {
                "lora_bias": "invalid",
            },
            "decoder.lora.bias",
        ),
    ],
)
def test_invalid_lora_configuration_fails_cleanly(
    kwargs,
    expected_message,
):
    """Bad LoRA configuration should produce a useful error."""

    pytest.importorskip("peft")

    arguments = {
        "vocab": VG_VOCAB,
        "cond_dim": 16,
        "max_seq_len": 64,

        "model_name": "fake-local-gpt2",
        "local_files_only": True,

        "gradient_checkpointing": False,

        "freeze_pretrained": False,

        "tie_graph_embeddings": False,

        "extend_positions": False,

        "dropout": 0.0,

        "lora_enabled": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_bias": "none",
    }

    arguments.update(
        kwargs
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        GPT2CrossAttnDecoder(
            **arguments
        )