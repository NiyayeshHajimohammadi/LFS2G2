"""Hugging Face GPT-2 graph decoder with visual/text cross-attention.

The pretrained GPT-2 checkpoint supplies the causal self-attention blocks, MLPs,
layer norms, and the original positional embeddings. PatchSGG keeps its own small
scene-graph vocabulary, so graph-token embeddings and the graph output head are
learned for this task rather than reusing GPT-2's English BPE vocabulary.

GPT-2 checkpoints do not contain cross-attention layers. Those layers are added
from the GPT-2 configuration and initialized randomly; all compatible pretrained
weights are copied from the downloaded checkpoint.

Optional LoRA support adapts only the pretrained GPT-2 causal self-attention
projections. The newly initialized cross-attention layers remain fully trainable.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.decoder.sampling import GenConfig, constrained_generate
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab


def _optional(value):
    """Normalize OmegaConf/AttrDict null-like values for Hugging Face kwargs."""
    return None if value in (None, "", "null", "None") else value


class GPT2CrossAttnDecoder(GraphDecoder):
    """A pretrained GPT-2 causal decoder conditioned through cross-attention.

    Notes
    -----
    * ``inputs_embeds`` is used, because PatchSGG graph token IDs are not GPT-2
      BPE token IDs.
    * The downloaded GPT-2 lexical embedding table is discarded after loading;
      it is never used by this model.
    * Cross-attention layers, the conditioning projection, graph embeddings,
      and graph output head are task-specific and start from random weights.
    * When the requested graph sequence is longer than GPT-2's native context,
      pretrained positional embeddings are copied and only the extra positions
      are randomly initialized.
    * LoRA is optional. When disabled, this decoder behaves exactly as the
      original implementation.
    * When LoRA is enabled, the copied pretrained GPT-2 backbone is frozen,
      LoRA adapters are attached to causal self-attention, and the newly added
      cross-attention remains fully trainable.
    """

    def __init__(
        self,
        vocab: GraphVocab,
        cond_dim: int,
        max_seq_len: int,
        model_name: str = "openai-community/gpt2",
        revision: str = "main",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        gradient_checkpointing: bool = True,
        freeze_pretrained: bool = False,
        tie_graph_embeddings: bool = True,
        extend_positions: bool = True,
        dropout: Optional[float] = None,

        # ------------------------------------------------------------------
        # Optional LoRA configuration
        # ------------------------------------------------------------------
        lora_enabled: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
    ):
        try:
            from transformers import GPT2Model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "decoder.type='gpt2_cross_attn' requires Hugging Face "
                "Transformers. Install it with "
                "`pip install transformers huggingface_hub`."
            ) from exc

        # ------------------------------------------------------------------
        # Hugging Face checkpoint-loading options
        # ------------------------------------------------------------------
        load_kwargs = {
            "revision": revision,
            "local_files_only": bool(local_files_only),
        }

        cache_dir = _optional(cache_dir)

        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        # ------------------------------------------------------------------
        # Load ordinary pretrained GPT-2
        # ------------------------------------------------------------------
        #
        # The standard GPT-2 checkpoint has:
        #
        #   causal self-attention
        #   MLPs
        #   layer norms
        #   positional embeddings
        #
        # but it does NOT contain cross-attention.
        #
        # We therefore:
        #
        #   1. load ordinary pretrained GPT-2;
        #   2. copy its configuration;
        #   3. enable cross-attention;
        #   4. construct a new GPT2Model;
        #   5. copy every compatible pretrained parameter.
        #
        # ------------------------------------------------------------------
        pretrained = GPT2Model.from_pretrained(
            model_name,
            **load_kwargs,
        )

        hf_cfg = copy.deepcopy(pretrained.config)

        native_positions = int(
            getattr(
                hf_cfg,
                "n_positions",
                1024,
            )
        )

        # ------------------------------------------------------------------
        # Context-length handling
        # ------------------------------------------------------------------
        if max_seq_len > native_positions and not extend_positions:
            del pretrained

            raise ValueError(
                f"PatchSGG needs {max_seq_len} GPT-2 positions, "
                f"but {model_name!r} provides {native_positions}. "
                "Set decoder.extend_positions=true or reduce eval.max_rels."
            )

        target_positions = max(
            native_positions,
            int(max_seq_len),
        )

        if target_positions > native_positions:
            import warnings

            warnings.warn(
                f"Extending GPT-2 positions from {native_positions} "
                f"to {target_positions}. Positions above "
                f"{native_positions - 1} are randomly initialized.",
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Convert GPT-2 into a decoder with cross-attention
        # ------------------------------------------------------------------
        hf_cfg.add_cross_attention = True
        hf_cfg.is_decoder = True

        hf_cfg.n_positions = target_positions
        hf_cfg.n_ctx = target_positions

        hf_cfg.use_cache = True

        # Optionally override GPT-2 dropout values.
        if dropout is not None:
            p = float(dropout)

            hf_cfg.resid_pdrop = p
            hf_cfg.embd_pdrop = p
            hf_cfg.attn_pdrop = p

        hidden_size = int(hf_cfg.n_embd)

        # ------------------------------------------------------------------
        # Initialize GraphDecoder task-specific components
        # ------------------------------------------------------------------
        super().__init__(
            vocab=vocab,
            cond_dim=cond_dim,
            d_model=hidden_size,
            max_seq_len=target_positions,
        )

        self.model_name = model_name

        self.native_max_positions = native_positions
        self.max_seq_len = target_positions

        # ------------------------------------------------------------------
        # Build cross-attention-enabled GPT-2
        # ------------------------------------------------------------------
        self.gpt2 = GPT2Model(hf_cfg)

        # Copy compatible pretrained parameters into the new model.
        self._copy_pretrained_weights(pretrained)

        del pretrained

        # ------------------------------------------------------------------
        # Remove unused English GPT-2 token embedding table
        # ------------------------------------------------------------------
        #
        # PatchSGG NEVER passes GPT-2 BPE token IDs.
        #
        # Instead:
        #
        #   scene-graph token IDs
        #          ↓
        #   self.token_embed
        #          ↓
        #   GPT-2 inputs_embeds
        #
        # Therefore GPT-2's original ~50k-word lexical embedding table would
        # only waste memory.
        #
        # GPT2Model still expects `wte` to exist, so we keep a one-row
        # placeholder.
        # ------------------------------------------------------------------
        self.gpt2.wte = nn.Embedding(
            1,
            hidden_size,
        )

        self.gpt2.wte.weight.requires_grad_(False)

        # ------------------------------------------------------------------
        # Remove GraphDecoder components GPT-2 does not need
        # ------------------------------------------------------------------
        #
        # GPT-2 already applies:
        #
        #   its own positional embeddings (wpe)
        #   its own final layer norm
        #
        # so GraphDecoder's versions are unnecessary.
        # ------------------------------------------------------------------
        self.norm = nn.Identity()

        del self.pos_embed

        # ------------------------------------------------------------------
        # Scene-graph vocabulary prediction head
        # ------------------------------------------------------------------
        self.head = nn.Linear(
            hidden_size,
            vocab.vocab_size,
            bias=False,
        )

        self._initialize_task_layers(
            float(hf_cfg.initializer_range)
        )

        if tie_graph_embeddings:
            self.head.weight = self.token_embed.weight

        # ------------------------------------------------------------------
        # LoRA state
        # ------------------------------------------------------------------
        self.lora_enabled = bool(lora_enabled)

        # Useful for debugging/tests so we can inspect exactly what LoRA
        # targeted.
        self.lora_target_modules: tuple[str, ...] = ()

        # ------------------------------------------------------------------
        # Gradient checkpointing
        # ------------------------------------------------------------------
        #
        # Enable this while self.gpt2 is still the ordinary GPT2Model.
        # PEFT wrapping later preserves these underlying GPT-2 modules.
        # ------------------------------------------------------------------
        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable()

        # ------------------------------------------------------------------
        # Select GPT-2 training strategy
        # ------------------------------------------------------------------
        #
        # There are three possible modes:
        #
        # 1. Full fine-tuning
        #
        #       lora_enabled=False
        #       freeze_pretrained=False
        #
        # 2. Frozen pretrained GPT-2
        #
        #       lora_enabled=False
        #       freeze_pretrained=True
        #
        # 3. LoRA
        #
        #       lora_enabled=True
        #
        # LoRA takes precedence over freeze_pretrained because PEFT itself
        # handles freezing the pretrained GPT-2 parameters.
        # ------------------------------------------------------------------
        if self.lora_enabled:
            self._apply_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                bias=lora_bias,
            )

        elif freeze_pretrained:
            self._freeze_pretrained_submodules()

    # =========================================================================
    # Pretrained GPT-2 weight loading
    # =========================================================================

    def _copy_pretrained_weights(
        self,
        source: nn.Module,
    ) -> None:
        """Copy all compatible GPT-2 weights.

        Most tensors have identical shapes and are copied directly.

        The only special case is the positional embedding table ``wpe`` when
        PatchSGG requests a context longer than standard GPT-2. In that case,
        the pretrained prefix is copied while the additional positions keep
        their random initialization.
        """
        source_state = source.state_dict()
        target_state = self.gpt2.state_dict()

        with torch.no_grad():

            for name, src in source_state.items():

                dst = target_state.get(name)

                if dst is None:
                    continue

                # ----------------------------------------------------------
                # Direct shape-compatible copy
                # ----------------------------------------------------------
                if dst.shape == src.shape:
                    dst.copy_(src)
                    continue

                # ----------------------------------------------------------
                # Extended positional embeddings
                # ----------------------------------------------------------
                if (
                    name == "wpe.weight"
                    and dst.ndim == 2
                    and src.ndim == 2
                    and dst.shape[1] == src.shape[1]
                    and dst.shape[0] >= src.shape[0]
                ):
                    dst[: src.shape[0]].copy_(src)

    # =========================================================================
    # Task-specific initialization
    # =========================================================================

    def _initialize_task_layers(
        self,
        std: float,
    ) -> None:
        """Initialize PatchSGG-specific layers using GPT-2's init scale."""

        # Graph-token embeddings.
        nn.init.normal_(
            self.token_embed.weight,
            mean=0.0,
            std=std,
        )

        # Encoder-conditioning projection.
        nn.init.normal_(
            self.cond_proj.weight,
            mean=0.0,
            std=std,
        )

        nn.init.zeros_(
            self.cond_proj.bias
        )

        # Graph-vocabulary output head.
        nn.init.normal_(
            self.head.weight,
            mean=0.0,
            std=std,
        )

    # =========================================================================
    # GPT-2 freezing / trainability
    # =========================================================================

    def _enable_cross_attention_training(self) -> None:
        """Keep newly initialized GPT-2 cross-attention fully trainable.

        This intentionally uses parameter names instead of ``self.gpt2.h``.

        Before LoRA:

            h.0.crossattention...
            h.0.ln_cross_attn...

        After PEFT wrapping, parameter names gain additional prefixes such as:

            base_model.model.h.0.crossattention...

        Name matching therefore works in both cases.
        """
        for name, parameter in self.gpt2.named_parameters():

            if ".crossattention." in name:
                parameter.requires_grad_(True)

            elif ".ln_cross_attn." in name:
                parameter.requires_grad_(True)

    def _freeze_pretrained_submodules(self) -> None:
        """Freeze GPT-2 while keeping newly added cross-attention trainable.

        This is the legacy frozen-GPT-2 mode and is used when:

            freeze_pretrained=True
            lora_enabled=False
        """

        # Freeze everything inside GPT-2.
        for parameter in self.gpt2.parameters():
            parameter.requires_grad_(False)

        # Cross-attention was not pretrained and must still learn.
        self._enable_cross_attention_training()

    # =========================================================================
    # LoRA
    # =========================================================================

    def _self_attention_lora_targets(self) -> list[str]:
        """Find exact pretrained GPT-2 causal self-attention LoRA targets.

        GPT-2 contains similarly named projections in several places:

            h.0.attn.c_attn
            h.0.attn.c_proj

            h.0.crossattention.c_attn
            h.0.crossattention.c_proj

            h.0.mlp.c_proj

        Using generic target names such as:

            ["c_attn", "c_proj"]

        could therefore target components that we do not want to adapt.

        For the initial PatchSGG LoRA experiment we target ONLY:

            h.*.attn.c_attn
            h.*.attn.c_proj

        Cross-attention remains fully trainable instead of LoRA-adapted, and
        the pretrained MLP stays frozen.
        """
        targets: list[str] = []

        for name, _module in self.gpt2.named_modules():

            if name.endswith(".attn.c_attn"):
                targets.append(name)

            elif name.endswith(".attn.c_proj"):
                targets.append(name)

        if not targets:
            raise RuntimeError(
                "LoRA is enabled, but no GPT-2 self-attention projection "
                "modules were found. Expected modules such as "
                "'h.0.attn.c_attn' and 'h.0.attn.c_proj'."
            )

        return sorted(targets)

    @staticmethod
    def _validate_lora_config(
        r: int,
        alpha: int,
        dropout: float,
        bias: str,
    ) -> tuple[int, int, float, str]:
        """Validate and normalize LoRA hyperparameters."""

        r = int(r)
        alpha = int(alpha)
        dropout = float(dropout)
        bias = str(bias)

        if r <= 0:
            raise ValueError(
                f"decoder.lora.r must be > 0, got {r}"
            )

        if alpha <= 0:
            raise ValueError(
                f"decoder.lora.alpha must be > 0, got {alpha}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "decoder.lora.dropout must satisfy "
                f"0 <= dropout < 1, got {dropout}"
            )

        if bias not in {
            "none",
            "all",
            "lora_only",
        }:
            raise ValueError(
                "decoder.lora.bias must be one of "
                "'none', 'all', or 'lora_only', "
                f"got {bias!r}"
            )

        return (
            r,
            alpha,
            dropout,
            bias,
        )

    def _apply_lora(
        self,
        *,
        r: int,
        alpha: int,
        dropout: float,
        bias: str,
    ) -> None:
        """Attach LoRA adapters to pretrained GPT-2 self-attention.

        PEFT freezes the wrapped GPT-2 base parameters and adds trainable
        low-rank adapter parameters.

        After PEFT wrapping, PatchSGG explicitly re-enables the newly
        initialized GPT-2 cross-attention because those layers did not come
        from the pretrained checkpoint and must be trained normally.

        The following PatchSGG layers live outside ``self.gpt2`` and therefore
        remain trainable automatically:

            self.cond_proj
            self.token_embed
            self.head
        """

        # --------------------------------------------------------------
        # PEFT remains an optional dependency
        # --------------------------------------------------------------
        #
        # If LoRA is disabled, this import is never executed.
        # --------------------------------------------------------------
        try:
            from peft import (
                LoraConfig,
                get_peft_model,
            )

        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "decoder.lora.enabled=true requires PEFT. "
                "Install the LoRA extra with "
                "`pip install -e '.[lora]'` "
                "or install PEFT directly with `pip install peft`."
            ) from exc

        # --------------------------------------------------------------
        # Validate configuration
        # --------------------------------------------------------------
        (
            r,
            alpha,
            dropout,
            bias,
        ) = self._validate_lora_config(
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=bias,
        )

        # --------------------------------------------------------------
        # Find only causal self-attention targets
        # --------------------------------------------------------------
        targets = self._self_attention_lora_targets()

        self.lora_target_modules = tuple(targets)

        # --------------------------------------------------------------
        # Construct PEFT LoRA configuration
        # --------------------------------------------------------------
        #
        # GPT-2 uses Hugging Face Conv1D projection layers. Their weight
        # orientation requires fan_in_fan_out=True for LoRA.
        #
        # We intentionally do NOT specify task_type="CAUSAL_LM":
        #
        #   self.gpt2 is GPT2Model
        #
        # not:
        #
        #   GPT2LMHeadModel
        #
        # PatchSGG owns its graph-vocabulary LM head outside the Hugging Face
        # model, so a generic PeftModel wrapper is the appropriate structure.
        # --------------------------------------------------------------
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias=bias,
            fan_in_fan_out=True,
            init_lora_weights=True,
        )

        # --------------------------------------------------------------
        # Wrap GPT-2 with PEFT
        # --------------------------------------------------------------
        #
        # After this:
        #
        #   pretrained GPT-2 weights      frozen
        #   LoRA matrices                 trainable
        #
        # --------------------------------------------------------------
        self.gpt2 = get_peft_model(
            self.gpt2,
            lora_config,
        )

        # --------------------------------------------------------------
        # Restore training for newly initialized cross-attention
        # --------------------------------------------------------------
        #
        # get_peft_model freezes the underlying base model. That includes
        # PatchSGG's randomly initialized cross-attention, so explicitly
        # turn those parameters back on.
        # --------------------------------------------------------------
        self._enable_cross_attention_training()

    # =========================================================================
    # Conditioning
    # =========================================================================

    def _memory(
        self,
        cond: ConditioningSet,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Project encoder conditioning into GPT-2 hidden space.

        Parameters
        ----------
        cond.tokens:
            [B, N, cond_dim]

        cond.mask:
            optional [B, N]

        Returns
        -------
        memory:
            [B, N, d_model]

        memory_mask:
            optional [B, N]
        """

        memory = self.cond_proj(
            cond.tokens
        )

        # Hugging Face expects:
        #
        #   1 / True  -> valid encoder position
        #   0 / False -> padding
        #
        memory_mask = (
            None
            if cond.mask is None
            else cond.mask.to(
                device=memory.device
            )
        )

        return (
            memory,
            memory_mask,
        )

    # =========================================================================
    # Training forward pass
    # =========================================================================

    def _hidden(
        self,
        cond: ConditioningSet,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return GPT-2 hidden states for teacher-forced graph tokens.

        Parameters
        ----------
        cond:
            Encoder conditioning.

        tokens:
            Graph decoder input tokens with shape:

                [B, T]

        Returns
        -------
        torch.Tensor
            Hidden states:

                [B, T, d_model]
        """

        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"token sequence length {tokens.shape[1]} "
                f"exceeds GPT-2 context {self.max_seq_len}"
            )

        memory, memory_mask = self._memory(
            cond
        )

        outputs = self.gpt2(
            # PatchSGG graph vocabulary embeddings.
            inputs_embeds=self.token_embed(tokens),

            # Image/text encoder conditioning.
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,

            # Full teacher-forced training does not use generation cache.
            use_cache=False,

            return_dict=True,
        )

        return outputs.last_hidden_state

    # =========================================================================
    # Autoregressive generation
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        cond: ConditioningSet,
        gen_cfg: GenConfig,
    ):
        """Generate graph tokens with GPT-2 key/value caching.

        Generation uses the same LF-SGG structured constraints implemented by
        ``constrained_generate``.

        GPT-2's cache means that after the first decoding step, only the newly
        generated token needs to be passed through GPT-2.
        """

        total_steps = (
            int(gen_cfg.max_rels)
            * TOKENS_PER_REL
        )

        # One START token plus the generated relation sequence.
        required_positions = (
            1
            + total_steps
        )

        if required_positions > self.max_seq_len:
            raise ValueError(
                f"generation needs {required_positions} positions "
                f"but GPT-2 has {self.max_seq_len}"
            )

        # ------------------------------------------------------------------
        # Initial START token
        # ------------------------------------------------------------------
        start_tokens = torch.full(
            (
                cond.batch_size,
                1,
            ),
            self.vocab.start_token,
            dtype=torch.long,
            device=cond.tokens.device,
        )

        # Conditioning memory is constant throughout generation, so compute
        # it once.
        memory, memory_mask = self._memory(
            cond
        )

        # ------------------------------------------------------------------
        # Stateful generation step
        # ------------------------------------------------------------------
        def step_fn(
            sequence: torch.Tensor,
            past_key_values,
        ):
            # First iteration:
            #
            #   sequence = [START]
            #
            # Later iterations:
            #
            # previous tokens are already stored in GPT-2's KV cache, so only
            # the newest graph token needs to be processed.
            current_token = (
                sequence
                if past_key_values is None
                else sequence[:, -1:]
            )

            outputs = self.gpt2(
                inputs_embeds=self.token_embed(
                    current_token
                ),

                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,

                past_key_values=past_key_values,

                use_cache=True,
                return_dict=True,
            )

            # Predict the next scene-graph vocabulary token.
            logits = self.head(
                outputs.last_hidden_state[:, -1]
            )

            return (
                logits,
                outputs.past_key_values,
            )

        # ------------------------------------------------------------------
        # Structured graph generation
        # ------------------------------------------------------------------
        return constrained_generate(
            step_fn=step_fn,
            start_tokens=start_tokens,
            vocab=self.vocab,
            cfg=gen_cfg,
            initial_state=None,
            stateful=True,
        )