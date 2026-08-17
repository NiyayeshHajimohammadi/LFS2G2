"""
Benchmark:
    1) Normal PyTorch CrossEntropyLoss
    2) PatchSGG custom loss

The custom loss is created exactly like training:
    PatchSGGModel:
        self.vocab = build_vocab(cfg)
        self.loss_fn = build_loss(cfg, self.vocab)

"""

import time
import torch

from patchsgg.config import load_config
from patchsgg.model import build_vocab
from patchsgg.losses import build_loss
from patchsgg.graph_seq.vocab import TOKENS_PER_REL


# ============================================================
# Settings
# ============================================================

CONFIG_PATH = "patchsgg/configs/location_free_paper.yaml"

DEVICE = "cuda"

BATCH_SIZE = 4
MAX_RELS = 150

WARMUP = 20
ITERATIONS = 200


# ============================================================
# Load configuration
# ============================================================

cfg = load_config(CONFIG_PATH)


# ============================================================
# Create vocabulary exactly like PatchSGGModel
# ============================================================

vocab = build_vocab(cfg)


print("--------------------------------")
print("Config:", CONFIG_PATH)
print("Vocabulary size:", vocab.vocab_size)
print("TOKENS_PER_REL:", TOKENS_PER_REL)
print("--------------------------------")


# ============================================================
# Create fake decoder output
#
# Real decoder output:
#
# logits:
#       [batch, sequence_length, vocabulary_size]
#
# target:
#       [batch, sequence_length]
#
# ============================================================


sequence_length = 1 + MAX_RELS * TOKENS_PER_REL


logits = torch.randn(
    BATCH_SIZE,
    sequence_length,
    vocab.vocab_size,
    device=DEVICE,
    requires_grad=True
)


targets = torch.randint(
    0,
    vocab.vocab_size,
    (BATCH_SIZE, sequence_length),
    device=DEVICE
)


print("logits:", logits.shape)
print("targets:", targets.shape)



# ============================================================
# Benchmark helper function
# ============================================================

def benchmark(loss_function):

    # --------------------------
    # Warmup
    # --------------------------
    for _ in range(WARMUP):

        loss = loss_function()

        loss.backward()

        logits.grad = None


    torch.cuda.synchronize()


    # --------------------------
    # Timing
    # --------------------------

    start = time.time()


    for _ in range(ITERATIONS):

        loss = loss_function()

        loss.backward()

        logits.grad = None


    torch.cuda.synchronize()


    end = time.time()


    return (end-start)/ITERATIONS



# ============================================================
# Normal PyTorch CE
# ============================================================

normal_ce = torch.nn.CrossEntropyLoss(
    ignore_index=vocab.no_known_token
)


def run_normal_ce():

    return normal_ce(
        logits.reshape(-1, vocab.vocab_size),
        targets.reshape(-1)
    )



# ============================================================
# PatchSGG loss
# ============================================================

patchsgg_loss = build_loss(
    cfg,
    vocab
).to(DEVICE)


print()
print("Loss selected from config:")
print(cfg.loss.type)



def run_patchsgg_loss():

    return patchsgg_loss(
        logits,
        targets
    )



# ============================================================
# Run benchmark
# ============================================================


ce_time = benchmark(
    run_normal_ce
)


patchsgg_time = benchmark(
    run_patchsgg_loss
)



# ============================================================
# Results
# ============================================================

print()
print("==============================")
print("RESULTS")
print("==============================")

print(
    f"Normal Cross Entropy : {ce_time:.8f} sec/iteration"
)

print(
    f"PatchSGG Loss        : {patchsgg_time:.8f} sec/iteration"
)

print()

print(
    f"PatchSGG / CE speed ratio: "
    f"{patchsgg_time / ce_time:.2f}x"
)