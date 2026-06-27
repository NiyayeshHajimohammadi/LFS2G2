"""End-to-end smoke + overfit sanity on the toy dataset (no external weights, CPU)."""
import torch
from torch.utils.data import DataLoader

from patchsgg.config import load_config
from patchsgg.data.collate import GraphCollator
from patchsgg.data.factory import build_dataset
from patchsgg.eval.evaluate import evaluate_graphs
from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.model import PatchSGGModel, build_vocab


def _tiny_cfg():
    return load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        [
            "vocab.max_num_rels=6",
            "data.toy_n_train=24",
            "data.toy_n_val=8",
            "data.toy_max_rels=2",
            "train.batch_size=8",
            "eval.max_rels=6",
        ],
    )


def test_train_step_and_predict_pipeline():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.device = "cpu"
    vocab = build_vocab(cfg)
    collate = GraphCollator(vocab=vocab, seed=0)
    ds = build_dataset(cfg, "train", vocab)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)
    model = PatchSGGModel(cfg)

    # encoders are frozen; only the decoder trains
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)

    batch = next(iter(loader))
    initial = model.compute_loss(batch, modality="text").item()
    model.train()
    for _ in range(60):
        for b in loader:
            loss = model.compute_loss(b, modality="text")
            opt.zero_grad()
            loss.backward()
            opt.step()
    final = model.compute_loss(batch, modality="text").item()
    assert final < initial, (initial, final)

    # prediction pipeline runs for both modalities and yields matcher tuples
    model.eval()
    preds_text = model.predict(batch, modality="text")
    preds_image = model.predict(batch, modality="image")
    assert len(preds_text) == len(batch["gt_graphs"])
    assert len(preds_image) == len(batch["gt_graphs"])
    for g in preds_text:
        for t in g:
            assert len(t) == 5

    # eval harness runs with identity-fallback matcher
    matcher = InstanceMatcher(allow_identity_fallback=True)
    samples = list(zip(batch["gt_graphs"], preds_text))
    out = evaluate_graphs(samples, ks=(20,), matcher=matcher)
    assert "R@20" in out and "mR@20" in out


def test_text2text_overfits_recall():
    """After enough steps the decoder should recover a non-trivial fraction of train graphs."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.device = "cpu"
    vocab = build_vocab(cfg)
    collate = GraphCollator(vocab=vocab, seed=0)
    ds = build_dataset(cfg, "train", vocab)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)
    model = PatchSGGModel(cfg)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    model.train()
    for _ in range(120):
        for b in loader:
            loss = model.compute_loss(b, modality="text")
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    matcher = InstanceMatcher(allow_identity_fallback=True)
    samples = []
    for b in DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate):
        preds = model.predict(b, modality="text")
        samples.extend(zip(b["gt_graphs"], preds))
    out = evaluate_graphs(samples, ks=(20,), matcher=matcher)
    # greedy text->text should memorise a good chunk of the (tiny) train set
    assert out["set/triplet"] > 0.5, out
