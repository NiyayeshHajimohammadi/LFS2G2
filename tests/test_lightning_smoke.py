"""Lightning training/validation smoke test on the toy diagnostic (skipped if pl is absent)."""
import pytest

pytest.importorskip("pytorch_lightning")

import pytorch_lightning as pl  # noqa: E402

from patchsgg.config import load_config  # noqa: E402
from patchsgg.lightning_module import SGGDataModule, SGGLightning  # noqa: E402


def test_lightning_fast_dev_run(tmp_path):
    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        [
            "vocab.max_num_rels=6",
            "data.toy_n_train=16",
            "data.toy_n_val=8",
            "data.toy_max_rels=2",
            "train.batch_size=8",
            "eval.max_rels=6",
            f"output_dir={tmp_path.as_posix()}",
        ],
    )
    cfg.device = "cuda"
    dm = SGGDataModule(cfg)
    model = SGGLightning(cfg)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)  # one train + one val batch; asserts the wiring runs


def test_checkpoint_roundtrip(tmp_path):
    cfg = load_config(
        "patchsgg/configs/diagnostic_text2text.yaml",
        ["vocab.max_num_rels=6", "data.toy_n_train=16", "data.toy_n_val=8", "train.batch_size=8"],
    )
    cfg.device = "cuda"
    model = SGGLightning(cfg)
    ckpt = tmp_path / "m.ckpt"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=SGGDataModule(cfg))
    trainer.save_checkpoint(ckpt.as_posix())
    reloaded = SGGLightning.from_checkpoint(ckpt.as_posix())
    assert reloaded.cfg.eval.eval_modality == cfg.eval.eval_modality
