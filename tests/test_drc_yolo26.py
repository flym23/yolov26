# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused CPU tests for DRC-YOLO26 modules and integration."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import CCQDetect, CDRStem, SDRFusion
from ultralytics.nn.tasks import DetectionModel
from tools.urpc.materialize_experiment_model import materialize
from tools.urpc.run_ablation import default_config


def build_model() -> DetectionModel:
    """Build a small-class DRC model with the loss arguments normally set by a trainer."""
    model = DetectionModel("ultralytics/cfg/models/26/yolo26-drc.yaml", nc=4, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=3)
    return model


@pytest.mark.parametrize("shape", [(1, 3, 256, 256), (2, 3, 320, 320), (1, 3, 255, 257)])
def test_cdr_shape_finiteness_and_zero_residual(shape):
    """CDRStem preserves stride-two geometry and starts as its pretrained Conv branch."""
    module = CDRStem(3, 16).eval()
    image = torch.randn(*shape, requires_grad=True)
    output = module(image)
    base = module.act(module.bn(module.conv(image)))
    assert output.shape == base.shape
    assert torch.isfinite(output).all()
    assert (output - base).abs().max() <= 1e-6
    output.mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_sdr_shape_zero_residual_and_geometry_error():
    """SDRFusion preserves the original Concat output and rejects malformed P2 geometry."""
    module = SDRFusion([128, 128, 64]).eval()
    td, p3, p2 = torch.randn(2, 128, 20, 20), torch.randn(2, 128, 20, 20), torch.randn(2, 64, 40, 40)
    output = module([td, p3, p2])
    assert output.shape == (2, 256, 20, 20)
    assert (output - torch.cat((td, p3), dim=1)).abs().max() <= 1e-6
    with pytest.raises(ValueError, match="P2=.*P3"):
        module([td, p3, p2[..., :-1, :]])


def test_sdr_pixel_unshuffle_group_order():
    """PixelUnshuffle groups map to the four 2x2 subpixel positions without channel mixing."""
    source = torch.arange(16.0).view(1, 1, 4, 4)
    unshuffled = F.pixel_unshuffle(source, 2)
    groups = unshuffled.view(1, 1, 4, 2, 2).permute(0, 2, 1, 3, 4)
    expected = [source[..., 0::2, 0::2], source[..., 0::2, 1::2], source[..., 1::2, 0::2], source[..., 1::2, 1::2]]
    assert all(torch.equal(groups[:, index], value) for index, value in enumerate(expected))


def test_ccq_train_and_inference_output_contract():
    """CCQ adds quality only to training metadata and keeps end-to-end inference at six columns."""
    head = CCQDetect(nc=4, quality_beta=0.5, reg_max=1, end2end=True, ch=(64, 128, 256))
    features = [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16), torch.randn(2, 256, 8, 8)]
    head.train()
    predictions = head(features)
    assert {"boxes", "scores"} <= predictions["one2many"].keys()
    assert {"boxes", "scores", "quality"} <= predictions["one2one"].keys()
    assert predictions["one2one"]["quality"].shape[-1] == predictions["one2one"]["scores"].shape[-1]
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    output, _ = head(features)
    assert output.shape[-1] == 6 and torch.isfinite(output).all()


def test_ccq_quality_changes_score_ranking():
    """Equal class logits receive a higher final score for the higher predicted localization quality."""
    head = CCQDetect(nc=2, quality_beta=1.0, reg_max=1, end2end=True, ch=(4,))
    head.stride = torch.tensor([8.0])
    prediction = {
        "feats": [torch.zeros(1, 4, 1, 2)],
        "boxes": torch.zeros(1, 4, 2),
        "scores": torch.zeros(1, 2, 2),
        "quality": torch.tensor([[[-2.0, 2.0]]]),
    }
    output = head._inference(prediction)
    assert output[0, 4, 1] > output[0, 4, 0]


def test_drc_model_parse_and_loss_with_normal_and_empty_targets():
    """The complete model retains P3/P4/P5 geometry and handles populated and empty batches."""
    model = build_model().train()
    assert isinstance(model.model[0], CDRStem)
    assert isinstance(model.model[15], SDRFusion)
    assert isinstance(model.model[-1], CCQDetect)
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    batch = {
        "img": torch.randn(2, 3, 256, 256),
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0.0], [1.0], [2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.2, 0.3]]),
    }
    loss, items = model(batch)
    assert loss.shape == (5,) and torch.isfinite(loss).all()
    assert {"box_loss", "cls_loss", "l1_loss", "ccq_box_loss", "quality_loss"} <= items.keys()
    loss.sum().backward()
    model.zero_grad(set_to_none=True)
    empty = {"img": torch.randn(2, 3, 256, 256), "batch_idx": torch.empty(0, dtype=torch.long), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}
    empty_loss, empty_items = model(empty)
    assert torch.isfinite(empty_loss).all() and empty_items["ccq_box_loss"] == 0
    empty_loss.sum().backward()


def test_drc_fuse_keeps_quality_head_and_predictions_finite():
    """Fusing preserves CDR logic, removes the dense head, and retains CCQ quality inference."""
    model = build_model().eval()
    fused = deepcopy(model).fuse()
    assert fused.model[-1].cv2 is None and fused.model[-1].cv3 is None
    assert fused.model[-1].one2one_q is not None
    output, _ = fused(torch.randn(1, 3, 256, 256))
    assert output.shape[-1] == 6 and torch.isfinite(output).all()


@pytest.mark.parametrize("mode,channels", [("stride_concat", 384), ("pixel_concat", 320), ("route", 256)])
def test_sdr_documented_ablation_modes_preserve_geometry(mode, channels):
    """Each S1-S4 route has the declared output channel contract and finite activations."""
    module = SDRFusion([128, 128, 64], mode=mode, use_reliability_gate=False, zero_init=False).eval()
    output = module([torch.randn(1, 128, 20, 20), torch.randn(1, 128, 20, 20), torch.randn(1, 64, 40, 40)])
    assert output.shape == (1, channels, 20, 20) and torch.isfinite(output).all()


def test_ccq_without_quality_head_keeps_six_column_inference_contract():
    """Q1-Q3 remove the quality branch instead of retaining an untrained score multiplier."""
    head = CCQDetect(nc=4, quality_beta=0.5, enable_quality=False, reg_max=1, end2end=True, ch=(64, 128, 256))
    features = [torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16), torch.randn(1, 256, 8, 8)]
    head.train()
    assert "quality" not in head(features)["one2one"]
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    output, _ = head(features)
    assert output.shape[-1] == 6 and torch.isfinite(output).all()


@pytest.mark.parametrize("config_id", ("A4", "A5", "A6", "D1", "D2", "D3", "S1", "S2", "S3", "S4", "Q1", "Q2", "Q3", "Q4"))
def test_materialized_ablation_yaml_parses(config_id, tmp_path):
    """Every missing A/D/S/Q configuration materializes to a standalone, parseable N-scale YAML."""
    model = DetectionModel(materialize(config_id, tmp_path / f"yolo26n-{config_id}.yaml"), verbose=False)
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_formal_matrix_excludes_removed_comparison_models():
    """The approved matrix contains only C0, C1, C5 and all requested ablation identifiers."""
    experiments = default_config()["experiments"]
    assert {"C2", "C3", "C4"}.isdisjoint(experiments)
    assert list(experiments) == [
        "C0",
        "C1",
        "C5",
        *[f"A{i}" for i in range(8)],
        *[f"D{i}" for i in range(5)],
        *[f"S{i}" for i in range(6)],
        *[f"Q{i}" for i in range(6)],
    ]


def test_truncated_label_cache_is_rescanned(monkeypatch, tmp_path):
    """An interrupted cache write is treated as stale instead of aborting a serial smoke run."""
    try:
        import ultralytics.data.dataset as dataset_module
        from ultralytics.data.dataset import YOLODataset
    except ImportError:
        pytest.skip("The local test environment lacks the data-stack dependency required by this cache unit test.")
    dataset = YOLODataset.__new__(YOLODataset)
    rebuilt = {"version": "test", "hash": "test"}
    monkeypatch.setattr(dataset_module, "load_dataset_cache_file", lambda _: (_ for _ in ()).throw(EOFError()))
    dataset.cache_labels = lambda _: rebuilt
    cache, exists = dataset._load_or_scan_cache(tmp_path / "labels.cache", "test")
    assert cache is rebuilt and not exists
