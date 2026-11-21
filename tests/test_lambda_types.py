"""estimate's list of the experiment's GPU in every size Lambda offers (pure)."""

from tinytools import lambda_run as lr


def t(cents, desc, regions=("us-east-1",)):
    return {"instance_type": {"price_cents_per_hour": cents, "description": desc},
            "regions_with_capacity_available": [{"name": r} for r in regions]}


TYPES = {
    "gpu_1x_a100_sxm4": t(129, "1x A100 (40 GB SXM4)"),
    "gpu_1x_a100": t(129, "1x A100 (40 GB PCIe)", ()),
    "gpu_8x_a100_80gb_sxm4": t(1432, "8x A100 (80 GB SXM4)"),
    "gpu_1x_h100_sxm5": t(429, "1x H100 (80 GB SXM5)"),
}


def test_gpu_family():
    assert lr.gpu_family("gpu_1x_a100_sxm4") == "a100"
    assert lr.gpu_family("gpu_8x_a100_80gb_sxm4") == "a100"
    assert lr.gpu_family("gpu_1x_h100_sxm5") == "h100"
    assert lr.gpu_family("odd") == "odd"


def test_same_gpu_lines_lists_every_a100_cheapest_first():
    lines = lr.same_gpu_lines(TYPES, "gpu_1x_a100_sxm4")
    assert lines[0].startswith("a100 types")
    assert [ln.split()[1] if ln.startswith(" *") else ln.split()[0] for ln in lines[1:]] == [
        "gpu_1x_a100", "gpu_1x_a100_sxm4", "gpu_8x_a100_80gb_sxm4"]
    assert lines[2].startswith(" * gpu_1x_a100_sxm4") and "40 GB SXM4" in lines[2]
    assert "[no capacity]" in lines[1]
    assert all("h100" not in ln for ln in lines)


def test_same_gpu_lines_when_none():
    assert lr.same_gpu_lines({}, "gpu_1x_a100_sxm4") == ["no a100 types in Lambda's list now"]


def test_a_missing_type_is_refused_with_its_family_named():
    exp = lr.Experiment.__new__(lr.Experiment)
    object.__setattr__(exp, "instance_type", "gpu_1x_a100_80gb")
    try:
        lr.make_estimate(exp, TYPES, [])
    except lr.Refused as e:
        assert "gpu_1x_a100_sxm4" in str(e) and "h100" not in str(e)
    else:
        raise AssertionError("not refused")
