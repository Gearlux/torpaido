"""A record item splits into tensor parts, scalar parts and facts — by introspection, overridable per type."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
import torch
from recordstream import Image, Label, register_item

from torpaido.lowering import ItemRule, ItemValue, item_rule, lowers_item


def test_a_plain_array_is_one_data_part() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    parts, facts = item_rule(np.ndarray).split(array)

    assert list(parts) == ["data"] and parts["data"] is array and facts == {}
    joined = item_rule(np.ndarray).join(parts, facts)
    assert type(joined) is np.ndarray and np.array_equal(joined, array)


def test_a_torch_tensor_entry_is_one_data_part() -> None:
    tensor = torch.arange(4)
    parts, facts = item_rule(torch.Tensor).split(tensor)

    assert list(parts) == ["data"] and facts == {}
    assert torch.equal(item_rule(torch.Tensor).join({"data": np.arange(4)}, {}), tensor)


def test_an_array_item_is_its_data_plus_its_declared_attributes_as_facts() -> None:
    image = Image(np.zeros((2, 2, 3), dtype=np.uint8), layout="CHW")
    parts, facts = item_rule(Image).split(image)

    assert list(parts) == ["data"] and parts["data"].dtype == np.uint8
    assert facts == {"layout": "CHW"}
    joined = item_rule(Image).join(parts, facts)
    assert isinstance(joined, Image) and joined.layout == "CHW"


def test_a_label_s_class_id_is_a_scalar_part_and_its_name_a_fact() -> None:
    parts, facts = item_rule(Label).split(Label(3))
    assert parts == {"value": 3} and facts == {"classes": None}
    assert item_rule(Label).join({"value": np.int64(3)}, {"classes": None}) == Label(3)

    parts, facts = item_rule(Label).split(Label("cat", classes=["cat", "dog"]))
    assert parts == {} and facts == {"value": "cat", "classes": ["cat", "dog"]}


@pytest.mark.parametrize(
    "value, parts, facts", [(2.5, {"value": 2.5}, {}), (7, {"value": 7}, {}), ("x", {}, {"value": "x"})]
)
def test_a_plain_value_is_a_scalar_part_or_a_fact(value: object, parts: dict, facts: dict) -> None:
    rule = item_rule(type(value))
    assert rule.split(value) == (parts, facts)
    assert rule.join(parts, facts) == value


@register_item
@dataclass
class Reading:
    data: Optional[np.ndarray] = None
    offsets: Optional[List[float]] = None
    unit: str = ""
    gain: float = 1.0


def test_a_dataclass_item_splits_field_by_field() -> None:
    reading = Reading(np.ones(3), [0.5, 1.5], "dB", 2.0)
    parts, facts = item_rule(Reading).split(reading)

    assert set(parts) == {"data", "offsets", "gain"}
    assert isinstance(parts["offsets"], np.ndarray) and parts["gain"] == 2.0
    assert facts == {"unit": "dB"}
    joined = item_rule(Reading).join({**parts, "gain": np.float64(2.0)}, facts)
    assert isinstance(joined, Reading) and joined.data is not None and reading.data is not None
    assert np.array_equal(joined.data, reading.data)
    assert joined.offsets == [0.5, 1.5] and isinstance(joined.offsets, list)
    assert joined.unit == "dB" and joined.gain == 2.0 and type(joined.gain) is float


def test_a_registered_rule_replaces_the_introspected_one() -> None:
    @dataclass
    class Span:
        lo: float = 0.0
        hi: float = 1.0

    @lowers_item(Span)
    class SpanRule(ItemRule):
        def split(self, item: Span) -> tuple:
            return {"bounds": np.array([item.lo, item.hi])}, {}

        def join(self, parts: dict, facts: dict) -> Span:
            return Span(float(parts["bounds"][0]), float(parts["bounds"][1]))

    assert isinstance(item_rule(Span), SpanRule)
    parts, _ = item_rule(Span).split(Span(1.0, 2.0))
    assert list(parts) == ["bounds"]


def test_an_item_value_replaces_a_part_or_a_fact_by_name() -> None:
    value = ItemValue(Image, {"data": torch.zeros(2, 2, 3)}, {"layout": "HWC"})

    brighter = value.replace(data=value.data + 1.0)
    assert brighter.item_type is Image and brighter.facts == {"layout": "HWC"} and float(brighter.data.mean()) == 1.0
    assert value.replace(layout="CHW").facts == {"layout": "CHW"}
    assert ItemValue(float, {"value": 2.0}).value == 2.0


def test_an_item_value_refuses_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="no part or fact named 'size'"):
        ItemValue(Image, {"data": torch.zeros(1)}, {"layout": "HWC"}).replace(size=3)
