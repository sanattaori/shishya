"""Conditions are evaluated by plain code: true, false, or unknown (missing facts are unknown, never false)."""
import pytest

from shishya.when import ConditionError, check, evaluate, holds

CTX = {
    "input": {"text": "need leave"},
    "facts": {
        "n": 5,
        "f": 2.5,
        "name": "abc",
        "flag": True,
        "zero": 0,
        "s": "5",
        "holdings": ["TCS", "INFY"],
        "items": [{"qty": 5}, {"qty": 1}],
        "country": "IN",
    },
}


# -- leaf operators -----------------------------------------------------------------------------------

@pytest.mark.parametrize("cond, expected", [
    ({"path": "facts.n", "eq": 5}, True),
    ({"path": "facts.n", "eq": 4}, False),
    ({"path": "facts.country", "eq": "IN"}, True),
    ({"path": "facts.country", "eq": "US"}, False),
    ({"path": "facts.n", "neq": 4}, True),
    ({"path": "facts.n", "neq": 5}, False),
    ({"path": "facts.gt_check", "eq": None}, None),        # missing path is unknown, not None-equal
    ({"path": "facts.n", "gt": 3}, True),
    ({"path": "facts.n", "gt": 5}, False),
    ({"path": "facts.n", "gte": 5}, True),
    ({"path": "facts.n", "gte": 6}, False),
    ({"path": "facts.n", "lt": 6}, True),
    ({"path": "facts.n", "lt": 5}, False),
    ({"path": "facts.n", "lte": 5}, True),
    ({"path": "facts.n", "lte": 4}, False),
    ({"path": "facts.country", "in": ["IN", "US"]}, True),
    ({"path": "facts.country", "in": ["UK"]}, False),
    ({"path": "facts.holdings", "contains": "TCS"}, True),
    ({"path": "facts.holdings", "contains": "WIPRO"}, False),
    ({"path": "facts.name", "contains": "b"}, True),
    ({"path": "facts.name", "contains": "z"}, False),
    ({"path": "input.text", "contains": "leave"}, True),
])
def test_leaf_operators(cond, expected):
    if expected is None:
        assert evaluate(cond, CTX) is None
    else:
        assert evaluate(cond, CTX) is expected
        assert holds(cond, CTX) is expected


def test_float_compares_with_int():
    assert evaluate({"path": "facts.f", "gt": 2}, CTX) is True
    assert evaluate({"path": "facts.f", "lte": 2}, CTX) is False


# -- unknown (missing paths) --------------------------------------------------------------------------

@pytest.mark.parametrize("op, value", [
    ("eq", 1), ("neq", 1), ("gt", 1), ("gte", 1), ("lt", 1), ("lte", 1),
    ("in", [1, 2]), ("contains", "x"),
])
def test_missing_path_is_unknown_for_every_operator(op, value):
    cond = {"path": "facts.missing", op: value}
    assert evaluate(cond, CTX) is None
    assert holds(cond, CTX) is False


def test_missing_root_key_is_unknown():
    assert evaluate({"path": "facts.deep.er.still", "eq": 1}, {"facts": {}}) is None


def test_not_of_unknown_stays_unknown():
    cond = {"not": {"path": "facts.missing", "eq": 1}}
    assert evaluate(cond, CTX) is None
    assert holds(cond, CTX) is False


def test_not_of_known_values():
    assert evaluate({"not": {"path": "facts.n", "eq": 5}}, CTX) is False
    assert evaluate({"not": {"path": "facts.n", "eq": 6}}, CTX) is True


def test_all_three_valued():
    unknown = {"path": "facts.missing", "eq": 1}
    yes = {"path": "facts.n", "eq": 5}
    no = {"path": "facts.n", "eq": 6}
    assert evaluate({"all": [yes, yes]}, CTX) is True
    assert evaluate({"all": [yes, unknown]}, CTX) is None
    assert evaluate({"all": [no, unknown]}, CTX) is False
    assert holds({"all": [yes, unknown]}, CTX) is False


def test_any_three_valued():
    unknown = {"path": "facts.missing", "eq": 1}
    yes = {"path": "facts.n", "eq": 5}
    no = {"path": "facts.n", "eq": 6}
    assert evaluate({"any": [no, no]}, CTX) is False
    assert evaluate({"any": [no, unknown]}, CTX) is None
    assert evaluate({"any": [yes, unknown]}, CTX) is True
    assert holds({"any": [no, unknown]}, CTX) is False


def test_exists():
    assert evaluate({"exists": "facts.n"}, CTX) is True
    assert evaluate({"exists": "facts.missing"}, CTX) is False
    assert evaluate({"exists": "facts.items.0.qty"}, CTX) is True
    # exists is never unknown: a missing fact is simply not there
    assert evaluate({"not": {"exists": "facts.missing"}}, CTX) is True


# -- paths --------------------------------------------------------------------------------------------

def test_list_index_paths():
    assert evaluate({"path": "facts.items.0.qty", "gte": 5}, CTX) is True
    assert evaluate({"path": "facts.items.1.qty", "gte": 5}, CTX) is False
    assert evaluate({"path": "facts.holdings.1", "eq": "INFY"}, CTX) is True


def test_list_index_out_of_range_is_unknown():
    assert evaluate({"path": "facts.items.3.qty", "gte": 1}, CTX) is None


def test_list_index_on_non_index_part_is_unknown():
    assert evaluate({"path": "facts.items.first.qty", "gte": 1}, CTX) is None


# -- no coercion, bool is not a number ----------------------------------------------------------------

def test_no_number_string_coercion_for_ordering():
    # "5" is a string, 3 is a number: no coercion, so the comparison is unknown
    assert evaluate({"path": "facts.s", "gt": 3}, CTX) is None
    assert evaluate({"path": "facts.n", "gt": "3"}, CTX) is None


def test_no_coercion_for_equality():
    assert evaluate({"path": "facts.s", "eq": 5}, CTX) is False
    assert evaluate({"path": "facts.n", "eq": "5"}, CTX) is False


def test_bool_is_not_a_number_for_ordering():
    assert evaluate({"path": "facts.flag", "gt": 0}, CTX) is None
    assert evaluate({"path": "facts.zero", "lt": True}, CTX) is None


def test_bool_equality_is_plain_equality():
    assert evaluate({"path": "facts.flag", "eq": True}, CTX) is True
    assert evaluate({"path": "facts.flag", "eq": False}, CTX) is False


# -- bad shapes ---------------------------------------------------------------------------------------

BAD_SHAPES = [
    ({}, "non-empty"),
    ({"path": "facts.n", "eq": 5, "gt": 1}, "exactly one operator"),
    ({"path": "state.x", "eq": 1}, "must start with one of"),
    ({"path": "facts.country", "in": "IN"}, "needs a list"),
]


@pytest.mark.parametrize("cond, match", BAD_SHAPES)
def test_bad_shape_raises_from_evaluate(cond, match):
    with pytest.raises(ConditionError, match=match):
        evaluate(cond, CTX)


@pytest.mark.parametrize("cond, match", BAD_SHAPES)
def test_bad_shape_raises_from_check(cond, match):
    with pytest.raises(ConditionError, match=match):
        check(cond)


def test_bad_shape_raises_from_holds():
    with pytest.raises(ConditionError):
        holds({"path": "state.x", "eq": 1}, CTX)


def test_check_accepts_good_conditions():
    check({"all": [
        {"path": "facts.n", "gte": 1},
        {"not": {"exists": "facts.x"}},
        {"any": [{"path": "input.text", "contains": "x"}]},
    ]})


def test_contains_with_non_string_value_on_string_path():
    assert evaluate({"path": "facts.name", "contains": 1}, CTX) is None


@pytest.mark.parametrize("cond", [
    {"path": "facts.x", "eq": True},
    {"path": "facts.x", "in": [True, "a"]},
    {"path": "facts.xs", "contains": True},
])
def test_booleans_are_not_numbers(cond):
    assert evaluate(cond, {"facts": {"x": 1, "xs": [1, 0]}}) is False
    assert evaluate({"path": "facts.x", "neq": True}, {"facts": {"x": 1}}) is True
