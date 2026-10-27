
"""Compiling a JSON Schema subset into decision fields."""

import json

import pytest

from shishya.schema import SchemaError, compile_schema


def obj(**props):
    return {"type": "object", "properties": props}


def test_compiles_boolean_and_enum_fields():
    comp = compile_schema(obj(
        notify_hr={"type": "boolean"},
        action={"type": "string", "enum": ["approve", "deny"]},
    ))
    assert comp.names == ["notify_hr", "action"]
    notify_hr = comp.field("notify_hr")
    assert notify_hr.kind == "noul"
    assert notify_hr.options == (True, False)
    action = comp.field("action")
    assert action.kind == "choice"
    assert action.options == ("approve", "deny")
    assert action.option_text == {}
    assert action.when is None
    assert action.option_when == {}


def test_enum_without_type_is_accepted():
    comp = compile_schema(obj(color={"enum": ["red", "blue"]}))
    assert comp.field("color").options == ("red", "blue")


def test_descriptions_become_instructions():
    comp = compile_schema(obj(
        leave={"type": "boolean", "description": "  Approve the leave request?  "},
        action={"type": "string", "enum": ["a"], "description": "Pick one action."},
    ))
    assert comp.field("leave").instructions == "Approve the leave request?"
    assert comp.field("action").instructions == "Pick one action."


def test_missing_description_gives_humanized_question():
    comp = compile_schema(obj(leave_balance={"type": "boolean"}, rule_id={"type": "boolean", "description": " "}))
    assert comp.field("leave_balance").instructions == "Leave balance?"
    assert comp.field("rule_id").instructions == "Rule id?"


def test_x_shishya_options_when_and_option_when_are_parsed():
    cond = {"path": "facts.leave_balance", "gte": 3}
    comp = compile_schema(obj(
        verdict={
            "type": "string",
            "enum": ["approve", "deny", "review"],
            "x-shishya": {
                "options": {"approve": "Grant the leave as asked", "deny": "Refuse the leave"},
                "when": {"path": "facts.requested", "eq": True},
                "option_when": {"approve": cond},
            },
        },
    ))
    f = comp.field("verdict")
    assert f.option_text == {"approve": "Grant the leave as asked", "deny": "Refuse the leave"}
    assert f.when == {"path": "facts.requested", "eq": True}
    assert f.option_when == {"approve": cond}


def test_x_shishya_when_on_boolean():
    comp = compile_schema(obj(notify_hr={"type": "boolean", "x-shishya": {"when": {"exists": "facts.order"}}}))
    assert comp.field("notify_hr").when == {"exists": "facts.order"}


# -- errors -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("schema, match", [
    ([], "must be a JSON object"),
    ({"type": "array", "properties": {"a": {"type": "boolean"}}}, 'must have "type": "object"'),
    ({"properties": {"a": {"type": "boolean"}}}, 'must have "type": "object"'),
    ({"type": "object", "properties": {}}, "non-empty 'properties'"),
    ({"type": "object"}, "non-empty 'properties'"),
    (obj(amount={"type": "number"}), "supports only boolean and string enum"),
    (obj(name={"type": "string"}), "supports only boolean and string enum"),
    (obj(a={"type": "string", "enum": ["x", "x"]}), "duplicate values"),
    (obj(a={"type": "string", "enum": ["x"], "x-shishya": {"options": {"y": "nope"}}}),
     "names values not in enum"),
    (obj(a={"type": "string", "enum": ["x"], "x-shishya": {"colour": "red"}}), "unknown keys"),
    (obj(a={"type": "string", "enum": ["x"], "x-shishya": {"when": {"path": "state.x", "eq": 1}}}),
     "when"),
    (obj(a={"type": "string", "enum": ["x"], "x-shishya": {"option_when": {"x": {"path": "facts.a"}}}}),
     "option_when.x"),
    (obj(a={"type": "boolean", "enum": [True, False]}), "boolean takes no enum"),
    (obj(a={"type": "boolean", "x-shishya": {"options": {"yes": "Yes"}}}), "boolean takes no enum"),
    (obj(a={"type": "string", "enum": []}), "non-empty list"),
    (obj(a={"type": "string", "enum": [1, 2]}), "must be non-empty strings"),
])
def test_schema_errors(schema, match):
    with pytest.raises(SchemaError, match=match):
        compile_schema(schema)


def test_order_reorders_fields():
    schema = obj(a={"type": "boolean"}, b={"type": "boolean"}, c={"type": "boolean"})
    schema["x-shishya"] = {"order": ["c", "a", "b"]}
    assert compile_schema(schema).names == ["c", "a", "b"]


@pytest.mark.parametrize("order", [
    ["a", "b"],                 # misses c
    ["a", "b", "c", "d"],       # names d, which is not a property
    ["a", "a", "b", "c"],       # repeats a
])
def test_order_must_list_every_property_exactly_once(order):
    schema = obj(a={"type": "boolean"}, b={"type": "boolean"}, c={"type": "boolean"})
    schema["x-shishya"] = {"order": order}
    with pytest.raises(SchemaError, match="must list every property exactly once"):
        compile_schema(schema)


def test_no_order_keeps_property_order():
    comp = compile_schema(obj(z={"type": "boolean"}, a={"type": "boolean"}))
    assert comp.names == ["z", "a"]


# -- digest -------------------------------------------------------------------------------------------

def test_digest_is_stable_and_keeps_property_order():
    a = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approve", "deny"], "description": "Decide."},
            "urgent": {"type": "boolean"},
        },
    }
    same = json.loads(json.dumps(a))
    assert compile_schema(a).digest == compile_schema(same).digest
    # property order is meaningful (search and answer order), so it changes the digest
    swapped = {"type": "object", "properties": {"urgent": a["properties"]["urgent"],
                                                "verdict": a["properties"]["verdict"]}}
    assert compile_schema(a).digest != compile_schema(swapped).digest


def test_digest_changes_when_schema_changes():
    a = obj(verdict={"type": "string", "enum": ["approve", "deny"]})
    b = obj(verdict={"type": "string", "enum": ["approve", "deny", "review"]})
    assert compile_schema(a).digest != compile_schema(b).digest
