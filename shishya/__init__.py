"""Shishya: give any AI model your rules; get the most likely answer that follows them."""
__version__ = "0.1.0"

from .engine import Decision, Shishya
from .rules import RegoRules, Rule, rule
from .schema import SchemaError, compile_schema

__all__ = ["Decision", "RegoRules", "Rule", "SchemaError", "Shishya", "__version__", "compile_schema", "rule"]
