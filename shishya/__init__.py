"""Shishya: give any AI model your rules; get the most likely answer that follows them."""
__version__ = "0.2.0"

from .engine import Decision, Shishya
from .learn import Learner
from .rules import RegoRules, Rule, rule
from .schema import SchemaError, compile_schema

__all__ = ["Decision", "Learner", "RegoRules", "Rule", "SchemaError", "Shishya", "__version__", "compile_schema",
           "rule"]
