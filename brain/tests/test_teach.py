"""The demo's weights are in the repo and rebuild what it needs. These tests don't need the 400 MB model."""
import torch

from brain import v5
from brain.teach import READER, load_address


def test_reader_is_the_one_v7_used():
    ck = torch.load(READER, weights_only=True)
    assert ck["args"] == {"who": "first", "pool": True, "layer": 3, "r": 32, "power": 16.0}
    assert ck["reader"]["W"].shape == (1024, 32)


def test_rare_names_pass_the_gate_and_common_ones_dont():
    who = v5.Who(load_address(), power=16.0, first=True)
    def passes(word):
        return who.word(word.encode())[1] ** who.power >= who.gate
    assert passes("Priya") and passes("Zubak")
    assert not passes("Marie") and not passes("John")
