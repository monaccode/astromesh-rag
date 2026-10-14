import pytest

from astromesh_rag._extras import require
from astromesh_rag.errors import MissingExtraError, SpecError
from astromesh_rag.registry import BuildContext, Registry


def test_register_and_build():
    r = Registry()
    r.register("splitter", "x", lambda cfg, ctx: ("built", cfg["n"]))
    assert r.build("splitter", "x", {"n": 2}, BuildContext()) == ("built", 2)


def test_later_registration_replaces():
    r = Registry()
    r.register("splitter", "x", lambda cfg, ctx: 1)
    r.register("splitter", "x", lambda cfg, ctx: 2)
    assert r.build("splitter", "x", {}, BuildContext()) == 2


def test_unknown_name_lists_known():
    r = Registry()
    r.register("index", "memory", lambda cfg, ctx: None)
    with pytest.raises(SpecError, match=r"unknown index 'nope'.*memory"):
        r.build("index", "nope", {}, BuildContext())


def test_unknown_kind_rejected():
    with pytest.raises(SpecError):
        Registry().register("banana", "x", lambda cfg, ctx: None)


def test_missing_extra_message():
    with pytest.raises(MissingExtraError, match=r"pip install 'astromesh-rag\[qdrant\]'"):
        require("definitely_not_a_module_xyz", "qdrant")
