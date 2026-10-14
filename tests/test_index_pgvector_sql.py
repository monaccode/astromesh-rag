import pytest

from astromesh_rag.errors import SpecError
from astromesh_rag.indexes.pgvector import PgVectorIndex, _where


def test_filter_uses_placeholders_not_values():
    sql, params = _where({"lang": "x'; DROP TABLE t;--", "k": "v"}, start=3)
    assert sql == "WHERE metadata->>$3 = $4 AND metadata->>$5 = $6"
    assert "DROP" not in sql
    assert params == ["lang", "x'; DROP TABLE t;--", "k", "v"]


def test_no_filter_is_empty():
    assert _where(None, start=3) == ("", [])


def test_table_name_validated():
    with pytest.raises(SpecError):
        PgVectorIndex(dsn="x", table="t; DROP TABLE y", dimension=3)
