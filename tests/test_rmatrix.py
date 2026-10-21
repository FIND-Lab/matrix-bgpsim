import json

import pytest

from matrix_bgpsim import RMatrix


def test_unreachable_route():
    # 1 - 2 - 3 are peers, so 1 can't reach 3 (valley-free). 4 is a stub under 1.
    rels = [("1", "2", 0), ("2", "3", 0), ("1", "4", -1)]
    rmatrix = RMatrix(rels)
    rmatrix.run(n_jobs=1)
    for asn1, asn2 in [("1", "3"), ("3", "1"), ("4", "3"), ("3", "4")]:
        assert rmatrix.get_state(asn1, asn2) == (None, 0)
        assert rmatrix.get_path(asn1, asn2) is None


def test_get_state_returns_int():
    rels = [("1", "2", -1), ("1", "3", -1), ("2", "4", -1), ("3", "4", -1), ("4", "5", -1)]
    rmatrix = RMatrix(rels)
    rmatrix.run(n_jobs=1)
    s_type, s_len = rmatrix.get_state("1", "5")
    assert (s_type, s_len) == (RMatrix.P2C, 3)
    assert type(s_len) is int
    json.dumps([s_type, s_len])


@pytest.mark.parametrize("make_excluded", [
    set, list, tuple, dict.fromkeys,
    lambda asns: (asn for asn in asns),
    lambda asns: (lambda asn: asn in asns),
], ids=["set", "list", "tuple", "dict", "generator", "callable"])
def test_excluded(make_excluded):
    rels = [("1", "2", 0), ("1", "3", -1), ("2", "4", -1), ("3", "5", -1),
            ("4", "5", -1), ("4", "6", -1), ("6", "7", -1)]
    rmatrix = RMatrix(rels, excluded=make_excluded(["3", "5"]))
    assert not rmatrix.has_asn("3")
    assert not rmatrix.has_asn("5")
    for asn in ["1", "2", "4", "6", "7"]:
        assert rmatrix.has_asn(asn)
