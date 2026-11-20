"""
Check that the torch/cupy backends give exactly the same matrices as the CPU backend.

These tests are skipped when the backend is not installed. The torch tests also
run without a GPU (device="cpu"), the cupy tests need a CUDA device.
"""
import random

import numpy as np
import pytest

from matrix_bgpsim import RMatrix

BACKENDS = {
    "torch": "cpu",  # torch device
    "cupy": 0,       # cupy device id
}

# 7 core relationships plus 14 isolated peer pairs. After the first iteration the
# state matrix has the same sum as before, although routes are still changing,
# so an early stop based on the sum of the state matrix stops too soon here.
EARLY_STOP_RELS = [
    ("20013", "53443", -1), ("20013", "343", 0), ("20013", "61681", -1),
    ("61681", "28615", -1), ("28615", "343", -1), ("28615", "59095", -1),
    ("28615", "53443", -1),
] + [(str(90000 + 2 * k), str(90001 + 2 * k), 0) for k in range(14)]


def random_rels(seed):
    """Random topology: providers always rank higher than customers (no cycles)."""
    rng = random.Random(seed)
    asns = [str(asn) for asn in rng.sample(range(1, 65536), rng.randint(4, 40))]
    p_c2p, p_p2p = rng.uniform(0.02, 0.3), rng.uniform(0.0, 0.25)
    rels = []
    for i, asn1 in enumerate(asns):
        for asn2 in asns[i + 1:]:
            r = rng.random()
            if r < p_c2p:
                rels.append((asn1, asn2, -1))
            elif r < p_c2p + p_p2p:
                rels.append((asn1, asn2, 0))
    rng.shuffle(rels)
    return rels


def simulate(rels, backend, save_next_hop, device=None):
    rmatrix = RMatrix(rels)
    rmatrix.run(n_jobs=2, max_iter=32, save_next_hop=save_next_hop,
                backend=backend, device=device)
    return rmatrix


def assert_same_matrices(rmatrix, expected):
    np.testing.assert_array_equal(rmatrix.__state__, expected.__state__)
    if expected.__next_hop__ is None:
        assert rmatrix.__next_hop__ is None
    else:
        np.testing.assert_array_equal(rmatrix.__next_hop__, expected.__next_hop__)


@pytest.fixture(params=sorted(BACKENDS))
def backend(request):
    module = pytest.importorskip(request.param)
    if request.param == "cupy":
        try:
            module.cuda.runtime.getDeviceCount()
        except Exception:
            pytest.skip("cupy needs a CUDA device")
    return request.param


@pytest.mark.parametrize("save_next_hop", [True, False])
def test_early_stop_waits_for_convergence(backend, save_next_hop):
    expected = simulate(EARLY_STOP_RELS, "cpu", save_next_hop)
    rmatrix = simulate(EARLY_STOP_RELS, backend, save_next_hop, BACKENDS[backend])
    assert_same_matrices(rmatrix, expected)
    # 343 is in the customer cone of 20013, so the customer route wins over the peer link
    assert rmatrix.get_state("20013", "343") == expected.get_state("20013", "343")
    if save_next_hop:
        assert rmatrix.get_path("20013", "343") == ["61681", "28615", "343"]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("save_next_hop", [True, False])
def test_same_as_cpu_backend(backend, seed, save_next_hop):
    rels = random_rels(seed)
    expected = simulate(rels, "cpu", save_next_hop)
    rmatrix = simulate(rels, backend, save_next_hop, BACKENDS[backend])
    assert_same_matrices(rmatrix, expected)
