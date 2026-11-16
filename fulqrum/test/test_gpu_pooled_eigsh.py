# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Device-side diagonals for the distributed PRIMME solver."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    cupy = None

from fulqrum.gpu.distributed import _collective as col


def _gpu_backend_available() -> bool:
    try:
        from fulqrum.gpu.eigensolver.primme_distributed import _load_backend

        return bool(_load_backend().gpu_backend_available())
    except ImportError:
        return False


needs_gpu = pytest.mark.skipif(cupy is None, reason="cupy required")
needs_gpu_primme = pytest.mark.skipif(
    cupy is None or not _gpu_backend_available(),
    reason="cupy and the cuBLAS PRIMME backend required",
)


def _hwn_strings(width: int, weight: int) -> list[str]:
    out = []
    for ones in combinations(range(width), weight):
        bits = ["0"] * width
        for i in ones:
            bits[width - 1 - i] = "1"
        out.append("".join(bits))
    return sorted(out)


def _lih_operator(monkeypatch, recompute: str):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    import fulqrum as fq
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    monkeypatch.setenv("FQ_DIAG_RECOMPUTE", recompute)
    OP = fq.FermionicOperator.from_json(
        Path(__file__).parent / "data/lih.json"
    ).extended_jw_transformation()
    OP.set_type(2)
    half = OP.width // 2
    alpha = _hwn_strings(half, 2)
    hsub = fq.SubspaceHamiltonian(OP, fq.Subspace([alpha, list(alpha)]))
    col.reset_for_testing()
    return TileGPUResidentHamiltonian(
        hsub, alpha, list(alpha), comm=MPI.COMM_SELF, comm_backend="host_mpi"
    )


@needs_gpu
@pytest.mark.parametrize("recompute", ["0", "1"], ids=["stored", "recomputed"])
def test_device_and_factored_diagonals_match_diag_local(monkeypatch, recompute):
    H = _lih_operator(monkeypatch, recompute)
    host = np.asarray(H.diag_local, dtype=np.float64)

    device = H.diag_local_device
    assert isinstance(device, cupy.ndarray) and device.shape == host.shape
    if recompute == "0":
        # the stored diagonal itself
        np.testing.assert_array_equal(cupy.asnumpy(device), host)
    else:
        # the GPU diagonal kernel and the CPU path round differently
        np.testing.assert_allclose(cupy.asnumpy(device), host, rtol=0, atol=1e-12)

    factored = H.diag_local_factored
    if recompute == "0":
        assert factored is None
        return
    assert factored.size == host.size
    np.testing.assert_allclose(
        cupy.asnumpy(factored.materialize()), host, rtol=0, atol=1e-12
    )
    rng = np.random.default_rng(3)
    off, m, shift = 7, host.size - 11, float(host.min()) + 0.125
    x = rng.uniform(-1.0, 1.0, m)
    d_y = cupy.asarray(x)
    factored.apply_shifted_inverse(d_y, d_y, shift, offset=off)
    t = host[off : off + m] - shift
    expected = x / np.copysign(np.maximum(np.abs(t), 1e-10), t)
    np.testing.assert_allclose(cupy.asnumpy(d_y), expected, rtol=1e-13, atol=0)


@needs_gpu_primme
@pytest.mark.parametrize("gpu", [False, True], ids=["host_leg", "gpu_leg"])
@pytest.mark.parametrize("kind", ["device", "factored"])
def test_device_diagonal_preconditioners_match_the_host_diagonal(
    monkeypatch, gpu, kind
):
    from fulqrum.gpu.eigensolver.primme_distributed import distributed_primme_eigsh

    H = _lih_operator(monkeypatch, "1")
    host = np.asarray(H.diag_local, dtype=np.float64)
    v0 = np.zeros(H.n_local)
    v0[int(host.argmin())] = 1.0
    precond = H.diag_local_device if kind == "device" else H.diag_local_factored
    assert precond is not None
    energies = []
    for diag in (host, precond):
        evals, evecs = distributed_primme_eigsh(
            H, k=1, tol=1e-9, conv="abs", precond_diag_local=diag, v0_local=v0, gpu=gpu
        )
        assert isinstance(evecs, np.ndarray) and evecs.shape == (H.n_local, 1)
        energies.append(float(evals[0]))
    assert energies[1] == pytest.approx(energies[0], abs=1e-10)


@needs_gpu
def test_device_diagonal_with_the_wrong_length_is_rejected(monkeypatch):
    from fulqrum.gpu.eigensolver.primme_distributed import _local_inputs

    H = _lih_operator(monkeypatch, "1")
    with pytest.raises(ValueError, match="precond_diag_local"):
        _local_inputs(H, cupy.zeros(H.n_local + 1), None)


@needs_gpu
def test_chunked_host_to_device_copy(monkeypatch):
    import fulqrum.gpu.eigensolver.primme_distributed as pd

    monkeypatch.setattr(pd, "_HOST_COPY_ROWS", 1000)
    rng = np.random.default_rng(11)
    host = np.asfortranarray(rng.normal(size=(10_007, 3)))
    device = cupy.zeros((10_007, 3), dtype=cupy.float64, order="F")
    pd._copy_to_device_columns(host, device, 2)
    out = cupy.asnumpy(device)
    np.testing.assert_array_equal(out[:, :2], host[:, :2])
    assert not out[:, 2].any()
