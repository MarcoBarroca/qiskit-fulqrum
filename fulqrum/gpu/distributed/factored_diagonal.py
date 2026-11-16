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

"""A local Hamiltonian diagonal in factored form, applied without storing it."""

from __future__ import annotations


class FactoredDiagonal:
    """This rank's diagonal, applied by the operator's own diagonal kernel.

    Element ``b * n_a + a`` (beta-outer, alpha-inner, as ``diag_local``) is
    computed when it is used, from the tables the matvec uses for its
    diagonal, so no vector of length ``n_local`` is stored.
    """

    def __init__(self, operator, geometry, tables, num_alpha):

        self._operator = operator
        self._geometry = tuple(geometry)
        self._tables = tuple(int(t) for t in tables)
        self._num_alpha = int(num_alpha)
        self.n_a = int(geometry[1])
        self.n_b = int(geometry[3])
        self.size = self.n_a * self.n_b

    def apply_shifted_inverse(self, x, y, shift, offset=0):
        """Set ``y = x / (d - shift)`` with the clamp of the Jacobi preconditioner.

        ``x`` and ``y`` are CuPy vectors of the same length holding the
        elements ``offset:offset + len(y)``; ``y`` may be ``x``.
        """
        import cupy  # noqa: PLC0415

        from .kernel_launch import launch_diagonal_shifted_inverse  # noqa: PLC0415

        launch_diagonal_shifted_inverse(
            self._geometry,
            *self._tables,
            self._num_alpha,
            int(x.data.ptr),
            int(y.data.ptr),
            int(offset),
            int(y.size),
            float(shift),
            int(cupy.cuda.get_current_stream().ptr),
        )

    def materialize(self):
        """Return the diagonal as a new CuPy vector (for example, to find its minimum)."""
        return self._operator.diag_local_device
