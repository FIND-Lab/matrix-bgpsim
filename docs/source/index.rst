.. matrix-bgpsim documentation master file, created by
   sphinx-quickstart on Tue Sep  2 00:11:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

matrix-bgpsim documentation
===========================

`matrix-bgpsim` is a Python project that provides a high-performance BGP routing simulator built on matrix operations. It enables full-scale AS-level route generation across the entire global Internet topology (77k+ ASes) in just a few hours, following the Gao-Rexford simulation model. It supports CPU backend with Python-native multiprocessing, and optionally GPU acceleration via PyTorch and CuPy backends.

Current version: |release|.

CPU multiprocessing
-------------------

Version 1.0.2 fixes CPU worker initialization with the ``spawn`` and
``forkserver`` start methods, including the defaults on Windows and macOS,
and on Linux with Python 3.14. See `PR #2
<https://github.com/FIND-Lab/matrix-bgpsim/pull/2>`_.

When running a CPU simulation from a Python script, protect the entry point
with ``if __name__ == "__main__":`` so worker processes can safely import it:

.. code-block:: python

   from matrix_bgpsim import RMatrix

   if __name__ == "__main__":
       rmatrix = RMatrix("20250101.as-rel2.txt")
       rmatrix.run(n_jobs=4, backend="cpu", save_next_hop=True)
       print(rmatrix.get_path("7018", "3356"))

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   matrix_bgpsim
