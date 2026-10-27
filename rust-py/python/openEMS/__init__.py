# -*- coding: utf-8 -*-
"""openEMS Python package (Rust core).

Import path mirrors the native package:

    import openEMS
    from openEMS import openEMS as solver
    from openEMS.utilities import HDF5Dump
    from openEMS.nf2ff import nf2ff_results

The solver class is the PyO3 wrapper from `openEMS._openems`; the other modules
are the native pure-Python modules and run unchanged against the Rust dump
format.
"""

from __future__ import absolute_import

import CSXCAD  # noqa: F401  (sets up the native import path)

try:
    from importlib.metadata import version

    __version__ = version("openEMS")
except Exception:  # pragma: no cover - only used without installed metadata
    __version__ = "0.0.0"

from openEMS.openEMS import openEMS  # noqa: E402,F401
