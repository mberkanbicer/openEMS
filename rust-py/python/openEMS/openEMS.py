# -*- coding: utf-8 -*-
"""Native `openEMS` solver class backed by the Rust core.

The PyO3 class is named `PyOpenEMS` in the extension module. This module
exposes it under the native name `openEMS` and adds the camelCase method
aliases used by the native Python and MATLAB/Octave layers.
"""

from __future__ import absolute_import

try:
    from openEMS._openems import PyOpenEMS as _RustOpenEMS
except ImportError:  # extension built as a standalone module
    from openems_py import PyOpenEMS as _RustOpenEMS


class openEMS(_RustOpenEMS):
    """openEMS FDTD solver (native method names)."""

    # -- constructor/setup -------------------------------------------------
    def SetBoundaryCond(self, bc):
        return self.set_boundary_cond(bc)

    def SetGaussExcite(self, f0, fc):
        return self.set_gauss_excite(f0, fc)

    def SetSinusExcite(self, f0):
        return self.set_sinus_excite(f0)

    def SetDiracExcite(self, fmax):
        return self.set_dirac_excite(fmax)

    def SetStepExcite(self, fmax):
        return self.set_step_excite(fmax)

    def SetCustomExcite(self, function, f0, fmax):
        return self.set_custom_excite(function, f0, fmax)

    def SetNumberOfThreads(self, n):
        return self.set_number_of_threads(n)

    def SetAbort(self, val):
        return self.set_abort(val)

    def SetEnableDumps(self, val):
        return self.set_enable_dumps(val)

    # -- execution ---------------------------------------------------------
    def Run(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def SetupFDTD(self, *args, **kwargs):
        return self.setup_fdtd(*args, **kwargs)

    def RunFDTD(self, *args, **kwargs):
        return self.run_fdtd(*args, **kwargs)

    def Reset(self):
        return self.reset()

    @staticmethod
    def WelcomeScreen():
        return _RustOpenEMS.welcome_screen()
