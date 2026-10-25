# -*- coding: utf-8 -*-
"""SAR averaging and calculation.

The calculation itself is the Rust implementation in ``openems_core::sar``; this
module keeps the native ``SAR_Calculation`` class, its keyword arguments and its
error messages.
"""

import os

from openEMS._openems import PySarCalculation as _RustSarCalculation


class SAR_Calculation(_RustSarCalculation):
    """SAR averaging and calculation.

    Keyword arguments accepted by the constructor (all optional):

    mass : float
        Averaging mass in grams (0 = local SAR, default 0).
    method : str
        Spatial averaging method: 'SIMPLE' (default), 'IEEE_C95_3', or
        'IEEE_62704'.
    verbose : int
        Debug verbosity level.
    autoRange : float
        Restrict the calculation to the cells within this many dB of the
        peak local SAR.
    EnableCubeStats : bool
        Record per-cube averaging statistics in the output file.
    """

    def __init__(self, **kw):
        for key, value in kw.items():
            if key == "mass":
                self.SetAveragingMass(value)
            elif key == "method":
                self.SetAveragingMethod(value)
            elif key == "verbose":
                self.SetDebugLevel(int(value))
            elif key == "debug":
                self.SetDebugLevel(int(value))
            elif key == "autoRange":
                self.EnableAutoRange(float(value))
            elif key == "EnableCubeStats":
                if value:
                    self.EnableCubeStats()
            else:
                raise Exception('Unknown keyword argument: "{}"'.format(key))

    def SetDebugLevel(self, level):
        """Set the verbosity of the calculation."""
        self.set_debug_level(int(level))

    def EnableProgress(self, enable):
        """Enable or disable the progress indicator during averaging."""
        self.enable_progress(bool(enable))

    def SetAveragingMass(self, mass):
        """Set averaging mass in grams (0 = local SAR)."""
        self.set_averaging_mass(float(mass) / 1000.0)

    def SetAveragingMethod(self, method, silent=True):
        """Set the spatial averaging method.

        Returns True on success, False if the method name is unknown.
        """
        return self.set_averaging_method(str(method), bool(silent))

    def EnableAutoRange(self, dBmax):
        """Restrict the calculation to the region around the peak."""
        self.enable_auto_range(float(dBmax))

    def EnableCubeStats(self):
        """Record per-cell averaging cube statistics in the output file."""
        self.enable_cube_stats()

    def CalcFromHDF5(self, h5_fn, out_name, export_cube_stats=False, numThreads=0):
        """Read raw field data from h5_fn, run the SAR calculation, and write
        the results to out_name. Returns True on success.

        numThreads=0 uses all available hardware threads.
        """
        if not os.path.exists(h5_fn):
            raise Exception('File "{}" does not exist'.format(h5_fn))
        if export_cube_stats:
            self.EnableCubeStats()
        return self.calc_from_hdf5(str(h5_fn), str(out_name), int(numThreads))


__all__ = ["SAR_Calculation"]
