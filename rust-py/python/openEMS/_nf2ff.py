# -*- coding: utf-8 -*-
"""Near-field to far-field transformation (native ``openEMS._nf2ff``).

The transform is the Rust implementation in ``openems_core::nf2ff``; this module
keeps the native class name, the file checks and the argument handling.
"""

import os

from openEMS._openems import PyNf2ff as _RustNf2ff


class _nf2ff(_RustNf2ff):
    """Near-field to far-field transformer."""

    def __init__(self, freq, theta, phi, center, numThreads=0, **kw):
        # every setting is applied by the Rust constructor
        self._ = None

    def AnalyseFile(self, e_file, h_file):
        if not os.path.exists(e_file):
            raise Exception('File "{}" does not exist'.format(e_file))
        if not os.path.exists(h_file):
            raise Exception('File "{}" does not exist'.format(h_file))
        return self.analyse_file(str(e_file), str(h_file))

    def SetMirror(self, mirr_type, ny, pos):
        if mirr_type <= 0:
            return
        if not mirr_type < 3:
            raise Exception("SetMirror: invalid mirror type!")
        from CSXCAD.Utilities import CheckNyDir

        self.set_mirror(int(mirr_type), CheckNyDir(ny), float(pos))

    def SetRadius(self, radius):
        self.set_radius(float(radius))

    def Write2HDF5(self, filename):
        return self.write2hdf5(str(filename))

    def SetVerboseLevel(self, level):
        self.set_verbose_level(int(level))

    def SetLegacyFormat(self, legacy=True):
        self.set_legacy_format(bool(legacy))


__all__ = ["_nf2ff"]
