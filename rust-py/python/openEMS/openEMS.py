# -*- coding: utf-8 -*-
"""Native `openEMS` solver class backed by the Rust core.

The PyO3 class in `openEMS._openems` owns the FDTD engine and every solver
setting. This module provides the native Python controller around it: the
simulation-path/cleanup handling, the command-line option mapping, the port
factories (forwarding to `openEMS.ports`) and the near-field/far-field box,
exactly like the original Cython class.
"""

from __future__ import absolute_import

import glob
import os
from pathlib import Path

import numpy as np

try:
    from openEMS._openems import PyOpenEMS as _RustOpenEMS
except ImportError:  # extension built as a standalone module
    from openems_py import PyOpenEMS as _RustOpenEMS

from CSXCAD.Utilities import GetMultiDirs

# `openEMS.ports` and `openEMS.nf2ff` import the solver package themselves; they
# are imported lazily where they are needed to keep `import openEMS` acyclic.


#: Files that `Run(cleanup=True)` removes from the simulation directory.
_ALWAYS_DELETE = [
    "et",
    "ht",
    "port_ut*",
    "port_it*",
    "nf2ff*.h5",
    "*.vtp",
    "*.vtk",
    "*.vtr",
    "*.pvd",
    "openEMS_run_stats.txt",
    "openEMS_stats.txt",
    "debugCSX.xml",
]


class openEMS(_RustOpenEMS):
    """openEMS FDTD solver (native class and method names)."""

    # -- construction ------------------------------------------------------
    def __init__(self, *args, **kwargs):
        # every solver setting is already applied by the Rust constructor
        self.__CSX = None

    # -- solver settings ---------------------------------------------------
    def SetNumberOfTimeSteps(self, val):
        return self.set_number_of_time_steps(int(val))

    def SetEndCriteria(self, val):
        return self.set_end_criteria(float(val))

    def SetOverSampling(self, val):
        return self.set_over_sampling(int(val))

    def SetCellConstantMaterial(self, val):
        return self.set_cell_constant_material(bool(val))

    def SetCoordSystem(self, val):
        return self.set_coord_system(int(val))

    def SetMultiGrid(self, radii):
        return self.set_multi_grid([float(value) for value in radii])

    def SetCylinderCoords(self, val=True):
        return self.set_cylinder_coords(bool(val))

    def SetTimeStepMethod(self, val):
        return self.set_time_step_method(int(val))

    def SetTimeStep(self, val):
        return self.set_time_step(float(val))

    def SetTimeStepFactor(self, val):
        return self.set_time_step_factor(float(val))

    def SetMaxTime(self, val):
        return self.set_max_time(float(val))

    def SetGaussExcite(self, f0, fc):
        return self.set_gauss_excite(float(f0), float(fc))

    def SetSinusExcite(self, f0):
        return self.set_sinus_excite(float(f0))

    def SetDiracExcite(self, fmax):
        return self.set_dirac_excite(float(fmax))

    def SetStepExcite(self, fmax):
        return self.set_step_excite(float(fmax))

    def SetCustomExcite(self, function, f0, fmax):
        return self.set_custom_excite(str(function), float(f0), float(fmax))

    def SetBoundaryCond(self, bc):
        return self.set_boundary_cond(list(bc))

    def SetNumberOfThreads(self, n):
        return self.set_number_of_threads(int(n))

    def SetAbort(self, val):
        return self.set_abort(bool(val))

    def SetEnableDumps(self, val):
        return self.set_enable_dumps(bool(val))

    # -- CSXCAD structure --------------------------------------------------
    def SetCSX(self, CSX):
        self.__CSX = CSX
        self.set_csx(CSX)

    def GetCSX(self):
        return self.__CSX

    def _push_csx_xml(self):
        """Hand the current CSXCAD geometry to the Rust engine."""
        if self.__CSX is not None:
            self.set_csx_xml(self.__CSX._live().to_xml_string())

    # -- ports -------------------------------------------------------------
    def AddLumpedPort(self, port_nr, R, start, stop, p_dir, excite=0, **kw):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddLumpedPort: CSX is not set!")
        port = ports.LumpedPort(
            self.__CSX, port_nr, R, start, stop, p_dir, excite, **kw
        )
        edges2grid = kw.get("edges2grid", None)
        if edges2grid is not None:
            grid = self.__CSX.GetGrid()
            for n in GetMultiDirs(edges2grid):
                grid.AddLine(n, start[n])
                if start[n] != stop[n]:
                    grid.AddLine(n, stop[n])
        return port

    def AddCurvePort(self, port_nr, R, start, stop, excite=0, **kw):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddCurvePort: CSX is not set!")
        return ports.CurvePort(self.__CSX, port_nr, R, start, stop, excite, **kw)

    def AddWaveGuidePort(
        self,
        port_nr,
        start,
        stop,
        p_dir,
        E_func=None,
        H_func=None,
        kc=0.0,
        excite=0,
        excite_type=0,
        E_file=None,
        H_file=None,
        **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddWaveGuidePort: CSX is not set!")
        return ports.WaveguidePort(
            self.__CSX,
            port_nr,
            start,
            stop,
            p_dir,
            E_func,
            H_func,
            kc,
            excite,
            excite_type,
            E_file,
            H_file,
            **kw
        )

    def AddRectWaveGuidePort(
        self, port_nr, start, stop, p_dir, a, b, mode_name, excite=0, **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddRectWaveGuidePort: CSX is not set!")
        return ports.RectWaveguidePort(
            self.__CSX, port_nr, start, stop, p_dir, a, b, mode_name, excite, **kw
        )

    def AddCircWaveGuidePort(
        self, port_nr, start, stop, exc_dir, radius, mode_name, pol_ang=0, excite=0, **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddCircWaveGuidePort: CSX is not set!")
        return ports.CircWaveguidePort(
            self.__CSX,
            port_nr,
            start,
            stop,
            exc_dir,
            radius,
            mode_name,
            pol_ang,
            excite,
            **kw
        )

    def AddCoaxialPort(
        self,
        port_nr,
        pec_prop,
        mat_prop,
        start,
        stop,
        prop_dir,
        r_i,
        r_o,
        r_os,
        excite=0,
        **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddCoaxialPort: CSX is not set!")
        return ports.CoaxialPort(
            self.__CSX,
            port_nr,
            pec_prop,
            mat_prop,
            start,
            stop,
            prop_dir,
            r_i,
            r_o,
            r_os,
            excite,
            **kw
        )

    def AddMSLPort(
        self, port_nr, metal_prop, start, stop, prop_dir, exc_dir, excite=0, **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddMSLPort: CSX is not set!")
        return ports.MSLPort(
            self.__CSX,
            port_nr,
            metal_prop,
            start,
            stop,
            prop_dir,
            exc_dir,
            excite,
            **kw
        )

    def AddStripLinePort(
        self, port_nr, metal_prop, start, stop, prop_dir, exc_dir, height, excite=0, **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddStripLinePort: CSX is not set!")
        return ports.StripLinePort(
            self.__CSX,
            port_nr,
            metal_prop,
            start,
            stop,
            prop_dir,
            exc_dir,
            height,
            excite,
            **kw
        )

    def AddCPWPort(
        self, port_nr, metal_prop, start, stop, prop_dir, exc_dir, gap_width, excite=0, **kw
    ):
        from openEMS import ports

        if self.__CSX is None:
            raise Exception("AddCPWPort: CSX is not set!")
        return ports.CPWPort(
            self.__CSX,
            port_nr,
            metal_prop,
            start,
            stop,
            prop_dir,
            exc_dir,
            gap_width,
            excite,
            **kw
        )

    # -- near-field to far-field ------------------------------------------
    def CreateNF2FFBox(self, name="nf2ff", start=None, stop=None, **kw):
        from openEMS import nf2ff as nf2ff_module

        if self.__CSX is None:
            raise Exception("CreateNF2FFBox: CSX is not set!")
        directions = [True] * 6
        mirror = [0] * 6
        bc_size = [0] * 6
        for n in range(6):
            bc_type = self.get_bc_type(n)
            if bc_type == 0:
                directions[n] = False
                mirror[n] = 1  # PEC mirror
            elif bc_type == 1:
                directions[n] = False
                mirror[n] = 2  # PMC mirror
            elif bc_type == 2:
                bc_size[n] = 2
            elif bc_type == 3:
                bc_size[n] = self.get_pml_size(n) + 1

        if start is None or stop is None:
            grid = self.__CSX.GetGrid()
            if not grid.IsValid():
                raise Exception("Error::CreateNF2FFBox: Grid is invalid")
            start = np.zeros(3)
            stop = np.zeros(3)
            for n in range(3):
                lines = grid.GetLines(n)
                if not len(lines) > (bc_size[2 * n] + bc_size[2 * n + 1]):
                    raise Exception(
                        "Error::CreateNF2FFBox: not enough lines in some direction"
                    )
                start[n] = lines[bc_size[2 * n]]
                stop[n] = lines[-1 * bc_size[2 * n + 1] - 1]
        directions = kw.pop("directions", directions)
        mirror = kw.pop("mirror", mirror)
        return nf2ff_module.nf2ff(
            self.__CSX, name, start, stop, directions=directions, mirror=mirror, **kw
        )

    # -- grid helpers ------------------------------------------------------
    def AddEdges2Grid(self, dirs, primitives=None, properties=None, **kw):
        if self.__CSX is None:
            raise Exception("AddEdges2Grid: CSX is not set!")
        structure = self.__CSX._live()
        for direction in GetMultiDirs(dirs):
            structure.insert_edges_to_grid(direction)

    # -- run ---------------------------------------------------------------
    def _SetLibraryArguments(self, arguments):
        """Apply the command-line style keyword options to the engine."""
        integer_options = ("verbose", "numthreads")
        for key, val in arguments.items():
            name = key.replace("-", "_").lower()
            if name == "numthreads":
                self.set_number_of_threads(int(val))
            elif name == "verbose":
                self.set_verbose_level(int(val))
            elif name == "disable_dumps":
                self.set_enable_dumps(not bool(val))
            elif name == "dump_statistics":
                self.set_dump_statistics(bool(val))
            elif name == "exact_endcriteria":
                self.set_exact_end_criteria(bool(val))
            elif name == "no_simulation":
                self.set_no_simulation(bool(val))
            elif name == "debug_csx":
                if val:
                    self.debug_csx()
            elif name == "debug_material":
                if val:
                    self.debug_material()
            elif name == "debug_pec":
                if val:
                    self.debug_pec()
            elif name == "debug_operator":
                if val:
                    self.debug_operator()
            elif name == "debug_boxes":
                if val:
                    self.debug_box()
            else:
                raise Exception(
                    "_SetLibraryArguments: unknown option {!r}".format(key)
                )

    def _cleanup_sim_path(self, sim_path, verbose=False):
        for pattern in _ALWAYS_DELETE:
            for name in glob.glob(os.path.join(sim_path, pattern)):
                if os.path.isfile(name):
                    if verbose:
                        print("cleanup: removing {}".format(name))
                    os.remove(name)
        # field dumps are only removed when they carry the openEMS fingerprint
        import h5py

        for name in glob.glob(os.path.join(sim_path, "*.h5")):
            if not os.path.isfile(name):
                continue
            try:
                with h5py.File(name, "r") as handle:
                    is_dump = "openEMS_HDF5_version" in handle.attrs or "nf2ff" in handle
            except OSError:
                continue
            if is_dump:
                if verbose:
                    print("cleanup: removing {}".format(name))
                os.remove(name)

    def Run(self, sim_path, cleanup=False, setup_only=False, **kw):
        sim_path = os.path.abspath(sim_path)
        if cleanup and os.path.exists(sim_path):
            self._cleanup_sim_path(sim_path, verbose=kw.get("verbose"))
        if not os.path.exists(sim_path):
            os.mkdir(sim_path)
        os.chdir(sim_path)

        self._SetLibraryArguments(kw)
        if setup_only:
            self.set_no_simulation(True)

        if Path(os.getcwd()).resolve() != Path(sim_path).resolve():
            raise RuntimeError(
                "Current working directory is different from `sim_path`. If you "
                "encounter this error, please report it to the developers, because "
                "it should never happen in normal conditions, it is not your fault. "
            )

        self.WelcomeScreen()
        self._push_csx_xml()
        error_code = self.run()
        if error_code != 0:
            print("Run: Setup failed, error code: {}".format(error_code))
        return error_code

    # -- XML ---------------------------------------------------------------
    _XML_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'

    def _document(self):
        """The native `openEMS` input document: `<FDTD>` plus the CSX."""
        csx = ""
        if self.__CSX is not None:
            csx = self.__CSX._live().to_xml_string()
            if csx.startswith("<?xml"):
                csx = csx[csx.index("?>") + 2 :]
        return (
            self._XML_HEADER
            + "<openEMS>"
            + self.fdtd_xml()
            + csx
            + "</openEMS>"
        )

    def Write2XML(self, file):
        document = self._document()
        with open(str(file), "w", encoding="utf-8") as handle:
            handle.write(document)
        return True

    def ReadFromXML(self, file):
        with open(str(file), encoding="utf-8") as handle:
            document = handle.read()
        self.apply_fdtd_xml(document)
        self.set_csx_xml(document)

        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        message = csx.ReadFromXML(str(file))
        if message != "":
            print("ReadFromXML: {}".format(message))
            return False
        self.__CSX = csx
        return True

    # -- misc --------------------------------------------------------------
    @staticmethod
    def WelcomeScreen():
        return _RustOpenEMS.welcome_screen()

    def __repr__(self):
        return "<openEMS>"
