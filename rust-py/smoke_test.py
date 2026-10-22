"""Smoke test for openems-py PyOpenEMS bindings."""
import openems_py

# Test constructor with kwargs
fdtd = openems_py.PyOpenEMS(NrTS=1000, EndCriteria=1e-4, MaxTime=1e-9)
print("Constructor: OK")

# Test setters
fdtd.set_number_of_time_steps(5000)
fdtd.set_end_criteria(1e-5)
fdtd.set_over_sampling(4)
fdtd.set_cell_constant_material(True)
fdtd.set_coord_system(0)
fdtd.set_time_step_method(3)
fdtd.set_time_step_factor(0.95)
fdtd.set_max_time(1e-8)
print("Setters: OK")

# Test excitations
fdtd.set_gauss_excite(1e9, 0.5e9)
fdtd.set_sinus_excite(2e9)
fdtd.set_dirac_excite(5e9)
fdtd.set_step_excite(3e9)
fdtd.set_custom_excite("sin(2*pi*1e9*t)", 1e9, 5e9)
print("Excitations: OK")

# Test boundary conditions
fdtd.set_boundary_cond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PEC', 'PEC'])
print("Boundary conditions: OK")

# Test constants
assert openems_py.BC_PEC == 0
assert openems_py.BC_PMC == 1
assert openems_py.BC_MUR == 2
assert openems_py.BC_PML == 3
print("Constants: OK")

# Test welcome screen
openems_py.PyOpenEMS.welcome_screen()
print("Welcome screen: OK")

# Test reset
fdtd.reset()
print("Reset: OK")

# Test run (stub - will call setup_fdtd and run_fdtd)
result = fdtd.run("/tmp/test_sim")
assert result == 0
print("Run: OK")

# Test set_abort
fdtd.set_abort(True)
fdtd.set_abort(False)
print("SetAbort: OK")

print("\nAll smoke tests passed!")
