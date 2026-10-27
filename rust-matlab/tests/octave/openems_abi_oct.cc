// Octave/.oct consumer of the openEMS MATLAB C ABI (RM-094).
//
// This is the MEX analogue: an Octave dynamically linked function that calls
// `libopenems_matlab` exactly like a MEX shim would, so the assertions run
// against the real C boundary (exported symbols, C types, column-major Octave
// arrays, `char*` error strings) and not against Rust-side calls.
//
// Build and run through `run_octave_tests.sh`.

#include <octave/oct.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
// The ABI under test (openEMS/rust-matlab/src/lib.rs).
void *openems_new(void);
void openems_free(void *);
void openems_set_bc_type(void *, unsigned int, int);
int openems_get_bc_type(void *, unsigned int);
void openems_set_pml_size(void *, unsigned int, unsigned int);
unsigned int openems_get_pml_size(void *, unsigned int);
unsigned char openems_set_boundary_cond(void *, const int *, unsigned int);
unsigned char openems_get_boundary_cond(void *, int *, unsigned int);
unsigned char openems_setup_fdtd(void *);
unsigned char openems_run_fdtd(void *);
void *openems_get_csx(void *);
unsigned char openems_set_csx(void *, void *);
unsigned char openems_set_coord_system(void *, int);
char *openems_last_error_id(void);
char *openems_last_error_message(void);
int openems_last_error_code(void);
void openems_free_string(char *);

// The CSXCAD delivery crate. Its handles are `Box<csxcad_core::model::Structure>`
// pointers, the same type this ABI takes ownership of (native
// `openEMS::SetCSX(ContinuousStructure*)`, openEMS/openems.h:115-119).
void *csxcad_new(void);
void csxcad_free(void *);
}

namespace {

std::vector<std::string> results;

void check(bool ok, const std::string &name, const std::string &detail = "")
{
  std::string line = ok ? "ok: " : "FAIL: ";
  line += name;
  if (!ok && !detail.empty())
    line += ": " + detail;
  results.push_back(line);
}

std::string take(char *text)
{
  std::string value = text == nullptr ? std::string() : std::string(text);
  openems_free_string(text);
  return value;
}

std::string error_id() { return take(openems_last_error_id()); }
std::string error_message() { return take(openems_last_error_message()); }

const char *kDeletedMessage =
    "wrapped C++ object of type openEMS has been deleted";
const char *kCsxNotSet = "openEMS::SetupFDTD: Error: CSXCAD is not set!";

void check_no_error(const std::string &context)
{
  std::string id = error_id();
  std::string message = error_message();
  int code = openems_last_error_code();
  check(id.empty() && message.empty() && code == 0, "no error after " + context,
        "id='" + id + "' message='" + message + "' code=" + std::to_string(code));
}

} // namespace

DEFUN_DLD(openems_abi_oct, args, nargout,
          "Run the openEMS C ABI contract checks from Octave.")
{
  results.clear();

  // --- handles -------------------------------------------------------------
  void *solver = openems_new();
  check(solver != nullptr, "openems_new returns a handle");
  check(openems_get_csx(solver) == nullptr, "a fresh solver owns no geometry");
  check_no_error("openems_new");

  // --- [1x6] row array, native face order xmin..zmax ------------------------
  // openEMS/matlab/SetBoundaryCond.m:55-61 (BC(1..6) -> xmin..zmax).
  // The ABI is zero-based: element 0 is xmin.
  const int32_t faces[6] = {1, 1, 0, 0, 2, 3};
  int32NDArray row(dim_vector(1, 6));
  for (octave_idx_type i = 0; i < 6; ++i)
    row(i) = faces[i];

  check(openems_set_boundary_cond(solver, reinterpret_cast<const int *> (row.data ()), 6) == 1,
        "set_boundary_cond accepts a 1x6 int32 row");
  check_no_error("set_boundary_cond(1x6)");
  for (unsigned int face = 0; face < 6; ++face) {
    check(openems_get_bc_type(solver, face) == faces[face],
          "row entry " + std::to_string(face) + " lands on face " +
              std::to_string(face),
          "got " + std::to_string(openems_get_bc_type(solver, face)));
  }

  // --- [6x1] column array gives the identical result -----------------------
  // Octave/MATLAB store both shapes as the same six contiguous elements
  // (column-major), so the C boundary must not care about the shape.
  int32NDArray column(dim_vector(6, 1));
  for (octave_idx_type i = 0; i < 6; ++i)
    column(i) = faces[i];
  check(openems_set_boundary_cond(solver, reinterpret_cast<const int *> (column.data ()), 6) == 1,
        "set_boundary_cond accepts a 6x1 int32 column");
  for (unsigned int face = 0; face < 6; ++face) {
    check(openems_get_bc_type(solver, face) == faces[face],
          "column entry " + std::to_string(face) + " lands on face " +
              std::to_string(face));
  }

  // --- read back exactly six values, nothing past the end ------------------
  int32NDArray out(dim_vector(1, 8));
  out.fill(777);
  check(openems_get_boundary_cond(solver, reinterpret_cast<int *> (out.fortran_vec ()), 6) == 1,
        "get_boundary_cond writes a 1x6 int32 array");
  for (octave_idx_type i = 0; i < 6; ++i)
    check(static_cast<int32_t>(out(i)) == faces[i],
          "read back entry " + std::to_string(i));
  check(static_cast<int32_t>(out(6)) == 777 &&
            static_cast<int32_t>(out(7)) == 777,
        "get_boundary_cond leaves the buffer tail untouched");

  // --- wrong length is the native M-file error -----------------------------
  int32NDArray short_array(dim_vector(1, 5));
  short_array.fill(1);
  check(openems_set_boundary_cond(solver, reinterpret_cast<const int *> (short_array.data ()), 5) == 0,
        "set_boundary_cond rejects five entries");
  check(error_id() == "openEMS:SetBoundaryCond",
        "wrong-length identifier is openEMS:SetBoundaryCond", error_id());
  check(error_message() == "wrong number of boundary conditions",
        "wrong-length message", error_message());

  // --- PML faces -----------------------------------------------------------
  // openEMS/openems.cpp:836-838: a non-PML face reports -1.
  check(static_cast<int>(openems_get_pml_size(solver, 2)) == -1,
        "a PEC face reports the native -1 PML size");
  // openEMS/openems.cpp:824-830: Set_BC_PML also forces the face type to PML,
  // and openEMS/openems.cpp:130 gives every face the default size 8.
  openems_set_pml_size(solver, 4, 10);
  check(openems_get_bc_type(solver, 4) == 3,
        "set_pml_size forces the face type to PML");
  check(openems_get_pml_size(solver, 4) == 10,
        "set_pml_size stores the requested thickness");

  // --- coordinate system ---------------------------------------------------
  check(openems_set_coord_system(solver, 2) == 0,
        "set_coord_system rejects the native invalid value");
  check(error_message() == "Invalid coordinate system",
        "invalid coordinate-system message", error_message());
  check(openems_set_coord_system(solver, 1) == 1,
        "set_coord_system accepts cylindrical");

  // --- CSX ownership across the two delivery crates ------------------------
  void *csx_a = csxcad_new();
  void *csx_b = csxcad_new();
  check(csx_a != nullptr && csx_b != nullptr, "csxcad_new returns handles");
  check(openems_set_csx(solver, csx_a) == 1, "set_csx transfers ownership");
  check(openems_get_csx(solver) == csx_a, "get_csx returns the same pointer");
  check(openems_set_csx(solver, csx_b) == 1, "set_csx replaces the geometry");
  check(openems_get_csx(solver) == csx_b, "get_csx follows the replacement");
  // Native SetCSX deletes the stored pointer before assigning the new one
  // (openEMS/openems.cpp:1165-1166); the ABI keeps a single owner instead of
  // double deleting.
  check(openems_set_csx(solver, csx_b) == 1,
        "setting the same structure twice keeps one owner");
  check(openems_get_csx(solver) == csx_b, "the owner is unchanged");
  check(openems_set_csx(solver, nullptr) == 1, "set_csx(NULL) clears");
  check(openems_get_csx(solver) == nullptr, "get_csx reports no geometry");

  // --- setup without geometry is the native code 3 -------------------------
  check(openems_setup_fdtd(solver) == 0, "setup without geometry fails");
  check(openems_last_error_code() == 3, "setup failure code is 3",
        std::to_string(openems_last_error_code()));
  check(error_message() == kCsxNotSet, "setup failure message", error_message());

  // --- run before setup ----------------------------------------------------
  check(openems_run_fdtd(solver) == 0, "run before setup fails");
  check_no_error("run before setup");

  openems_free(solver);

  // --- double free and use after free --------------------------------------
  openems_free(solver);
  check(openems_get_bc_type(solver, 0) == -1,
        "a freed handle reports the native -1");
  check(error_id().empty(), "the deleted-object error has no identifier");
  check(error_message() == kDeletedMessage, "deleted-object message",
        error_message());

  // --- null handle ---------------------------------------------------------
  check(openems_get_bc_type(nullptr, 0) == -1, "a null handle reports -1");
  check(error_message() == kDeletedMessage, "null-handle message",
        error_message());
  openems_free(nullptr);

  Cell cell(dim_vector(static_cast<octave_idx_type>(results.size()), 1));
  for (std::size_t i = 0; i < results.size(); ++i)
    cell(i) = octave_value(results[i]);

  octave_value_list retval;
  retval(0) = cell;
  return retval;
}
