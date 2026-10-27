#!/usr/bin/env bash
# RM-094: exercise the openEMS MATLAB C ABI from Octave.
#
#   ./run_octave_tests.sh
#
# Builds `libopenems_matlab` and the CSXCAD delivery crate, compiles the
# `openems_abi_oct` .oct shim (the MEX analogue), and runs
# `test_c_abi_octave`, which asserts the contract from the consumer side:
# exported symbol names, C argument/return types, column-major 1x6 and 6x1
# int32 arrays, the native `-1` sentinels, the native error identifiers and
# messages, CSX ownership transfer between the two delivery crates, and the
# double-free / use-after-free outcomes.
#
# Octave's `loadlibrary` is not implemented (Octave 11.1.0), so the .oct shim is
# the consumer; it links the cdylib exactly like a MEX file would.
set -euo pipefail

# `cargo` is not on PATH in a non-login shell on the validation host.
export PATH="$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LIB_DIR="$REPO_ROOT/target/debug"

if ! command -v octave >/dev/null 2>&1; then
  echo "octave is required for this check" >&2
  exit 1
fi
if ! command -v mkoctfile >/dev/null 2>&1; then
  echo "mkoctfile is required for this check (octave-dev)" >&2
  exit 1
fi

echo "== cargo: build the delivery libraries"
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p openems-matlab -p csxcad-matlab

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "== mkoctfile: compile the MEX-analogue shim"
mkoctfile -o "$BUILD_DIR/openems_abi_oct.oct" "$SCRIPT_DIR/openems_abi_oct.cc" \
  -L"$LIB_DIR" -lopenems_matlab -lcsxcad_matlab

echo "== octave: test_c_abi_octave"
# Run inside the scratch directory: the ABI contract test runs a real solver,
# which writes the native `et`/`ht` excitation waveforms into the CWD.
cd "$BUILD_DIR"
LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  octave --no-gui --quiet \
  --eval "addpath('$BUILD_DIR'); addpath('$SCRIPT_DIR'); test_c_abi_octave" 2>&1 \
  | grep -v -E "libEGL|MESA|ZINK"
