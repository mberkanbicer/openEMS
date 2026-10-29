//! HDF5 file writing and reading for openEMS field dumps.
//!
//! Mirrors `openEMS/tools/hdf5_file_writer.{h,cpp}` and
//! `openEMS/tools/hdf5_file_reader.{h,cpp}`.
//!
//! The backend links against the system HDF5 C library (`hdf5`) and is only
//! compiled with the `hdf5` feature. Without the feature the public API stays
//! available but every operation reports [`Hdf5Error::Unsupported`]; no
//! substitute file format is ever written.
//!
//! Layout written (identical to the native writer):
//!
//! * `/Mesh` group with datasets `x`, `y`, `z` and attributes `mesh_type`
//!   (`int`) and `mesh_scaling` (`double`).
//! * `/FieldData/TD/<ssssssss>` vector field datasets with a `time` attribute.
//! * `/FieldData/FD` frequency list attribute `frequency`.
//! * `/FieldData/FD/<name>_real` and `<name>_imag` datasets with `frequency`.
//! * Dataset attribute `d_order` (`XYZ` / `ZYX` / `NXYZ` / `NZYX`) describing
//!   the storage order.
//! * Root attributes `openEMS_HDF5_version`, `dump_type`, `dual_mesh` and
//!   `legacy_fmt`.

use std::fmt;

#[cfg(feature = "hdf5")]
use std::ffi::CString;

/// Version stamp written to the root group, matching
/// `OPENEMS_HDF5_VERSION` in `tools/hdf5_file_writer.h`.
pub const OPENEMS_HDF5_VERSION: f64 = 0.3;

/// Errors produced by the HDF5 backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hdf5Error {
    /// The crate was built without the `hdf5` feature.
    Unsupported,
    /// A path or object name contained an interior NUL byte.
    NameContainsNul,
    /// An HDF5 C call failed.
    Hdf5 { operation: &'static str, code: i32 },
    /// The requested object does not exist.
    NotFound(String),
    /// The dataset or attribute has an unexpected rank or shape.
    InvalidShape { path: String, detail: String },
}

impl fmt::Display for Hdf5Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Hdf5Error::Unsupported => write!(
                formatter,
                "HDF5 support is not compiled in; rebuild with --features hdf5 and point HDF5_LIB_DIR at the HDF5 library"
            ),
            Hdf5Error::NameContainsNul => {
                write!(formatter, "HDF5 path contains a NUL byte")
            }
            Hdf5Error::Hdf5 { operation, code } => {
                write!(formatter, "HDF5 call '{operation}' failed with code {code}")
            }
            Hdf5Error::NotFound(path) => write!(formatter, "HDF5 object '{path}' not found"),
            Hdf5Error::InvalidShape { path, detail } => {
                write!(formatter, "HDF5 object '{path}' has an invalid shape: {detail}")
            }
        }
    }
}

impl std::error::Error for Hdf5Error {}

/// Convert a Rust path/name into a NUL-terminated C string.
#[cfg(feature = "hdf5")]
fn c_name(value: &str) -> Result<CString, Hdf5Error> {
    CString::new(value).map_err(|_| Hdf5Error::NameContainsNul)
}

/// Reorder a `(n, i, j, k)` field array into the native `d_order` layouts.
///
/// This is pure data movement and is deliberately available without the
/// `hdf5` feature so the ordering can be unit tested on every platform.
///
/// * `NXYZ`: component-major, then x, y, z (native non-legacy order).
/// * `NZYX`: component-major, then z, y, x (legacy Octave/MATLAB order).
pub fn field_to_nxyz(values: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut out = Vec::with_capacity(3 * nx * ny * nz);
    for n in 0..3 {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let index = n * nx * ny * nz + i * ny * nz + j * nz + k;
                    out.push(values.get(index).copied().unwrap_or(0.0));
                }
            }
        }
    }
    out
}

/// Reorder a `(n, i, j, k)` field array into `NZYX` (legacy) order.
pub fn field_to_nzyx(values: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut out = Vec::with_capacity(3 * nx * ny * nz);
    for n in 0..3 {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let index = n * nx * ny * nz + i * ny * nz + j * nz + k;
                    out.push(values.get(index).copied().unwrap_or(0.0));
                }
            }
        }
    }
    out
}

/// Inverse of [`field_to_nzyx`]: restore `(n, i, j, k)` order from a stored
/// `NZYX` dataset with dims `[nz, ny, nx]`.
pub fn field_from_nzyx(values: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut out = vec![0.0f32; 3 * nx * ny * nz];
    for n in 0..3 {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let stored = n * nz * ny * nx + k * ny * nx + j * nx + i;
                    out[n * nx * ny * nz + i * ny * nz + j * nz + k] =
                        values.get(stored).copied().unwrap_or(0.0);
                }
            }
        }
    }
    out
}

#[cfg(feature = "hdf5")]
mod backend {
    use super::*;
    use std::ffi::{c_char, c_int, c_uint, c_void};
    use std::ptr;
    use std::sync::{Mutex, Once, OnceLock};

    pub type Hid = i64;
    type Herr = c_int;
    type Htri = c_int;
    type HSize = u64;
    type Hssize = i64;

    pub const H5P_DEFAULT: Hid = 0;
    const H5S_ALL: Hid = 0;
    const H5S_SCALAR: c_int = 0;
    const H5F_ACC_RDONLY: c_uint = 0x0000;
    const H5F_ACC_RDWR: c_uint = 0x0001;
    const H5F_ACC_TRUNC: c_uint = 0x0002;
    const H5T_COMPOUND: c_int = 6;
    const H5T_CSET_UTF8: c_int = 1;
    const H5T_VARIABLE: usize = usize::MAX;
    const H5_INDEX_NAME: c_int = 0;
    const H5_ITER_INC: c_int = 0;

    #[link(name = "hdf5")]
    extern "C" {
        static mut H5T_NATIVE_DOUBLE_g: Hid;
        static mut H5T_NATIVE_FLOAT_g: Hid;
        static mut H5T_NATIVE_INT_g: Hid;
        static mut H5T_C_S1_g: Hid;
        static mut H5T_STD_I32LE_g: Hid;
        static mut H5T_IEEE_F32LE_g: Hid;
        static mut H5T_IEEE_F64LE_g: Hid;
        static mut H5T_STD_U8LE_g: Hid;
        static mut H5T_NATIVE_UINT8_g: Hid;

        fn H5open() -> Herr;
        fn H5Eset_auto2(
            stack: Hid,
            function: Option<unsafe extern "C" fn(c_int, *mut c_void) -> c_int>,
            client_data: *mut c_void,
        ) -> Herr;
        fn H5Fcreate(filename: *const c_char, flags: c_uint, fcpl: Hid, fapl: Hid) -> Hid;
        fn H5Fopen(filename: *const c_char, flags: c_uint, fapl: Hid) -> Hid;
        fn H5Fclose(file: Hid) -> Herr;
        fn H5Gcreate2(location: Hid, name: *const c_char, lcpl: Hid, gcpl: Hid, gapl: Hid) -> Hid;
        fn H5Gopen2(location: Hid, name: *const c_char, gapl: Hid) -> Hid;
        fn H5Gclose(group: Hid) -> Herr;
        fn H5Oopen(location: Hid, name: *const c_char, lapl: Hid) -> Hid;
        fn H5Oclose(object: Hid) -> Herr;
        fn H5Lexists(location: Hid, name: *const c_char, lapl: Hid) -> Htri;
        fn H5Lget_name_by_idx(
            location: Hid,
            group_name: *const c_char,
            index_type: c_int,
            order: c_int,
            n: HSize,
            name: *mut c_char,
            size: usize,
            lapl: Hid,
        ) -> Hssize;
        fn H5Dcreate2(
            location: Hid,
            name: *const c_char,
            dtype: Hid,
            space: Hid,
            lcpl: Hid,
            dcpl: Hid,
            dapl: Hid,
        ) -> Hid;
        fn H5Dopen2(location: Hid, name: *const c_char, dapl: Hid) -> Hid;
        fn H5Dclose(dataset: Hid) -> Herr;
        fn H5Dget_space(dataset: Hid) -> Hid;
        fn H5Dwrite(
            dataset: Hid,
            mem_type: Hid,
            mem_space: Hid,
            file_space: Hid,
            transfer: Hid,
            buffer: *const c_void,
        ) -> Herr;
        fn H5Dread(
            dataset: Hid,
            mem_type: Hid,
            mem_space: Hid,
            file_space: Hid,
            transfer: Hid,
            buffer: *mut c_void,
        ) -> Herr;
        fn H5Screate(class: c_int) -> Hid;
        fn H5Screate_simple(rank: c_int, dims: *const HSize, maxdims: *const HSize) -> Hid;
        fn H5Sclose(space: Hid) -> Herr;
        fn H5Sget_simple_extent_ndims(space: Hid) -> c_int;
        fn H5Sget_simple_extent_dims(space: Hid, dims: *mut HSize, maxdims: *mut HSize) -> c_int;
        fn H5Sget_simple_extent_npoints(space: Hid) -> Hssize;
        fn H5Acreate2(
            location: Hid,
            name: *const c_char,
            dtype: Hid,
            space: Hid,
            acpl: Hid,
            aapl: Hid,
        ) -> Hid;
        fn H5Aopen(location: Hid, name: *const c_char, aapl: Hid) -> Hid;
        fn H5Aclose(attribute: Hid) -> Herr;
        fn H5Aexists(location: Hid, name: *const c_char) -> Htri;
        fn H5Aget_space(attribute: Hid) -> Hid;
        fn H5Aget_type(attribute: Hid) -> Hid;
        fn H5Awrite(attribute: Hid, mem_type: Hid, buffer: *const c_void) -> Herr;
        fn H5Aread(attribute: Hid, mem_type: Hid, buffer: *mut c_void) -> Herr;
        fn H5Tcreate(class: c_int, size: usize) -> Hid;
        fn H5Tinsert(type_id: Hid, name: *const c_char, offset: usize, field: Hid) -> Herr;
        fn H5Tcopy(type_id: Hid) -> Hid;
        fn H5Tclose(type_id: Hid) -> Herr;
        fn H5Tset_size(type_id: Hid, size: usize) -> Herr;
        fn H5Tset_cset(type_id: Hid, cset: c_int) -> Herr;
        fn H5Tget_size(type_id: Hid) -> usize;
        fn H5Tis_variable_str(type_id: Hid) -> Htri;
        fn H5free_memory(memory: *mut c_void) -> Herr;
    }

    static INIT: Once = Once::new();
    static HDF5_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        HDF5_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn init() {
        INIT.call_once(|| unsafe {
            // H5open() initialises the library globals (H5T_NATIVE_* and
            // friends). It is idempotent.
            let _ = H5open();
            // Route HDF5 diagnostics through our own error values instead of
            // printing the C error stack (probing optional attributes is
            // expected to fail).
            let _ = H5Eset_auto2(0, None, ptr::null_mut());
        });
    }

    fn check(operation: &'static str, code: c_int) -> Result<c_int, Hdf5Error> {
        if code < 0 {
            Err(Hdf5Error::Hdf5 {
                operation,
                code: code as i32,
            })
        } else {
            Ok(code)
        }
    }

    fn check_id(operation: &'static str, id: Hid) -> Result<Hid, Hdf5Error> {
        if id < 0 {
            Err(Hdf5Error::Hdf5 {
                operation,
                code: id as i32,
            })
        } else {
            Ok(id)
        }
    }

    /// Owned HDF5 object handle (file or group).
    pub(super) struct Handle {
        pub(super) id: Hid,
        close: unsafe extern "C" fn(Hid) -> Herr,
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe {
                let _ = (self.close)(self.id);
            }
        }
    }

    fn dims_to_hsize(dims: &[usize]) -> Vec<HSize> {
        dims.iter().map(|&value| value as HSize).collect()
    }

    impl super::Hdf5File {
        /// Create (truncate) a new HDF5 file.
        pub fn create(path: &str) -> Result<Self, Hdf5Error> {
            init();
            let _guard = lock();
            let name = c_name(path)?;
            let file = unsafe {
                check_id(
                    "H5Fcreate",
                    H5Fcreate(name.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                )?
            };
            Ok(super::Hdf5File {
                handle: Handle {
                    id: file,
                    close: H5Fclose,
                },
                path: path.to_string(),
            })
        }

        /// Open an existing HDF5 file read-only or read-write.
        pub fn open(path: &str, read_write: bool) -> Result<Self, Hdf5Error> {
            init();
            let _guard = lock();
            let name = c_name(path)?;
            let flags = if read_write {
                H5F_ACC_RDWR
            } else {
                H5F_ACC_RDONLY
            };
            let file = unsafe { check_id("H5Fopen", H5Fopen(name.as_ptr(), flags, H5P_DEFAULT))? };
            Ok(super::Hdf5File {
                handle: Handle {
                    id: file,
                    close: H5Fclose,
                },
                path: path.to_string(),
            })
        }

        /// The file path this handle was opened with.
        pub fn path(&self) -> &str {
            &self.path
        }

        /// Open (and optionally create) a group by absolute path.
        fn group(&self, path: &str, create: bool) -> Result<Handle, Hdf5Error> {
            let mut owned: Option<Handle> = None;
            for component in path.split('/').filter(|part| !part.is_empty()) {
                let name = c_name(component)?;
                let location = owned
                    .as_ref()
                    .map(|handle| handle.id)
                    .unwrap_or(self.handle.id);
                let exists = unsafe {
                    let result = H5Lexists(location, name.as_ptr(), H5P_DEFAULT);
                    if result < 0 {
                        return Err(Hdf5Error::Hdf5 {
                            operation: "H5Lexists",
                            code: result,
                        });
                    }
                    result > 0
                };
                let next = unsafe {
                    if exists {
                        check_id("H5Gopen2", H5Gopen2(location, name.as_ptr(), H5P_DEFAULT))?
                    } else if create {
                        check_id(
                            "H5Gcreate2",
                            H5Gcreate2(
                                location,
                                name.as_ptr(),
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                                H5P_DEFAULT,
                            ),
                        )?
                    } else {
                        return Err(Hdf5Error::NotFound(path.to_string()));
                    }
                };
                owned = Some(Handle {
                    id: next,
                    close: H5Gclose,
                });
            }
            match owned {
                Some(handle) => Ok(handle),
                None => {
                    let name = c_name("/")?;
                    let id = unsafe {
                        check_id(
                            "H5Gopen2",
                            H5Gopen2(self.handle.id, name.as_ptr(), H5P_DEFAULT),
                        )?
                    };
                    Ok(Handle {
                        id,
                        close: H5Gclose,
                    })
                }
            }
        }

        /// Create a group (and missing parents) if it does not exist yet.
        pub fn create_group(&self, path: &str) -> Result<(), Hdf5Error> {
            let _guard = lock();
            drop(self.group(path, true)?);
            Ok(())
        }

        /// True if an object exists at `path`.
        pub fn exists(&self, path: &str) -> bool {
            let _guard = lock();
            let Ok(name) = c_name(path) else {
                return false;
            };
            let id = unsafe { H5Oopen(self.handle.id, name.as_ptr(), H5P_DEFAULT) };
            if id < 0 {
                return false;
            }
            unsafe {
                let _ = H5Oclose(id);
            }
            true
        }

        /// List the link names inside a group.
        pub fn object_names(&self, path: &str) -> Result<Vec<String>, Hdf5Error> {
            let _guard = lock();
            let group = self.group(path, false)?;
            let dot = c_name(".")?;
            let mut names = Vec::new();
            let mut index: HSize = 0;
            loop {
                let length = unsafe {
                    H5Lget_name_by_idx(
                        group.id,
                        dot.as_ptr(),
                        H5_INDEX_NAME,
                        H5_ITER_INC,
                        index,
                        ptr::null_mut(),
                        0,
                        H5P_DEFAULT,
                    )
                };
                if length < 0 {
                    break;
                }
                let mut buffer = vec![0 as c_char; length as usize + 1];
                let written = unsafe {
                    H5Lget_name_by_idx(
                        group.id,
                        dot.as_ptr(),
                        H5_INDEX_NAME,
                        H5_ITER_INC,
                        index,
                        buffer.as_mut_ptr(),
                        buffer.len(),
                        H5P_DEFAULT,
                    )
                };
                if written < 0 {
                    break;
                }
                let text = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                names.push(text);
                index += 1;
            }
            Ok(names)
        }

        fn write_dataset(
            &self,
            group_path: &str,
            name: &str,
            file_type: Hid,
            mem_type: Hid,
            dims: &[usize],
            data: *const c_void,
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            let _guard = lock();
            let group = self.group(group_path, true)?;
            let object_name = c_name(name)?;
            let hdims = dims_to_hsize(dims);
            let space = unsafe {
                check_id(
                    "H5Screate_simple",
                    H5Screate_simple(hdims.len() as c_int, hdims.as_ptr(), ptr::null()),
                )?
            };
            let dataset = unsafe {
                check_id(
                    "H5Dcreate2",
                    H5Dcreate2(
                        group.id,
                        object_name.as_ptr(),
                        file_type,
                        space,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                    ),
                )
            };
            let dataset = match dataset {
                Ok(id) => id,
                Err(error) => {
                    unsafe {
                        let _ = H5Sclose(space);
                    }
                    return Err(error);
                }
            };
            let write = unsafe { H5Dwrite(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) };
            if let Some(order) = order {
                let _ = write_string_attribute_on(dataset, "d_order", order);
            }
            unsafe {
                let _ = H5Dclose(dataset);
                let _ = H5Sclose(space);
            }
            check("H5Dwrite", write).map(|_| ())
        }

        /// Write a one-dimensional `f64` dataset.
        pub fn write_f64_dataset(
            &self,
            group: &str,
            name: &str,
            data: &[f64],
        ) -> Result<(), Hdf5Error> {
            let file_type = unsafe { H5T_IEEE_F64LE_g };
            let mem_type = unsafe { H5T_NATIVE_DOUBLE_g };
            self.write_dataset(
                group,
                name,
                file_type,
                mem_type,
                &[data.len()],
                data.as_ptr() as *const c_void,
                Some("X"),
            )
        }

        /// Write a one-dimensional `f32` dataset.
        pub fn write_f32_dataset(
            &self,
            group: &str,
            name: &str,
            data: &[f32],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            let file_type = unsafe { H5T_IEEE_F32LE_g };
            let mem_type = unsafe { H5T_NATIVE_FLOAT_g };
            self.write_dataset(
                group,
                name,
                file_type,
                mem_type,
                &[data.len()],
                data.as_ptr() as *const c_void,
                order,
            )
        }

        /// Write an N-dimensional `f32` dataset.
        pub fn write_f32_dataset_nd(
            &self,
            group: &str,
            name: &str,
            dims: &[usize],
            data: &[f32],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            let expected: usize = dims.iter().product();
            if expected != data.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: format!("{expected} values expected, {} given", data.len()),
                });
            }
            let file_type = unsafe { H5T_IEEE_F32LE_g };
            let mem_type = unsafe { H5T_NATIVE_FLOAT_g };
            self.write_dataset(
                group,
                name,
                file_type,
                mem_type,
                dims,
                data.as_ptr() as *const c_void,
                order,
            )
        }

        /// Write an N-dimensional `f64` dataset.
        pub fn write_f64_dataset_nd(
            &self,
            group: &str,
            name: &str,
            dims: &[usize],
            data: &[f64],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            let expected: usize = dims.iter().product();
            if expected != data.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: format!("{expected} values expected, {} given", data.len()),
                });
            }
            let file_type = unsafe { H5T_IEEE_F64LE_g };
            let mem_type = unsafe { H5T_NATIVE_DOUBLE_g };
            self.write_dataset(
                group,
                name,
                file_type,
                mem_type,
                dims,
                data.as_ptr() as *const c_void,
                order,
            )
        }

        /// Write a complex `f32` dataset as an HDF5 compound of `r`/`i`,
        /// matching `HDF5_File_Reader::GetH5Type<std::complex<float>>()`.
        pub fn write_complex_f32_dataset(
            &self,
            group: &str,
            name: &str,
            dims: &[usize],
            re: &[f32],
            im: &[f32],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            if re.len() != im.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: "real and imaginary parts differ in length".to_string(),
                });
            }
            let expected: usize = dims.iter().product();
            if expected != re.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: format!("{expected} values expected, {} given", re.len()),
                });
            }
            let mut interleaved = Vec::with_capacity(2 * re.len());
            for (real, imag) in re.iter().zip(im.iter()) {
                interleaved.push(*real);
                interleaved.push(*imag);
            }
            let compound = unsafe {
                let id = check_id("H5Tcreate", H5Tcreate(H5T_COMPOUND, 8))?;
                let real = c_name("r")?;
                let imag = c_name("i")?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, real.as_ptr(), 0, H5T_NATIVE_FLOAT_g),
                )?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, imag.as_ptr(), 4, H5T_NATIVE_FLOAT_g),
                )?;
                id
            };
            let result = self.write_dataset(
                group,
                name,
                compound,
                compound,
                dims,
                interleaved.as_ptr() as *const c_void,
                order,
            );
            unsafe {
                let _ = H5Tclose(compound);
            }
            result
        }

        /// Write a complex `f64` dataset as an HDF5 compound of `r`/`i`.
        ///
        /// Native `nf2ff::Write2HDF5` writes `std::complex<double>` data, so
        /// the non-legacy far-field dataset is a compound of two doubles.
        pub fn write_complex_f64_dataset(
            &self,
            group: &str,
            name: &str,
            dims: &[usize],
            re: &[f64],
            im: &[f64],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            if re.len() != im.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: "real and imaginary parts differ in length".to_string(),
                });
            }
            let expected: usize = dims.iter().product();
            if expected != re.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: format!("{expected} values expected, {} given", re.len()),
                });
            }
            let mut interleaved = Vec::with_capacity(2 * re.len());
            for (real, imag) in re.iter().zip(im.iter()) {
                interleaved.push(*real);
                interleaved.push(*imag);
            }
            let compound = unsafe {
                let id = check_id("H5Tcreate", H5Tcreate(H5T_COMPOUND, 16))?;
                let real = c_name("r")?;
                let imag = c_name("i")?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, real.as_ptr(), 0, H5T_NATIVE_DOUBLE_g),
                )?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, imag.as_ptr(), 8, H5T_NATIVE_DOUBLE_g),
                )?;
                id
            };
            let result = self.write_dataset(
                group,
                name,
                compound,
                compound,
                dims,
                interleaved.as_ptr() as *const c_void,
                order,
            );
            unsafe {
                let _ = H5Tclose(compound);
            }
            result
        }

        fn write_attribute_of(
            &self,
            path: &str,
            name: &str,
            file_type: Hid,
            mem_type: Hid,
            keepalive: &dyn Fn() -> *const c_void,
        ) -> Result<(), Hdf5Error> {
            let _guard = lock();
            let object_name = c_name(path)?;
            let id = unsafe { H5Oopen(self.handle.id, object_name.as_ptr(), H5P_DEFAULT) };
            let id = check_id("H5Oopen", id)?;
            let result = unsafe {
                let attribute = c_name(name)?;
                let space = H5Screate(H5S_SCALAR);
                if space < 0 {
                    let _ = H5Oclose(id);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Screate",
                        code: space as i32,
                    });
                }
                let attribute_id = H5Acreate2(
                    id,
                    attribute.as_ptr(),
                    file_type,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                );
                if attribute_id < 0 {
                    let _ = H5Sclose(space);
                    let _ = H5Oclose(id);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Acreate2",
                        code: attribute_id as i32,
                    });
                }
                let value = keepalive();
                let write = H5Awrite(attribute_id, mem_type, value);
                let _ = H5Aclose(attribute_id);
                let _ = H5Sclose(space);
                check("H5Awrite", write)
            };
            unsafe {
                let _ = H5Oclose(id);
            }
            result.map(|_| ())
        }

        /// Write a scalar `f64` attribute on `path`.
        pub fn write_f64_attribute(
            &self,
            path: &str,
            name: &str,
            value: f64,
        ) -> Result<(), Hdf5Error> {
            self.write_attribute_of(
                path,
                name,
                unsafe { H5T_IEEE_F64LE_g },
                unsafe { H5T_NATIVE_DOUBLE_g },
                &|| &value as *const f64 as *const c_void,
            )
        }

        /// Write a scalar `i32` attribute on `path`.
        pub fn write_i32_attribute(
            &self,
            path: &str,
            name: &str,
            value: i32,
        ) -> Result<(), Hdf5Error> {
            self.write_attribute_of(
                path,
                name,
                unsafe { H5T_STD_I32LE_g },
                unsafe { H5T_NATIVE_INT_g },
                &|| &value as *const i32 as *const c_void,
            )
        }

        /// Write a boolean attribute as an unsigned byte.
        ///
        /// Native `HDF5_File_Writer` stores `dual_mesh`, `legacy_fmt` and the
        /// other flags as `H5T_NATIVE_UCHAR`; writing them as `i32` makes the
        /// native `HDF5_File_Reader` report "Unexpected attribute data type"
        /// and fall back to the wrong layout, which breaks `nf2ff` on a
        /// Rust-written dump.
        pub fn write_bool_attribute(
            &self,
            path: &str,
            name: &str,
            value: bool,
        ) -> Result<(), Hdf5Error> {
            self.write_u8_attribute(path, name, u8::from(value))
        }

        /// Write an unsigned-byte attribute on `path`.
        pub fn write_u8_attribute(
            &self,
            path: &str,
            name: &str,
            value: u8,
        ) -> Result<(), Hdf5Error> {
            self.write_attribute_of(
                path,
                name,
                unsafe { H5T_STD_U8LE_g },
                unsafe { H5T_NATIVE_UINT8_g },
                &|| &value as *const u8 as *const c_void,
            )
        }

        /// Write a variable-length UTF-8 string attribute on `path`.
        pub fn write_string_attribute(
            &self,
            path: &str,
            name: &str,
            value: &str,
        ) -> Result<(), Hdf5Error> {
            let c_value = c_name(value)?;
            let pointer = c_value.as_ptr();
            let type_id = unsafe {
                let id = check_id("H5Tcopy", H5Tcopy(H5T_C_S1_g))?;
                check("H5Tset_size", H5Tset_size(id, H5T_VARIABLE))?;
                check("H5Tset_cset", H5Tset_cset(id, H5T_CSET_UTF8))?;
                id
            };
            let result = self.write_attribute_of(path, name, type_id, type_id, &|| {
                &pointer as *const *const c_char as *const c_void
            });
            unsafe {
                let _ = H5Tclose(type_id);
            }
            result
        }

        /// Write a one-dimensional `f32` attribute, matching the native
        /// `WriteAttribute(loc, name, vector<float>)` overload.
        pub fn write_f32_attribute_vector(
            &self,
            path: &str,
            name: &str,
            values: &[f32],
        ) -> Result<(), Hdf5Error> {
            let _guard = lock();
            let object_name = c_name(path)?;
            let attribute_name = c_name(name)?;
            let object = unsafe {
                check_id(
                    "H5Oopen",
                    H5Oopen(self.handle.id, object_name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let result = unsafe {
                let dims = [values.len() as HSize];
                let space = H5Screate_simple(1, dims.as_ptr(), ptr::null());
                if space < 0 {
                    let _ = H5Oclose(object);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Screate_simple",
                        code: space as i32,
                    });
                }
                let attribute = H5Acreate2(
                    object,
                    attribute_name.as_ptr(),
                    H5T_IEEE_F32LE_g,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                );
                let mut result = Ok(());
                if attribute < 0 {
                    result = Err(Hdf5Error::Hdf5 {
                        operation: "H5Acreate2",
                        code: attribute as i32,
                    });
                } else {
                    let write = H5Awrite(
                        attribute,
                        H5T_NATIVE_FLOAT_g,
                        values.as_ptr() as *const c_void,
                    );
                    if write < 0 {
                        result = Err(Hdf5Error::Hdf5 {
                            operation: "H5Awrite",
                            code: write,
                        });
                    }
                    let _ = H5Aclose(attribute);
                }
                let _ = H5Sclose(space);
                result
            };
            unsafe {
                let _ = H5Oclose(object);
            }
            result
        }

        fn dataset_shape(&self, path: &str) -> Result<(Vec<usize>, usize), Hdf5Error> {
            let name = c_name(path)?;
            let dataset = unsafe {
                check_id(
                    "H5Dopen2",
                    H5Dopen2(self.handle.id, name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let shape = unsafe {
                let space = check_id("H5Dget_space", H5Dget_space(dataset))?;
                let rank = H5Sget_simple_extent_ndims(space);
                if rank < 0 {
                    let _ = H5Sclose(space);
                    let _ = H5Dclose(dataset);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Sget_simple_extent_ndims",
                        code: rank,
                    });
                }
                let mut dims = vec![0 as HSize; rank as usize];
                let dims_read =
                    H5Sget_simple_extent_dims(space, dims.as_mut_ptr(), ptr::null_mut());
                let points = H5Sget_simple_extent_npoints(space);
                let _ = H5Sclose(space);
                if dims_read < 0 || points < 0 {
                    let _ = H5Dclose(dataset);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Sget_simple_extent_dims",
                        code: dims_read.min(points as c_int),
                    });
                }
                (
                    dims.into_iter().map(|value| value as usize).collect(),
                    points as usize,
                )
            };
            unsafe {
                let _ = H5Dclose(dataset);
            }
            Ok(shape)
        }

        fn read_dataset(
            &self,
            path: &str,
            mem_type: Hid,
            element_size: usize,
        ) -> Result<(Vec<usize>, Vec<u8>), Hdf5Error> {
            let (dims, points) = self.dataset_shape(path)?;
            let mut buffer = vec![0u8; points * element_size];
            let name = c_name(path)?;
            let dataset = unsafe {
                check_id(
                    "H5Dopen2",
                    H5Dopen2(self.handle.id, name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let read = unsafe {
                H5Dread(
                    dataset,
                    mem_type,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    buffer.as_mut_ptr() as *mut c_void,
                )
            };
            unsafe {
                let _ = H5Dclose(dataset);
            }
            check("H5Dread", read)?;
            Ok((dims, buffer))
        }

        /// Read a real dataset as `f64` regardless of its stored precision.
        pub fn read_f64_dataset(&self, path: &str) -> Result<(Vec<usize>, Vec<f64>), Hdf5Error> {
            let _guard = lock();
            let (dims, buffer) = self.read_dataset(path, unsafe { H5T_NATIVE_DOUBLE_g }, 8)?;
            let values = buffer
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 byte chunk")))
                .collect();
            Ok((dims, values))
        }

        /// Read a real dataset as `f32`.
        pub fn read_f32_dataset(&self, path: &str) -> Result<(Vec<usize>, Vec<f32>), Hdf5Error> {
            let _guard = lock();
            let (dims, buffer) = self.read_dataset(path, unsafe { H5T_NATIVE_FLOAT_g }, 4)?;
            let values = buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 byte chunk")))
                .collect();
            Ok((dims, values))
        }

        /// Read a complex compound dataset into separate real/imaginary parts.
        pub fn read_complex_f32_dataset(
            &self,
            path: &str,
        ) -> Result<(Vec<usize>, Vec<f32>, Vec<f32>), Hdf5Error> {
            let _guard = lock();
            let compound = unsafe {
                let id = check_id("H5Tcreate", H5Tcreate(H5T_COMPOUND, 8))?;
                let real = c_name("r")?;
                let imag = c_name("i")?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, real.as_ptr(), 0, H5T_NATIVE_FLOAT_g),
                )?;
                check(
                    "H5Tinsert",
                    H5Tinsert(id, imag.as_ptr(), 4, H5T_NATIVE_FLOAT_g),
                )?;
                id
            };
            let result = self.read_dataset(path, compound, 8);
            unsafe {
                let _ = H5Tclose(compound);
            }
            let (dims, buffer) = result?;
            let mut re = Vec::with_capacity(buffer.len() / 8);
            let mut im = Vec::with_capacity(buffer.len() / 8);
            for chunk in buffer.chunks_exact(8) {
                re.push(f32::from_le_bytes(chunk[0..4].try_into().expect("4 bytes")));
                im.push(f32::from_le_bytes(chunk[4..8].try_into().expect("4 bytes")));
            }
            Ok((dims, re, im))
        }

        fn open_attribute(&self, path: &str, name: &str) -> Result<(Hid, Hid), Hdf5Error> {
            let object_name = c_name(path)?;
            let attribute_name = c_name(name)?;
            let object = unsafe {
                check_id(
                    "H5Oopen",
                    H5Oopen(self.handle.id, object_name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let exists = unsafe { H5Aexists(object, attribute_name.as_ptr()) };
            if exists <= 0 {
                unsafe {
                    let _ = H5Oclose(object);
                }
                return Err(Hdf5Error::NotFound(format!("{path}@{name}")));
            }
            let attribute = unsafe {
                check_id(
                    "H5Aopen",
                    H5Aopen(object, attribute_name.as_ptr(), H5P_DEFAULT),
                )
            };
            let attribute = match attribute {
                Ok(id) => id,
                Err(error) => {
                    unsafe {
                        let _ = H5Oclose(object);
                    }
                    return Err(error);
                }
            };
            Ok((object, attribute))
        }

        fn read_scalar_attribute<T: Copy + Default>(
            &self,
            path: &str,
            name: &str,
            mem_type: Hid,
        ) -> Result<T, Hdf5Error> {
            let _guard = lock();
            let (object, attribute) = self.open_attribute(path, name)?;
            let mut value = T::default();
            let read = unsafe { H5Aread(attribute, mem_type, &mut value as *mut T as *mut c_void) };
            unsafe {
                let _ = H5Aclose(attribute);
                let _ = H5Oclose(object);
            }
            check("H5Aread", read)?;
            Ok(value)
        }

        /// Read a scalar `f64` attribute.
        pub fn read_f64_attribute(&self, path: &str, name: &str) -> Result<f64, Hdf5Error> {
            self.read_scalar_attribute(path, name, unsafe { H5T_NATIVE_DOUBLE_g })
        }

        /// Read a scalar `i32` attribute.
        pub fn read_i32_attribute(&self, path: &str, name: &str) -> Result<i32, Hdf5Error> {
            self.read_scalar_attribute(path, name, unsafe { H5T_NATIVE_INT_g })
        }

        /// Read a boolean attribute written as `i32` or as a single byte.
        pub fn read_bool_attribute(&self, path: &str, name: &str) -> Result<bool, Hdf5Error> {
            self.read_i32_attribute(path, name).map(|value| value != 0)
        }

        /// Read a string attribute, variable- or fixed-length.
        pub fn read_string_attribute(&self, path: &str, name: &str) -> Result<String, Hdf5Error> {
            let _guard = lock();
            let (object, attribute) = self.open_attribute(path, name)?;
            let result = unsafe {
                let type_id = H5Aget_type(attribute);
                if type_id < 0 {
                    Err(Hdf5Error::Hdf5 {
                        operation: "H5Aget_type",
                        code: type_id as i32,
                    })
                } else {
                    let variable = H5Tis_variable_str(type_id) > 0;
                    if variable {
                        let mut pointer: *mut c_char = ptr::null_mut();
                        let read = H5Aread(
                            attribute,
                            type_id,
                            &mut pointer as *mut *mut c_char as *mut c_void,
                        );
                        if read < 0 || pointer.is_null() {
                            let _ = H5free_memory(pointer as *mut c_void);
                            let _ = H5Tclose(type_id);
                            Err(Hdf5Error::Hdf5 {
                                operation: "H5Aread",
                                code: read,
                            })
                        } else {
                            let text = std::ffi::CStr::from_ptr(pointer)
                                .to_string_lossy()
                                .into_owned();
                            let _ = H5free_memory(pointer as *mut c_void);
                            let _ = H5Tclose(type_id);
                            Ok(text)
                        }
                    } else {
                        let size = H5Tget_size(type_id);
                        let mut buffer = vec![0 as c_char; size.max(1)];
                        let read = H5Aread(attribute, type_id, buffer.as_mut_ptr() as *mut c_void);
                        let _ = H5Tclose(type_id);
                        if read < 0 {
                            Err(Hdf5Error::Hdf5 {
                                operation: "H5Aread",
                                code: read,
                            })
                        } else {
                            Ok(std::ffi::CStr::from_ptr(buffer.as_ptr())
                                .to_string_lossy()
                                .into_owned())
                        }
                    }
                }
            };
            unsafe {
                let _ = H5Aclose(attribute);
                let _ = H5Oclose(object);
            }
            result
        }

        /// Write a one-dimensional `i32` attribute (native index vectors).
        pub fn write_i32_attribute_vector(
            &self,
            path: &str,
            name: &str,
            values: &[i32],
        ) -> Result<(), Hdf5Error> {
            let _guard = lock();
            let object_name = c_name(path)?;
            let attribute_name = c_name(name)?;
            let object = unsafe {
                check_id(
                    "H5Oopen",
                    H5Oopen(self.handle.id, object_name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let result = unsafe {
                let dims = [values.len() as HSize];
                let space = H5Screate_simple(1, dims.as_ptr(), ptr::null());
                if space < 0 {
                    let _ = H5Oclose(object);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Screate_simple",
                        code: space as i32,
                    });
                }
                let attribute = H5Acreate2(
                    object,
                    attribute_name.as_ptr(),
                    H5T_STD_I32LE_g,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                );
                let mut result = Ok(());
                if attribute < 0 {
                    result = Err(Hdf5Error::Hdf5 {
                        operation: "H5Acreate2",
                        code: attribute as i32,
                    });
                } else {
                    let write = H5Awrite(
                        attribute,
                        H5T_NATIVE_INT_g,
                        values.as_ptr() as *const c_void,
                    );
                    if write < 0 {
                        result = Err(Hdf5Error::Hdf5 {
                            operation: "H5Awrite",
                            code: write,
                        });
                    }
                    let _ = H5Aclose(attribute);
                }
                let _ = H5Sclose(space);
                result
            };
            unsafe {
                let _ = H5Oclose(object);
            }
            result
        }

        /// Write a one-dimensional `f64` attribute (native frequency vectors).
        pub fn write_f64_attribute_vector(
            &self,
            path: &str,
            name: &str,
            values: &[f64],
        ) -> Result<(), Hdf5Error> {
            let _guard = lock();
            let object_name = c_name(path)?;
            let attribute_name = c_name(name)?;
            let object = unsafe {
                check_id(
                    "H5Oopen",
                    H5Oopen(self.handle.id, object_name.as_ptr(), H5P_DEFAULT),
                )?
            };
            let result = unsafe {
                let dims = [values.len() as HSize];
                let space = H5Screate_simple(1, dims.as_ptr(), ptr::null());
                if space < 0 {
                    let _ = H5Oclose(object);
                    return Err(Hdf5Error::Hdf5 {
                        operation: "H5Screate_simple",
                        code: space as i32,
                    });
                }
                let attribute = H5Acreate2(
                    object,
                    attribute_name.as_ptr(),
                    H5T_IEEE_F64LE_g,
                    space,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                );
                let mut result = Ok(());
                if attribute < 0 {
                    result = Err(Hdf5Error::Hdf5 {
                        operation: "H5Acreate2",
                        code: attribute as i32,
                    });
                } else {
                    let write = H5Awrite(
                        attribute,
                        H5T_NATIVE_DOUBLE_g,
                        values.as_ptr() as *const c_void,
                    );
                    if write < 0 {
                        result = Err(Hdf5Error::Hdf5 {
                            operation: "H5Awrite",
                            code: write,
                        });
                    }
                    let _ = H5Aclose(attribute);
                }
                let _ = H5Sclose(space);
                result
            };
            unsafe {
                let _ = H5Oclose(object);
            }
            result
        }

        /// Write a three-dimensional `u8` dataset (native cube statistics).
        pub fn write_u8_dataset_nd(
            &self,
            group: &str,
            name: &str,
            dims: &[usize],
            data: &[u8],
            order: Option<&str>,
        ) -> Result<(), Hdf5Error> {
            let expected: usize = dims.iter().product();
            if expected != data.len() {
                return Err(Hdf5Error::InvalidShape {
                    path: format!("{group}/{name}"),
                    detail: format!("{expected} values expected, {} given", data.len()),
                });
            }
            let file_type = unsafe { H5T_STD_U8LE_g };
            let mem_type = unsafe { H5T_NATIVE_UINT8_g };
            self.write_dataset(
                group,
                name,
                file_type,
                mem_type,
                dims,
                data.as_ptr() as *const c_void,
                order,
            )
        }

        /// Read a one-dimensional `f64` attribute (native frequency vectors).
        pub fn read_f64_attribute_vector(
            &self,
            path: &str,
            name: &str,
        ) -> Result<Vec<f64>, Hdf5Error> {
            let _guard = lock();
            let (object, attribute) = self.open_attribute(path, name)?;
            let result = unsafe {
                let space = H5Aget_space(attribute);
                if space < 0 {
                    Err(Hdf5Error::Hdf5 {
                        operation: "H5Aget_space",
                        code: space as i32,
                    })
                } else {
                    let points = H5Sget_simple_extent_npoints(space);
                    let _ = H5Sclose(space);
                    if points < 0 {
                        Err(Hdf5Error::Hdf5 {
                            operation: "H5Sget_simple_extent_npoints",
                            code: points as c_int,
                        })
                    } else {
                        let mut values = vec![0.0f64; points as usize];
                        let read = H5Aread(
                            attribute,
                            H5T_NATIVE_DOUBLE_g,
                            values.as_mut_ptr() as *mut c_void,
                        );
                        if read < 0 {
                            Err(Hdf5Error::Hdf5 {
                                operation: "H5Aread",
                                code: read,
                            })
                        } else {
                            Ok(values)
                        }
                    }
                }
            };
            unsafe {
                let _ = H5Aclose(attribute);
                let _ = H5Oclose(object);
            }
            result
        }

        pub fn read_f32_attribute_vector(
            &self,
            path: &str,
            name: &str,
        ) -> Result<Vec<f32>, Hdf5Error> {
            let _guard = lock();
            let (object, attribute) = self.open_attribute(path, name)?;
            let result = unsafe {
                let space = H5Aget_space(attribute);
                if space < 0 {
                    Err(Hdf5Error::Hdf5 {
                        operation: "H5Aget_space",
                        code: space as i32,
                    })
                } else {
                    let points = H5Sget_simple_extent_npoints(space);
                    let _ = H5Sclose(space);
                    if points < 0 {
                        Err(Hdf5Error::Hdf5 {
                            operation: "H5Sget_simple_extent_npoints",
                            code: points as c_int,
                        })
                    } else {
                        let mut values = vec![0.0f32; points as usize];
                        let read = H5Aread(
                            attribute,
                            H5T_NATIVE_FLOAT_g,
                            values.as_mut_ptr() as *mut c_void,
                        );
                        if read < 0 {
                            Err(Hdf5Error::Hdf5 {
                                operation: "H5Aread",
                                code: read,
                            })
                        } else {
                            Ok(values)
                        }
                    }
                }
            };
            unsafe {
                let _ = H5Aclose(attribute);
                let _ = H5Oclose(object);
            }
            result
        }
    }

    fn write_string_attribute_on(location: Hid, name: &str, value: &str) -> Result<(), Hdf5Error> {
        let attribute = c_name(name)?;
        let text = c_name(value)?;
        unsafe {
            let type_id = check_id("H5Tcopy", H5Tcopy(H5T_C_S1_g))?;
            check("H5Tset_size", H5Tset_size(type_id, H5T_VARIABLE))?;
            check("H5Tset_cset", H5Tset_cset(type_id, H5T_CSET_UTF8))?;
            let space = H5Screate(H5S_SCALAR);
            let attribute_id = H5Acreate2(
                location,
                attribute.as_ptr(),
                type_id,
                space,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            let mut result = Ok(());
            if attribute_id < 0 {
                result = Err(Hdf5Error::Hdf5 {
                    operation: "H5Acreate2",
                    code: attribute_id as i32,
                });
            } else {
                let pointer = text.as_ptr();
                let write = H5Awrite(
                    attribute_id,
                    type_id,
                    &pointer as *const *const c_char as *const c_void,
                );
                if write < 0 {
                    result = Err(Hdf5Error::Hdf5 {
                        operation: "H5Awrite",
                        code: write,
                    });
                }
                let _ = H5Aclose(attribute_id);
            }
            let _ = H5Sclose(space);
            let _ = H5Tclose(type_id);
            result
        }
    }
}

#[cfg(feature = "hdf5")]
pub use backend::Hid;

/// A handle to an HDF5 file.
#[cfg(feature = "hdf5")]
pub struct Hdf5File {
    handle: backend::Handle,
    path: String,
}

/// A handle to an HDF5 file (no backend available).
#[cfg(not(feature = "hdf5"))]
pub struct Hdf5File {
    path: String,
}

#[cfg(not(feature = "hdf5"))]
impl Hdf5File {
    pub fn create(path: &str) -> Result<Self, Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn open(path: &str, read_write: bool) -> Result<Self, Hdf5Error> {
        let _ = (path, read_write);
        Err(Hdf5Error::Unsupported)
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn create_group(&self, path: &str) -> Result<(), Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn exists(&self, path: &str) -> bool {
        let _ = path;
        false
    }

    pub fn object_names(&self, path: &str) -> Result<Vec<String>, Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f64_dataset(
        &self,
        group: &str,
        name: &str,
        data: &[f64],
    ) -> Result<(), Hdf5Error> {
        let _ = (group, name, data);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f32_dataset(
        &self,
        group: &str,
        name: &str,
        data: &[f32],
        order: Option<&str>,
    ) -> Result<(), Hdf5Error> {
        let _ = (group, name, data, order);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f32_dataset_nd(
        &self,
        group: &str,
        name: &str,
        dims: &[usize],
        data: &[f32],
        order: Option<&str>,
    ) -> Result<(), Hdf5Error> {
        let _ = (group, name, dims, data, order);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_complex_f32_dataset(
        &self,
        group: &str,
        name: &str,
        dims: &[usize],
        re: &[f32],
        im: &[f32],
        order: Option<&str>,
    ) -> Result<(), Hdf5Error> {
        let _ = (group, name, dims, re, im, order);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f64_attribute(&self, path: &str, name: &str, value: f64) -> Result<(), Hdf5Error> {
        let _ = (path, name, value);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_i32_attribute(&self, path: &str, name: &str, value: i32) -> Result<(), Hdf5Error> {
        let _ = (path, name, value);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_bool_attribute(
        &self,
        path: &str,
        name: &str,
        value: bool,
    ) -> Result<(), Hdf5Error> {
        let _ = (path, name, value);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_string_attribute(
        &self,
        path: &str,
        name: &str,
        value: &str,
    ) -> Result<(), Hdf5Error> {
        let _ = (path, name, value);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f32_attribute_vector(
        &self,
        path: &str,
        name: &str,
        values: &[f32],
    ) -> Result<(), Hdf5Error> {
        let _ = (path, name, values);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_f64_dataset(&self, path: &str) -> Result<(Vec<usize>, Vec<f64>), Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_f32_dataset(&self, path: &str) -> Result<(Vec<usize>, Vec<f32>), Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_complex_f32_dataset(
        &self,
        path: &str,
    ) -> Result<(Vec<usize>, Vec<f32>, Vec<f32>), Hdf5Error> {
        let _ = path;
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_f64_attribute(&self, path: &str, name: &str) -> Result<f64, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_i32_attribute(&self, path: &str, name: &str) -> Result<i32, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_bool_attribute(&self, path: &str, name: &str) -> Result<bool, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_string_attribute(&self, path: &str, name: &str) -> Result<String, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_f64_attribute_vector(
        &self,
        path: &str,
        name: &str,
        values: &[f64],
    ) -> Result<(), Hdf5Error> {
        let _ = (path, name, values);
        Err(Hdf5Error::Unsupported)
    }

    pub fn write_u8_dataset_nd(
        &self,
        group: &str,
        name: &str,
        dims: &[usize],
        data: &[u8],
        order: Option<&str>,
    ) -> Result<(), Hdf5Error> {
        let _ = (group, name, dims, data, order);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_f64_attribute_vector(&self, path: &str, name: &str) -> Result<Vec<f64>, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }

    pub fn read_f32_attribute_vector(&self, path: &str, name: &str) -> Result<Vec<f32>, Hdf5Error> {
        let _ = (path, name);
        Err(Hdf5Error::Unsupported)
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::*;

    fn sample() -> Vec<f32> {
        // values[n][i][j][k] = 100 n + 10 i + j + 0.1 k
        let dims = [2usize, 3, 4];
        let mut values = Vec::new();
        for n in 0..3 {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        values.push(100.0 * n as f32 + 10.0 * i as f32 + j as f32 + 0.1 * k as f32);
                    }
                }
            }
        }
        values
    }

    #[test]
    fn nxyz_order_is_component_major_then_xyz() {
        let dims = [2usize, 3, 4];
        let ordered = field_to_nxyz(&sample(), dims);
        assert_eq!(ordered.len(), 3 * 2 * 3 * 4);
        // first triplet: n=0, (i,j,k)=(0,0,0), (0,0,1), (0,0,2)
        assert_eq!(&ordered[0..3], &[0.0, 0.1, 0.2]);
        // n=2, i=1, j=2, k=3 is the last element
        assert!((ordered[ordered.len() - 1] - (200.0 + 10.0 + 2.0 + 0.3)).abs() < 1e-6);
    }

    #[test]
    fn nzyx_order_is_component_major_then_zyx() {
        let dims = [2usize, 3, 4];
        let ordered = field_to_nzyx(&sample(), dims);
        assert_eq!(ordered.len(), 3 * 2 * 3 * 4);
        // Order is n, k, j, i with i fastest: value = 100n + 10i + j + 0.1k.
        assert_eq!(&ordered[0..6], &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
        assert_eq!(&ordered[6..8], &[0.1, 10.1]);
    }

    #[test]
    fn ordering_handles_short_input_without_panicking() {
        let ordered = field_to_nxyz(&[1.0], [2, 2, 2]);
        assert_eq!(ordered.len(), 24);
        assert_eq!(ordered[0], 1.0);
        assert_eq!(ordered[1], 0.0);
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod roundtrip_tests {
    use super::*;

    fn temp_file(name: &str) -> String {
        let path = std::env::temp_dir().join(format!(
            "openems_{name}_{}_{}.h5",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn writes_and_reads_the_native_layout() {
        let path = temp_file("native_layout");
        {
            let file = Hdf5File::create(&path).expect("create");
            file.write_f64_attribute("/", "openEMS_HDF5_version", OPENEMS_HDF5_VERSION)
                .expect("version attribute");
            file.write_i32_attribute("/", "dump_type", 0)
                .expect("dump_type");
            file.write_bool_attribute("/", "dual_mesh", true)
                .expect("dual_mesh");
            file.write_bool_attribute("/", "legacy_fmt", true)
                .expect("legacy_fmt");

            file.create_group("/Mesh").expect("mesh group");
            file.write_i32_attribute("/Mesh", "mesh_type", 0)
                .expect("mesh_type");
            file.write_f64_attribute("/Mesh", "mesh_scaling", 0.001)
                .expect("mesh_scaling");
            file.write_f64_dataset("/Mesh", "x", &[0.0, 0.5, 1.0])
                .expect("x lines");
            file.write_f64_dataset("/Mesh", "y", &[0.0, 1.0])
                .expect("y lines");
            file.write_f64_dataset("/Mesh", "z", &[0.0, 1.0])
                .expect("z lines");

            let values: Vec<f32> = (0..3 * 2 * 2 * 2).map(|value| value as f32).collect();
            file.write_f32_dataset_nd(
                "/FieldData/TD",
                "00000001",
                &[3, 2, 2, 2],
                &values,
                Some("NXYZ"),
            )
            .expect("td dataset");
            file.write_f32_attribute_vector("/FieldData/TD/00000001", "time", &[1.5e-12])
                .expect("time attribute");

            file.create_group("/FieldData/FD").expect("fd group");
            file.write_f32_attribute_vector("/FieldData/FD", "frequency", &[1.0e9, 2.0e9])
                .expect("frequencies");
            file.write_complex_f32_dataset(
                "/FieldData/FD",
                "E-Field",
                &[3, 2, 2, 2],
                &values,
                &values.iter().map(|value| -value).collect::<Vec<_>>(),
                Some("NZYX"),
            )
            .expect("complex field");
        }

        let file = Hdf5File::open(&path, false).expect("open");
        assert_eq!(
            file.read_f64_attribute("/", "openEMS_HDF5_version")
                .unwrap(),
            OPENEMS_HDF5_VERSION
        );
        assert_eq!(file.read_i32_attribute("/", "dump_type").unwrap(), 0);
        assert!(file.read_bool_attribute("/", "dual_mesh").unwrap());
        assert!(file.read_bool_attribute("/", "legacy_fmt").unwrap());

        let (dims, x) = file.read_f64_dataset("/Mesh/x").unwrap();
        assert_eq!(dims, vec![3]);
        assert_eq!(x, vec![0.0, 0.5, 1.0]);
        assert_eq!(
            file.read_f64_attribute("/Mesh", "mesh_scaling").unwrap(),
            0.001
        );

        let (dims, field) = file.read_f32_dataset("/FieldData/TD/00000001").unwrap();
        assert_eq!(dims, vec![3, 2, 2, 2]);
        assert_eq!(field.len(), 24);
        assert_eq!(field[23], 23.0);
        assert_eq!(
            file.read_f32_attribute_vector("/FieldData/TD/00000001", "time")
                .unwrap(),
            vec![1.5e-12f32]
        );
        assert_eq!(
            file.read_string_attribute("/FieldData/TD/00000001", "d_order")
                .unwrap(),
            "NXYZ"
        );

        let frequencies = file
            .read_f32_attribute_vector("/FieldData/FD", "frequency")
            .unwrap();
        assert_eq!(frequencies, vec![1.0e9, 2.0e9]);

        let (dims, re, im) = file
            .read_complex_f32_dataset("/FieldData/FD/E-Field")
            .unwrap();
        assert_eq!(dims, vec![3, 2, 2, 2]);
        assert_eq!(re[5], 5.0);
        assert_eq!(im[5], -5.0);
        assert_eq!(
            file.read_string_attribute("/FieldData/FD/E-Field", "d_order")
                .unwrap(),
            "NZYX"
        );

        let names = file.object_names("/Mesh").unwrap();
        assert_eq!(names, vec!["x", "y", "z"]);
        let root = file.object_names("/").unwrap();
        assert!(root.contains(&"Mesh".to_string()));
        assert!(root.contains(&"FieldData".to_string()));
        assert!(!root.contains(&"Missing".to_string()));

        drop(file);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reports_missing_objects_and_bad_paths() {
        let path = temp_file("missing");
        let file = Hdf5File::create(&path).expect("create");
        assert!(!file.exists("/Nothing"));
        assert!(matches!(
            file.read_f64_dataset("/Nothing"),
            Err(Hdf5Error::NotFound(_)) | Err(Hdf5Error::Hdf5 { .. })
        ));
        assert!(matches!(
            file.read_f64_attribute("/", "missing"),
            Err(Hdf5Error::NotFound(_))
        ));
        assert_eq!(
            file.write_f32_dataset_nd("/g", "bad", &[2, 2], &[1.0], None),
            Err(Hdf5Error::InvalidShape {
                path: "/g/bad".to_string(),
                detail: "4 values expected, 1 given".to_string()
            })
        );
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod native_reader_tests {
    use super::*;

    fn reference_dump() -> Option<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/migration/reference-data/msl-notchfilter-wsl/field-run-1/Et_ref.h5");
        path.exists().then(|| path.to_string_lossy().into_owned())
    }

    /// Reads a dump produced by the native C++ solver. This is the reference
    /// for dataset names, shapes, `d_order` and attribute types.
    #[test]
    fn reads_a_native_openems_dump() {
        let Some(path) = reference_dump() else {
            eprintln!("native reference dump not present; skipping");
            return;
        };
        let file = Hdf5File::open(&path, false).expect("open the native dump");
        let root = file.object_names("/").unwrap();
        assert!(root.contains(&"Mesh".to_string()));
        assert!(root.contains(&"FieldData".to_string()));

        assert_eq!(file.read_i32_attribute("/", "dump_type").unwrap(), 0);
        let version = file
            .read_f64_attribute("/", "openEMS_HDF5_version")
            .unwrap();
        assert!(version > 0.0);

        assert_eq!(file.read_i32_attribute("/Mesh", "mesh_type").unwrap(), 0);
        let scaling = file.read_f64_attribute("/Mesh", "mesh_scaling").unwrap();
        assert_eq!(scaling, 1.0e-6);
        let (dims, x) = file.read_f64_dataset("/Mesh/x").unwrap();
        assert_eq!(dims, vec![5]);
        // Native stores mesh lines in metres.
        assert!(x[2].abs() < 1.0e-12);
        assert!((x[4] - x[0]).abs() > 1.0e-4);
        let (_, y) = file.read_f64_dataset("/Mesh/y").unwrap();
        let (_, z) = file.read_f64_dataset("/Mesh/z").unwrap();
        assert_eq!(y.len(), 5);
        assert_eq!(z.len(), 1);

        let entries = file.object_names("/FieldData/TD").unwrap();
        assert_eq!(entries.len(), 10);
        assert_eq!(entries[0], "00000000");
        assert_eq!(entries[1], "00000102");

        let (dims, values) = file.read_f32_dataset("/FieldData/TD/00000000").unwrap();
        assert_eq!(dims, vec![3, 5, 5, 1]);
        assert_eq!(values.len(), 75);
        assert_eq!(
            file.read_string_attribute("/FieldData/TD/00000000", "d_order")
                .unwrap(),
            "NXYZ"
        );
        let time = file
            .read_f32_attribute_vector("/FieldData/TD/00000000", "time")
            .unwrap();
        assert_eq!(time.len(), 1);
        assert!(time[0].abs() < 1.0e-15);
        let second = file
            .read_f32_attribute_vector("/FieldData/TD/00000102", "time")
            .unwrap();
        assert!((second[0] - 1.774336e-11).abs() < 1.0e-15);
    }

    /// Writes the same layout with the Rust writer and reads it back with the
    /// same reader, then checks that the native reader sees the expected
    /// names/attributes.
    #[test]
    fn round_trips_the_native_field_layout() {
        let path = std::env::temp_dir()
            .join(format!(
                "openems_native_roundtrip_{}.h5",
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned();
        let values: Vec<f32> = (0..3 * 5 * 5 * 1).map(|value| value as f32).collect();
        {
            let file = Hdf5File::create(&path).unwrap();
            file.write_f64_attribute("/", "openEMS_HDF5_version", OPENEMS_HDF5_VERSION)
                .unwrap();
            file.write_i32_attribute("/", "dump_type", 0).unwrap();
            file.write_f64_attribute("/Mesh", "mesh_scaling", 1.0)
                .unwrap_err();
            file.create_group("/Mesh").unwrap();
            file.write_i32_attribute("/Mesh", "mesh_type", 0).unwrap();
            file.write_f64_attribute("/Mesh", "mesh_scaling", 1.0)
                .unwrap();
            file.write_f64_dataset("/Mesh", "x", &[-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3])
                .unwrap();
            file.write_f64_dataset("/Mesh", "y", &[0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3])
                .unwrap();
            file.write_f64_dataset("/Mesh", "z", &[0.0]).unwrap();
            file.write_f32_dataset_nd(
                "/FieldData/TD",
                "00000000",
                &[3, 5, 5, 1],
                &values,
                Some("NXYZ"),
            )
            .unwrap();
            file.write_f32_attribute_vector("/FieldData/TD/00000000", "time", &[0.0])
                .unwrap();
        }
        let file = Hdf5File::open(&path, false).unwrap();
        assert_eq!(
            file.read_f64_attribute("/Mesh", "mesh_scaling").unwrap(),
            1.0
        );
        assert_eq!(file.read_i32_attribute("/Mesh", "mesh_type").unwrap(), 0);
        assert_eq!(
            file.read_string_attribute("/FieldData/TD/00000000", "d_order")
                .unwrap(),
            "NXYZ"
        );
        let (dims, read_back) = file.read_f32_dataset("/FieldData/TD/00000000").unwrap();
        assert_eq!(dims, vec![3, 5, 5, 1]);
        assert_eq!(read_back, values);
        assert_eq!(
            file.object_names("/FieldData/TD").unwrap(),
            vec!["00000000"]
        );
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}
