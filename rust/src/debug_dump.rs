//! Debug dumps used by the native `--debug-*` CLI options.
//!
//! Writes plain legacy VTK files for scalar cell data and for primitive boxes.

use std::fs::File;
use std::io::{self, Write};

/// Write a scalar cell field as a legacy VTK structured grid.
///
/// `values` is indexed `(i, j, k)` with `i` slowest, matching the solver's
/// cell ordering. Points are the cell centres.
pub fn write_scalar_vtk(
    path: &str,
    name: &str,
    dims: [usize; 3],
    lines: [&[f64]; 3],
    values: &[f64],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "openEMS {name}")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET STRUCTURED_GRID")?;
    writeln!(file, "DIMENSIONS {} {} {}", dims[0], dims[1], dims[2])?;
    writeln!(file, "POINTS {} double", dims[0] * dims[1] * dims[2])?;
    write_points(&mut file, dims, lines)?;
    writeln!(file, "POINT_DATA {}", values.len())?;
    writeln!(file, "SCALARS {name} double 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    // VTK orders point data with i fastest, like `VTK_File_Writer`.
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let index = (i * dims[1] + j) * dims[2] + k;
                writeln!(file, "{:.12e}", values[index])?;
            }
        }
    }
    Ok(())
}

/// Write the operator coefficients as a legacy VTK structured grid.
///
/// Mirrors the native `--debug-operator` dump (`VTK_File_Writer` with
/// `SetNativeDump(true)`): every coefficient set is a three-component vector
/// at the mesh nodes, in the order `vv`, `vi`, `ii`, `iv`.
pub fn write_operator_vtk(
    path: &str,
    dims: [usize; 3],
    lines: [&[f64]; 3],
    fields: &[(&str, [&[f64]; 3])],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "openEMS operator")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET STRUCTURED_GRID")?;
    writeln!(file, "DIMENSIONS {} {} {}", dims[0], dims[1], dims[2])?;
    writeln!(file, "POINTS {} double", dims[0] * dims[1] * dims[2])?;
    write_points(&mut file, dims, lines)?;
    writeln!(file, "POINT_DATA {}", dims[0] * dims[1] * dims[2])?;
    for (name, components) in fields {
        writeln!(file, "VECTORS {name} double")?;
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let index = (i * dims[1] + j) * dims[2] + k;
                    writeln!(
                        file,
                        "{:.12e} {:.12e} {:.12e}",
                        components[0][index], components[1][index], components[2][index]
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn write_points(file: &mut File, dims: [usize; 3], lines: [&[f64]; 3]) -> io::Result<()> {
    // VTK stores points with i fastest; native `VTK_File_Writer` iterates
    // k, j, i (`vtk_file_writer.cpp`, the `SetTuple3` loop).
    for k in 0..dims[2] {
        let z = if lines[2].is_empty() {
            k as f64
        } else {
            lines[2][k]
        };
        for j in 0..dims[1] {
            let y = if lines[1].is_empty() {
                j as f64
            } else {
                lines[1][j]
            };
            for i in 0..dims[0] {
                let x = if lines[0].is_empty() {
                    i as f64
                } else {
                    lines[0][i]
                };
                writeln!(file, "{x:.12e} {y:.12e} {z:.12e}")?;
            }
        }
    }
    Ok(())
}

/// Write primitive boxes as a legacy VTK unstructured grid of edges.
pub fn write_boxes_vtk(path: &str, name: &str, boxes: &[[f64; 6]]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "openEMS {name}")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(file, "POINTS {} double", boxes.len() * 8)?;
    for bounds in boxes {
        let [x0, y0, z0, x1, y1, z1] = *bounds;
        for (x, y, z) in [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ] {
            writeln!(file, "{x:.12e} {y:.12e} {z:.12e}")?;
        }
    }
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    writeln!(file, "CELLS {}", boxes.len() * edges.len())?;
    for b in 0..boxes.len() {
        for (a, b) in edges {
            writeln!(file, "2 {} {}", b * 8 + a, b * 8 + b)?;
        }
    }
    writeln!(file, "CELL_TYPES {}", boxes.len() * edges.len())?;
    for _ in 0..boxes.len() * edges.len() {
        writeln!(file, "3")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_vtk_has_the_expected_header_and_values() {
        let path = std::env::temp_dir().join("openems_scalar_debug.vtk");
        let path_string = path.to_string_lossy().into_owned();
        let lines = [&[0.0, 1.0][..], &[0.0, 1.0][..], &[0.0, 1.0][..]];
        write_scalar_vtk(&path_string, "eps", [2, 2, 2], lines, &[1.0; 8]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("SCALARS eps double 1"));
        assert!(text.contains("DIMENSIONS 2 2 2"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn box_vtk_writes_twelve_edges_per_box() {
        let path = std::env::temp_dir().join("openems_box_debug.vtk");
        let path_string = path.to_string_lossy().into_owned();
        write_boxes_vtk(&path_string, "boxes", &[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("CELLS 12"));
        assert!(text.contains("CELL_TYPES 12"));
        std::fs::remove_file(path).ok();
    }
}

/// Write the PEC edges as a legacy-VTK `.vtp` PolyData file (native
/// `--debug-PEC`): one line per metal edge, a `direction` cell array (0, 1, 2
/// for x, y, z) and the `LengthUnit` field data.
pub fn write_pec_edges_vtp(
    path: &str,
    points: &[[f64; 3]],
    lines: &[[usize; 2]],
    directions: &[u8],
    length_unit: f64,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "<?xml version=\"1.0\"?>")?;
    writeln!(
        file,
        "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(file, "  <PolyData>")?;
    writeln!(
        file,
        "    <Piece NumberOfPoints=\"{}\" NumberOfLines=\"{}\" NumberOfVerts=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">",
        points.len(),
        lines.len()
    )?;
    write!(file, "      <Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")?;
    for point in points {
        write!(file, " {} {} {}", point[0], point[1], point[2])?;
    }
    writeln!(file, "</DataArray></Points>")?;
    write!(file, "      <Lines><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")?;
    for line in lines {
        write!(file, " {} {}", line[0], line[1])?;
    }
    writeln!(file, "</DataArray>")?;
    write!(file, "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")?;
    for index in 1..=lines.len() {
        write!(file, " {}", index * 2)?;
    }
    writeln!(file, "</DataArray></Lines>")?;
    write!(file, "      <CellData><DataArray type=\"UInt8\" Name=\"direction\" format=\"ascii\">")?;
    for direction in directions {
        write!(file, " {direction}")?;
    }
    writeln!(file, "</DataArray></CellData>")?;
    writeln!(
        file,
        "      <FieldData><DataArray type=\"Float64\" Name=\"LengthUnit\" NumberOfTuples=\"1\" format=\"ascii\"> {length_unit} </DataArray></FieldData>"
    )?;
    writeln!(file, "    </Piece>")?;
    writeln!(file, "  </PolyData>")?;
    writeln!(file, "</VTKFile>")?;
    Ok(())
}

#[cfg(test)]
mod pec_edges_tests {
    use super::*;

    #[test]
    fn the_pec_edges_are_written_with_direction_and_unit() {
        let path = std::env::temp_dir().join("openems_pec_edges_test.vtp");
        let path_text = path.to_string_lossy().to_string();
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let lines = [[0usize, 1], [0, 2]];
        write_pec_edges_vtp(&path_text, &points, &lines, &[0, 1], 1.0e-3).expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("type=\"PolyData\""), "{text}");
        assert!(text.contains("NumberOfPoints=\"3\""));
        assert!(text.contains("NumberOfLines=\"2\""));
        assert!(text.contains("Name=\"direction\""));
        assert!(text.contains("Name=\"LengthUnit\""));
        assert!(text.contains("1e-3") || text.contains("0.001"));
        let _ = std::fs::remove_file(&path);
    }
}
