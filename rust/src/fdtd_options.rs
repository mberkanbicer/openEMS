//! FDTD option parsing from the native `<FDTD>` XML element.
//!
//! Mirrors `openEMS::ParseFDTDSetup`: the solver settings live as attributes of
//! the `<FDTD>` element that precedes `<ContinuousStructure>` in the native
//! input file. The CSXCAD parser only consumes the geometry, so these options
//! are read here and applied to the solver.

/// Excitation setup from the `<Excitation>` child of `<FDTD>`.
#[derive(Debug, Clone, PartialEq)]
pub enum FdtdExcitation {
    Gauss {
        f0: f64,
        fc: f64,
    },
    Sinus {
        f0: f64,
    },
    Dirac {
        f_max: f64,
    },
    Step {
        f_max: f64,
    },
    Custom {
        function: String,
        f0: f64,
        f_max: f64,
    },
}

/// Solver options gathered from the `<FDTD>` element.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FdtdOptions {
    pub nr_ts: Option<u32>,
    pub end_criteria: Option<f64>,
    pub max_time: Option<f64>,
    pub over_sampling: Option<u32>,
    pub cell_constant_material: Option<bool>,
    pub time_step: Option<f64>,
    pub time_step_factor: Option<f64>,
    pub time_step_method: Option<i32>,
    pub coord_system: Option<i32>,
    pub cylinder_coords: Option<bool>,
    pub multi_grid: Option<Vec<f64>>,
    /// Boundary conditions by native index (0 PEC, 1 PMC, 2 MUR, 3 PML).
    pub bc_type: Option<[i32; 6]>,
    /// PML layer count per boundary index.
    pub pml_size: Option<[u32; 6]>,
    /// Per-boundary Mur phase velocity, `None` when not set.
    pub mur_phase_velocity: [Option<f64>; 6],
    pub excitation: Option<FdtdExcitation>,
}

/// Extract the attribute string of the `<FDTD …>` element, if present.
fn fdtd_tag<'a>(xml: &'a str) -> Option<&'a str> {
    let start = xml.find("<FDTD")?;
    let rest = &xml[start..];
    let end = rest.find('>')?;
    Some(&rest[..end])
}

/// Split `name="value"` pairs out of a tag body.
fn attributes(tag: &str) -> Vec<(String, String)> {
    let bytes = tag.as_bytes();
    let mut result = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        // Find an attribute name.
        while index < bytes.len() && !(bytes[index] as char).is_ascii_alphabetic() {
            index += 1;
        }
        let name_start = index;
        while index < bytes.len()
            && ((bytes[index] as char).is_ascii_alphanumeric()
                || bytes[index] == b'_'
                || bytes[index] == b'-')
        {
            index += 1;
        }
        if name_start == index {
            break;
        }
        let name = &tag[name_start..index];
        let mut probe = index;
        while probe < bytes.len() && (bytes[probe] as char).is_whitespace() {
            probe += 1;
        }
        if probe >= bytes.len() || bytes[probe] != b'=' {
            // This token is the element name, not an attribute.
            continue;
        }
        index = probe + 1; // '='
        while index < bytes.len() && (bytes[index] as char).is_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let quote = bytes[index];
        if quote != b'"' && quote != b'\'' {
            continue;
        }
        index += 1;
        let value_start = index;
        while index < bytes.len() && bytes[index] != quote {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        result.push((name.to_string(), tag[value_start..index].to_string()));
        index += 1;
    }
    result
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parse the `<FDTD>` attributes from a complete native input file.
pub fn parse_fdtd_options(xml: &str) -> FdtdOptions {
    let mut options = FdtdOptions::default();
    let Some(tag) = fdtd_tag(xml) else {
        return options;
    };
    for (name, value) in attributes(tag) {
        match name.to_ascii_lowercase().as_str() {
            "nrts" | "numberoftimesteps" => {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    if parsed > 0.0 {
                        options.nr_ts = Some(parsed as u32);
                    }
                }
            }
            "endcriteria" => {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    // native maps 0 to the default and lets negatives through
                    if parsed != 0.0 {
                        options.end_criteria = Some(parsed);
                    }
                }
            }
            "maxtime" => {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    if parsed > 0.0 {
                        options.max_time = Some(parsed);
                    }
                }
            }
            "oversampling" => {
                if let Ok(parsed) = value.trim().parse::<u32>() {
                    if parsed >= 1 {
                        options.over_sampling = Some(parsed);
                    }
                }
            }
            "cellconstantmaterial" => {
                options.cell_constant_material = parse_bool(&value);
            }
            "timestep" => {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    if parsed > 0.0 {
                        options.time_step = Some(parsed);
                    }
                }
            }
            "timestepfactor" => {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    if parsed > 0.0 {
                        options.time_step_factor = Some(parsed);
                    }
                }
            }
            "timestepmethod" => {
                if let Ok(parsed) = value.trim().parse::<i32>() {
                    options.time_step_method = Some(parsed);
                }
            }
            "coordsystem" => {
                if let Ok(parsed) = value.trim().parse::<i32>() {
                    options.coord_system = Some(parsed);
                }
            }
            "cylindercoords" => {
                options.cylinder_coords = Some(value.trim() == "1");
            }
            "multigrid" => {
                options.multi_grid = Some(
                    value
                        .split(',')
                        .filter_map(|entry| entry.trim().parse::<f64>().ok())
                        .collect(),
                );
            }
            _ => {}
        }
    }

    // `<BoundaryCond ...>` child
    if let Some(tag) = child_tag(xml, "BoundaryCond") {
        let mut bc = [0i32; 6];
        let mut pml = [0u32; 6];
        let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
        let attributes = attributes(&tag);
        for (name, value) in &attributes {
            if let Some(index) = names.iter().position(|entry| entry == name) {
                let value = value.trim();
                match value {
                    "PEC" => bc[index] = 0,
                    "PMC" => bc[index] = 1,
                    "MUR" => bc[index] = 2,
                    _ => {
                        if let Some(size) = value.strip_prefix("PML_") {
                            bc[index] = 3;
                            pml[index] = size.trim().parse::<u32>().unwrap_or(8);
                        } else if let Ok(parsed) = value.parse::<i32>() {
                            bc[index] = parsed;
                        }
                    }
                }
            }
            let mur_names = [
                "MUR_PhaseVelocity_xmin",
                "MUR_PhaseVelocity_xmax",
                "MUR_PhaseVelocity_ymin",
                "MUR_PhaseVelocity_ymax",
                "MUR_PhaseVelocity_zmin",
                "MUR_PhaseVelocity_zmax",
            ];
            if let Some(index) = mur_names.iter().position(|entry| entry == name) {
                options.mur_phase_velocity[index] = value.trim().parse::<f64>().ok();
            }
        }
        options.bc_type = Some(bc);
        options.pml_size = Some(pml);
    }

    // `<Excitation ...>` child
    if let Some(tag) = child_tag(xml, "Excitation") {
        let attributes = attributes(&tag);
        let get = |name: &str| {
            attributes
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        let excitation_type = get("Type")
            .and_then(|value| value.trim().parse::<i32>().ok())
            .unwrap_or(-1);
        let f0 = get("f0")
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        let fc = get("fc")
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        let f_max = get("f_max")
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        options.excitation = match excitation_type {
            0 => Some(FdtdExcitation::Gauss { f0, fc }),
            1 => Some(FdtdExcitation::Sinus { f0 }),
            2 => Some(FdtdExcitation::Dirac { f_max }),
            3 => Some(FdtdExcitation::Step { f_max }),
            10 => Some(FdtdExcitation::Custom {
                function: get("Function").unwrap_or_default(),
                f0,
                f_max,
            }),
            _ => None,
        };
    }

    options
}

/// Attribute body of the first `<name ...>` child element.
fn child_tag(xml: &str, name: &str) -> Option<String> {
    let start = xml.find(&format!("<{name}"))?;
    let rest = &xml[start..];
    let end = rest.find('>')?;
    Some(rest[..end].to_string())
}

/// The native `<FDTD ...>` element (attributes plus the `Excitation` and
/// `BoundaryCond` children) for the given solver settings.
pub fn write_fdtd_element(settings: &FdtdWriteSettings) -> String {
    let mut element = String::from("<FDTD");
    attribute(
        &mut element,
        "NumberOfTimesteps",
        &settings.nr_ts.to_string(),
    );
    if settings.cylinder_coords {
        attribute(&mut element, "CylinderCoords", "1");
        if !settings.multi_grid.is_empty() {
            let radii = settings
                .multi_grid
                .iter()
                .map(|radius| radius.to_string())
                .collect::<Vec<_>>()
                .join(",");
            attribute(&mut element, "MultiGrid", &radii);
        }
    }
    if settings.max_time > 0.0 {
        attribute(&mut element, "MaxTime", &settings.max_time.to_string());
    }
    attribute(
        &mut element,
        "endCriteria",
        &settings.end_criteria.to_string(),
    );
    attribute(
        &mut element,
        "OverSampling",
        &settings.over_sampling.to_string(),
    );
    if settings.cell_constant_material {
        attribute(&mut element, "CellConstantMaterial", "1");
    }
    if let FdtdExcitation::Dirac { f_max }
    | FdtdExcitation::Step { f_max }
    | FdtdExcitation::Custom { f_max, .. } = settings.excitation
    {
        attribute(&mut element, "f_max", &f_max.to_string());
    }
    attribute(
        &mut element,
        "TimeStepMethod",
        &settings.time_step_method.to_string(),
    );
    if settings.time_step > 0.0 {
        attribute(&mut element, "TimeStep", &settings.time_step.to_string());
    }
    if settings.time_step_factor > 1.0 {
        attribute(
            &mut element,
            "TimeStepFactor",
            &settings.time_step_factor.to_string(),
        );
    }
    element.push('>');

    let mut excitation = String::from("<Excitation");
    match &settings.excitation {
        FdtdExcitation::Gauss { f0, fc } => {
            attribute(&mut excitation, "Type", "0");
            attribute(&mut excitation, "f0", &f0.to_string());
            attribute(&mut excitation, "fc", &fc.to_string());
        }
        FdtdExcitation::Sinus { f0 } => {
            attribute(&mut excitation, "Type", "1");
            attribute(&mut excitation, "f0", &f0.to_string());
        }
        FdtdExcitation::Dirac { .. } => attribute(&mut excitation, "Type", "2"),
        FdtdExcitation::Step { .. } => attribute(&mut excitation, "Type", "3"),
        FdtdExcitation::Custom { function, f0, .. } => {
            attribute(&mut excitation, "Type", "10");
            attribute(&mut excitation, "f0", &f0.to_string());
            attribute(&mut excitation, "Function", &escape_attribute(function));
        }
    }
    excitation.push_str("/>");
    element.push_str(&excitation);

    let mut boundary = String::from("<BoundaryCond");
    let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
    for index in 0..6 {
        let value = match settings.bc_type[index] {
            0 => "PEC".to_string(),
            1 => "PMC".to_string(),
            2 => "MUR".to_string(),
            3 => format!("PML_{}", settings.pml_size[index]),
            other => other.to_string(),
        };
        attribute(&mut boundary, names[index], &value);
        if let Some(velocity) = settings.mur_phase_velocity[index] {
            if velocity > 0.0 {
                let name = format!("MUR_PhaseVelocity_{}", names[index]);
                attribute(&mut boundary, &name, &velocity.to_string());
            }
        }
    }
    boundary.push_str("/>");
    element.push_str(&boundary);
    element.push_str("</FDTD>");
    element
}

fn attribute(element: &mut String, name: &str, value: &str) {
    element.push(' ');
    element.push_str(name);
    element.push('=');
    element.push('"');
    element.push_str(value);
    element.push('"');
}

fn escape_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Everything the native `<FDTD>` writer emits.
#[derive(Debug, Clone, PartialEq)]
pub struct FdtdWriteSettings {
    pub nr_ts: u32,
    pub end_criteria: f64,
    pub max_time: f64,
    pub over_sampling: u32,
    pub cell_constant_material: bool,
    pub cylinder_coords: bool,
    pub multi_grid: Vec<f64>,
    pub time_step: f64,
    pub time_step_factor: f64,
    pub time_step_method: i32,
    pub bc_type: [i32; 6],
    pub pml_size: [u32; 6],
    pub mur_phase_velocity: [Option<f64>; 6],
    pub excitation: FdtdExcitation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_native_fdtd_attributes() {
        let xml = r#"<?xml version="1.0"?>
            <openEMS>
                <FDTD NumberOfTimesteps="1000" EndCriteria="1e-5" MaxTime="2e-9"
                      OverSampling="4" CellConstantMaterial="1" TimeStepMethod="3"
                      TimeStepFactor="0.9" CoordSystem="0"/>
                <ContinuousStructure CoordSystem="0"/>
            </openEMS>"#;
        let options = parse_fdtd_options(xml);
        assert_eq!(options.nr_ts, Some(1000));
        assert_eq!(options.end_criteria, Some(1e-5));
        assert_eq!(options.max_time, Some(2e-9));
        assert_eq!(options.over_sampling, Some(4));
        assert_eq!(options.cell_constant_material, Some(true));
        assert_eq!(options.time_step_method, Some(3));
        assert_eq!(options.time_step_factor, Some(0.9));
        assert_eq!(options.coord_system, Some(0));
    }

    #[test]
    fn a_file_without_fdtd_keeps_the_defaults() {
        assert_eq!(
            parse_fdtd_options("<openEMS><ContinuousStructure/></openEMS>"),
            FdtdOptions::default()
        );
    }

    #[test]
    fn single_quotes_and_case_insensitive_names_are_accepted() {
        let xml = "<FDTD nrts='42' endcriteria='1e-6'/>";
        let options = parse_fdtd_options(xml);
        assert_eq!(options.nr_ts, Some(42));
        assert_eq!(options.end_criteria, Some(1e-6));
    }
}
