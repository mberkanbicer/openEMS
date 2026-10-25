//! FDTD option parsing from the native `<FDTD>` XML element.
//!
//! Mirrors `openEMS::ParseFDTDSetup`: the solver settings live as attributes of
//! the `<FDTD>` element that precedes `<ContinuousStructure>` in the native
//! input file. The CSXCAD parser only consumes the geometry, so these options
//! are read here and applied to the solver.

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
        while index < bytes.len()
            && !(bytes[index] as char).is_ascii_alphabetic()
        {
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
                    if parsed > 0.0 {
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
            _ => {}
        }
    }
    options
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
        assert_eq!(parse_fdtd_options("<openEMS><ContinuousStructure/></openEMS>"), FdtdOptions::default());
    }

    #[test]
    fn single_quotes_and_case_insensitive_names_are_accepted() {
        let xml = "<FDTD nrts='42' endcriteria='1e-6'/>";
        let options = parse_fdtd_options(xml);
        assert_eq!(options.nr_ts, Some(42));
        assert_eq!(options.end_criteria, Some(1e-6));
    }
}
