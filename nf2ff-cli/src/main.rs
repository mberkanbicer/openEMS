//! `nf2ff` command-line tool: near-field to far-field transformation.
//!
//! Mirrors `nf2ff/main.cpp` + `nf2ff::AnalyseXMLFile`: the first positional
//! argument is an `<nf2ff>` XML file that lists the frequencies, angles,
//! centre, output file and the recorded box faces.
//!
//! Exit codes follow the native entry point: `0` on success and `1` (the native
//! `EXIT_FAILURE`) when the file cannot be analysed or no file is given.

use std::process::ExitCode;

use openems_core::nf2ff_hdf5::Nf2ffRunner;

const BANNER: &str = "\
 ---------------------------------------------------------------------- 
 | nf2ff, near-field to far-field transformation for openEMS 
 | (C) 2012-2025 Thorsten Liebig <thorsten.liebig@gmx.de>  GPL license
 ---------------------------------------------------------------------- ";

fn main() -> ExitCode {
    println!("{BANNER}");

    let mut args = std::env::args().skip(1);
    let mut file = None;
    for argument in args.by_ref() {
        file = Some(argument);
    }
    let Some(file) = file else {
        println!(" Usage: nf2ff <nf2ff-xml-file>");
        return ExitCode::FAILURE;
    };

    match analyse_xml_file(&file) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nf2ff: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Parse and run one `<nf2ff>` document (native `AnalyseXMLNode`).
fn analyse_xml_file(path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("AnalyseXMLFile: Error loading xml-file failed!!! File: {path} ({error})"))?;

    let (root_tag, root_body_start) = find_tag(&text, "nf2ff")
        .ok_or_else(|| "AnalyseXMLFile: Can't read nf2ff ...".to_string())?;
    let attributes = parse_attributes(&root_tag);

    let frequencies = get_attribute(&attributes, "freq")
        .map(|value| float_list(&value))
        .ok_or_else(|| "AnalyseXMLNode: Can't read frequency information...".to_string())?;
    if frequencies.is_empty() {
        return Err("AnalyseXMLNode: Can't read frequency information...".to_string());
    }

    let center = get_attribute(&attributes, "Center")
        .map(|value| float_list(&value))
        .unwrap_or_default();

    let outfile = get_attribute(&attributes, "Outfile")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "AnalyseXMLNode: Can't read output file name...".to_string())?;

    let body = &text[root_body_start..];
    let theta = float_list(
        &element_text(body, "theta")
            .ok_or_else(|| "AnalyseXMLNode: Can't read theta values ...".to_string())?,
    );
    let phi = float_list(
        &element_text(body, "phi")
            .ok_or_else(|| "AnalyseXMLNode: Can't read phi values ...".to_string())?,
    );

    let mut runner = Nf2ffRunner::new(frequencies, theta, phi, {
        let mut center = center;
        center.resize(3, 0.0);
        [center[0], center[1], center[2]]
    });

    if let Some(verbose) = get_attribute(&attributes, "Verbose") {
        let level = verbose.trim().parse::<i32>().unwrap_or(0);
        eprintln!("nf2ff: Set verbose level to {level}");
        runner.set_verbose_level(level);
    }
    if let Some(eps) = get_attribute(&attributes, "Eps_r") {
        runner.set_permittivity(&float_list(&eps));
    }
    if let Some(mue) = get_attribute(&attributes, "Mue_r") {
        runner.set_permeability(&float_list(&mue));
    }
    if let Some(radius) = get_attribute(&attributes, "Radius") {
        if let Ok(radius) = radius.trim().parse::<f64>() {
            runner.set_radius(radius);
        }
    }
    if let Some(legacy) = get_attribute(&attributes, "LegacyHDF5") {
        runner.set_legacy_format(legacy.trim().parse::<i32>().unwrap_or(0) != 0);
    }

    for tag in find_all_tags(body, "Mirror") {
        let attributes = parse_attributes(&tag);
        let mirror_type = get_attribute(&attributes, "Type").unwrap_or_default();
        let direction = get_attribute(&attributes, "Dir")
            .and_then(|value| value.trim().parse::<i32>().ok());
        let position = get_attribute(&attributes, "Pos")
            .and_then(|value| value.trim().parse::<f64>().ok());
        let (Some(direction), Some(position)) = (direction, position) else {
            continue;
        };
        let mirror = match mirror_type.trim() {
            "PEC" => 1,
            "PMC" => 2,
            _ => continue,
        };
        runner.set_mirror(mirror, direction as usize, position);
    }

    let mut planes = 0usize;
    for tag in find_all_tags(body, "Planes") {
        let attributes = parse_attributes(&tag);
        let e_field = get_attribute(&attributes, "E_Field").unwrap_or_default();
        let h_field = get_attribute(&attributes, "H_Field").unwrap_or_default();
        if e_field.is_empty() || h_field.is_empty() {
            return Err("AnalyseXMLNode: Error, invalid plane entry ...".to_string());
        }
        runner
            .analyse_file(&e_field, &h_field)
            .map_err(|error| format!("AnalyseXMLNode: Error, analysing Plane ... {error}"))?;
        planes += 1;
    }
    if planes == 0 {
        eprintln!("nf2ff: Warning: no planes analysed");
    }

    runner
        .write_hdf5(&outfile)
        .map_err(|error| format!("Write2HDF5: {error}"))?;
    Ok(())
}

/// The complete `<name ...>` tag body and the offset just past its `>`.
fn find_tag(text: &str, name: &str) -> Option<(String, usize)> {
    let start = text.find(&format!("<{name}"))?;
    let rest = &text[start..];
    let end = rest.find('>')?;
    Some((rest[..end].to_string(), start + end + 1))
}

fn find_all_tags(text: &str, name: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut offset = 0usize;
    while let Some((tag, end)) = find_tag(&text[offset..], name) {
        tags.push(tag);
        offset += end;
    }
    tags
}

/// Text content of the first `<name>...</name>` element.
fn element_text(text: &str, name: &str) -> Option<String> {
    let (_, start) = find_tag(text, name)?;
    let rest = &text[start..];
    let end = rest.find(&format!("</{name}>"))?;
    Some(rest[..end].to_string())
}

fn parse_attributes(tag: &str) -> Vec<(String, String)> {
    let bytes = tag.as_bytes();
    let mut attributes = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
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
            continue;
        }
        let mut value_start = probe + 1;
        while value_start < bytes.len() && (bytes[value_start] as char).is_whitespace() {
            value_start += 1;
        }
        if value_start >= bytes.len() {
            break;
        }
        let quote = bytes[value_start];
        if quote != b'"' && quote != b'\'' {
            continue;
        }
        value_start += 1;
        let mut value_end = value_start;
        while value_end < bytes.len() && bytes[value_end] != quote {
            value_end += 1;
        }
        attributes.push((name.to_string(), tag[value_start..value_end].to_string()));
        index = value_end + 1;
    }
    attributes
}

fn get_attribute(attributes: &[(String, String)], name: &str) -> Option<String> {
    attributes
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

fn float_list(text: &str) -> Vec<f64> {
    text.split(|character: char| {
        character == ',' || character == ';' || character.is_whitespace()
    })
    .filter(|part| !part.is_empty())
    .filter_map(|part| part.parse::<f64>().ok())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_native_nf2ff_xml() {
        let text = r#"<?xml version="1.0"?>
<nf2ff Outfile="nf2ff.h5" Center="0,0,0" Radius="1" Eps_r="1" Mue_r="1" LegacyHDF5="1" Verbose="0" freq="1e9,2e9">
    <theta>0, 0.5, 1</theta>
    <phi>0 1.5707963</phi>
    <Mirror Type="PEC" Dir="0" Pos="-0.01"/>
    <Planes E_Field="Et_0.h5" H_Field="Ht_0.h5"/>
</nf2ff>"#;
        let (root, start) = find_tag(text, "nf2ff").unwrap();
        let attributes = parse_attributes(&root);
        assert_eq!(
            get_attribute(&attributes, "Outfile").as_deref(),
            Some("nf2ff.h5")
        );
        assert_eq!(get_attribute(&attributes, "LegacyHDF5").as_deref(), Some("1"));
        assert_eq!(float_list(&get_attribute(&attributes, "freq").unwrap()), vec![1e9, 2e9]);
        let body = &text[start..];
        assert_eq!(
            float_list(&element_text(body, "theta").unwrap()),
            vec![0.0, 0.5, 1.0]
        );
        let mirrors = find_all_tags(body, "Mirror");
        assert_eq!(mirrors.len(), 1);
        let mirror_attributes = parse_attributes(&mirrors[0]);
        assert_eq!(
            get_attribute(&mirror_attributes, "Type").as_deref(),
            Some("PEC")
        );
        let planes = find_all_tags(body, "Planes");
        assert_eq!(planes.len(), 1);
        assert_eq!(
            get_attribute(&parse_attributes(&planes[0]), "E_Field").as_deref(),
            Some("Et_0.h5")
        );
    }
}
