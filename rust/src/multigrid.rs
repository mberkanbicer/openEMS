//! Cylindrical multi-grid operator plan (E-007, stage 1).
//!
//! Mirrors the native `Operator_CylinderMultiGrid` split/level rules and the
//! `Operator_Extension::IsCylindricalMultiGridSave` compatibility matrix.
//! The engine is not wired yet: `setup_fdtd` still rejects multi-grid radii, but
//! the plan is validated here and its tests pin the native rules.

/// Native `CYLIDINDERMULTIGRID_LIMIT`.
pub const MULTIGRID_LIMIT: usize = 20;

/// Radii of a (possibly cascaded) cylindrical multi-grid.
///
/// The last radius is the base/child split; the earlier radii describe the
/// nested child levels, exactly like the native constructor.
#[derive(Clone, Debug, PartialEq)]
pub struct CylinderMultiGridPlan {
    radii: Vec<f64>,
}

impl CylinderMultiGridPlan {
    pub fn new(radii: &[f64]) -> Result<Self, String> {
        if radii.is_empty() {
            return Err("cylindrical multi-grid needs at least one split radius".to_string());
        }
        if radii.len() > MULTIGRID_LIMIT {
            return Err(format!(
                "cylindrical multi-grid supports at most {MULTIGRID_LIMIT} levels, got {}",
                radii.len()
            ));
        }
        let mut previous = f64::NEG_INFINITY;
        for (index, radius) in radii.iter().enumerate() {
            if !radius.is_finite() || *radius <= 0.0 {
                return Err(format!("multi-grid radius {index} must be finite and positive"));
            }
            if *radius <= previous {
                return Err(format!(
                    "multi-grid radii must be strictly increasing (index {index})"
                ));
            }
            previous = *radius;
        }
        Ok(Self {
            radii: radii.to_vec(),
        })
    }

    /// Number of grids: one base plus one per split radius.
    pub fn grid_count(&self) -> usize {
        self.radii.len() + 1
    }

    /// The radius that separates the base grid from its first child.
    pub fn split_radius(&self) -> f64 {
        *self.radii.last().expect("validated non-empty")
    }

    /// Radii of the child levels (all but the outer split).
    pub fn child_radii(&self) -> &[f64] {
        &self.radii[..self.radii.len() - 1]
    }

    pub fn radii(&self) -> &[f64] {
        &self.radii
    }
}

/// Result of placing the split radius into the base mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSplit {
    pub split_pos: usize,
    pub split_radius: f64,
}

/// Native `Operator_CylinderMultiGrid::SetupCSXGrid` split search.
///
/// The first rho line strictly above the requested split radius wins; the
/// split must stay at least four lines away from both rho ends, and the alpha
/// line count (minus a closed alpha plane) must be odd.
pub fn find_grid_split(
    rho_lines: &[f64],
    requested_split: f64,
    alpha_lines: usize,
    closed_alpha: bool,
) -> Result<GridSplit, String> {
    if rho_lines.len() < 8 {
        return Err("multi-grid needs at least 8 rho lines".to_string());
    }
    if (alpha_lines - usize::from(closed_alpha)) % 2 != 1 {
        return Err(format!(
            "alpha line count must be odd for multi-grid (got {alpha_lines}, closed={closed_alpha})"
        ));
    }
    let Some(position) = rho_lines.iter().position(|line| requested_split < *line) else {
        return Err(format!(
            "split radius {requested_split} is outside the rho mesh (max {})",
            rho_lines.last().copied().unwrap_or(0.0)
        ));
    };
    if position < 4 || position > rho_lines.len() - 4 {
        return Err(format!(
            "split position {position} must stay 4 lines away from the rho ends ({} lines)",
            rho_lines.len()
        ));
    }
    Ok(GridSplit {
        split_pos: position,
        split_radius: rho_lines[position],
    })
}

/// Native `SetGeometryCSX` grid derivation: the base operator gets the rho
/// lines up to the split and every second alpha line, while the child keeps the
/// full mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiGridLines {
    pub base_rho: Vec<f64>,
    pub base_alpha: Vec<f64>,
    pub child_rho: Vec<f64>,
    pub child_alpha: Vec<f64>,
}

pub fn split_lines(
    rho_lines: &[f64],
    alpha_lines: &[f64],
    split: GridSplit,
) -> MultiGridLines {
    MultiGridLines {
        base_rho: rho_lines[..split.split_pos].to_vec(),
        base_alpha: alpha_lines.iter().copied().step_by(2).collect(),
        child_rho: rho_lines.to_vec(),
        child_alpha: alpha_lines.to_vec(),
    }
}

/// Operator extension families with a native `IsCylindricalMultiGridSave` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionKind {
    ConductingSheet,
    Cylinder,
    Debye,
    Excitation,
    Lorentz,
    LumpedRlc,
    SteadyState,
    Mur { component: usize, top: bool },
    Tfsf,
    Upml,
    Unknown,
}

impl ExtensionKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::ConductingSheet => "Conducting Sheet",
            Self::Cylinder => "Cylinder",
            Self::Debye => "Debye",
            Self::Excitation => "Excitation",
            Self::Lorentz => "Lorentz",
            Self::LumpedRlc => "Lumped RLC",
            Self::SteadyState => "Steady-State",
            Self::Mur { .. } => "Mur ABC",
            Self::Tfsf => "TFSF",
            Self::Upml => "UPML",
            Self::Unknown => "unknown",
        }
    }

    /// Native `IsCylindricalMultiGridSave(child)`.
    pub fn multigrid_safe(self, child: bool) -> bool {
        match self {
            Self::ConductingSheet
            | Self::Cylinder
            | Self::Debye
            | Self::Excitation
            | Self::Lorentz
            | Self::LumpedRlc
            | Self::SteadyState => true,
            Self::Mur { component, top } => {
                if component == 2 {
                    true
                } else {
                    component == 0 && top && !child
                }
            }
            Self::Tfsf => false,
            Self::Upml => !child,
            Self::Unknown => false,
        }
    }
}

/// Validate a set of extensions against the native multi-grid matrix.
///
/// `child` selects which side of the split the extension is attached to; the
/// caller checks both sides for extensions that span the interface.
pub fn check_extensions(extensions: &[(ExtensionKind, bool)]) -> Result<(), String> {
    for (kind, child) in extensions {
        if !kind.multigrid_safe(*child) {
            return Err(format!(
                "{} is not safe on the {} side of a cylindrical multi-grid",
                kind.name(),
                if *child { "child" } else { "base" }
            ));
        }
    }
    Ok(())
}



/// One base alpha index mapped to two child samples with weights
/// (native `m_interpol_*_2p/2pp` tables).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolationPair {
    pub pos_p: usize,
    pub pos_pp: usize,
    pub weight_p: f64,
    pub weight_pp: f64,
}

/// Child-to-base interpolation tables for the alpha direction
/// (native `Operator_CylinderMultiGrid::SetupInterpolation`).
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaInterpolation {
    pub voltage: Vec<InterpolationPair>,
    pub current: Vec<InterpolationPair>,
}

fn alpha_line(lines: &[f64], index: usize, dual: bool) -> f64 {
    if !dual {
        lines[index]
    } else if index + 1 < lines.len() {
        0.5 * (lines[index] + lines[index + 1])
    } else {
        lines[index] + 0.5 * (lines[index] - lines[index - 1])
    }
}

fn alpha_delta(lines: &[f64], index: usize, dual: bool) -> f64 {
    if !dual {
        if index + 1 < lines.len() {
            lines[index + 1] - lines[index]
        } else {
            lines[index] - lines[index - 1]
        }
    } else if index == 0 {
        lines[1] - lines[0]
    } else {
        alpha_line(lines, index, true) - alpha_line(lines, index - 1, true)
    }
}

impl AlphaInterpolation {
    /// Build the native tables. `base` is the coarse operator's alpha mesh and
    /// `child` the fine one; `closed` marks a full 2*pi range.
    pub fn new(base: &[f64], child: &[f64], closed: bool) -> Result<Self, String> {
        if base.len() < 2 || child.len() < 3 {
            return Err("interpolation needs at least two base and three child lines".to_string());
        }
        let mut voltage = Vec::with_capacity(base.len());
        let mut current = Vec::with_capacity(base.len());
        for a_n in 0..base.len() {
            let is_odd = a_n % 2;
            let is_even = 1 - is_odd;

            let mut i_p = a_n / 2;
            let mut i_pp = a_n / 2 + is_odd;
            if !closed && a_n == 1 {
                i_p = 2;
            }
            if !closed && a_n + 2 == base.len() {
                i_pp = a_n / 2 - 1;
            }
            let i_p = i_p.min(child.len() - 1);
            let i_pp = i_pp.min(child.len() - 1);
            let (i_wp, i_wpp) = if i_p == i_pp {
                let ratio = alpha_delta(base, a_n, true) / alpha_delta(child, i_p, true);
                (ratio, ratio)
            } else {
                let line_p = alpha_line(child, i_p, false);
                let line_pp = alpha_line(child, i_pp, false);
                let this_line = alpha_line(base, a_n, false);
                let base_delta = alpha_delta(base, a_n, true);
                (
                    (line_pp - this_line) / (line_pp - line_p)
                        * base_delta / alpha_delta(child, i_p, true),
                    (this_line - line_p) / (line_pp - line_p)
                        * base_delta / alpha_delta(child, i_pp, true),
                )
            };
            current.push(InterpolationPair {
                pos_p: i_p,
                pos_pp: i_pp,
                weight_p: i_wp,
                weight_pp: i_wpp,
            });

            let mut v_p = a_n / 2;
            let mut v_pp = (a_n / 2 + is_odd) as isize - is_even as isize;
            if a_n == 0 && closed {
                v_pp = child.len() as isize - 3;
            } else if a_n == 0 && !closed {
                v_pp = 1;
            }
            if !closed && a_n + 2 == base.len() {
                v_pp = a_n as isize / 2 - 1;
            }
            if !closed && a_n + 1 == base.len() {
                v_p = 0;
                v_pp = 0;
            }
            let v_p = v_p.min(child.len() - 1);
            let v_pp = v_pp.max(0).min(child.len() as isize - 1) as usize;
            let (v_wp, v_wpp) = if v_p == v_pp {
                (0.0, 0.0)
            } else {
                let line_p = alpha_line(child, v_p, true);
                let line_pp = alpha_line(child, v_pp, true);
                let this_line = alpha_line(base, a_n, true);
                let base_delta = alpha_delta(base, a_n, false);
                (
                    (line_pp - this_line) / (line_pp - line_p)
                        * base_delta / alpha_delta(child, v_p, false),
                    (this_line - line_p) / (line_pp - line_p)
                        * base_delta / alpha_delta(child, v_pp, false),
                )
            };
            voltage.push(InterpolationPair {
                pos_p: v_p,
                pos_pp: v_pp,
                weight_p: v_wp,
                weight_pp: v_wpp,
            });
        }
        Ok(Self { voltage, current })
    }
}

/// A base/child cylindrical operator pair built from one CSXCAD model.
///
/// Stage 2 constructs and validates the two native-style operator instances.
/// The interface interpolation and the multi-grid engine are not wired yet.
pub struct MultiGridOperatorPair {
    pub plan: CylinderMultiGridPlan,
    pub split: GridSplit,
    pub base: crate::operator::Operator,
    pub child: crate::operator::Operator,
}

fn grid_lines(xml: &str, axis: char) -> Result<Vec<f64>, String> {
    let open = format!("<{axis}Lines");
    let close = format!("</{axis}Lines>");
    let start = xml
        .find(&open)
        .ok_or_else(|| format!("missing <{axis}Lines>"))?;
    let body_start = xml[start..]
        .find('>')
        .map(|offset| start + offset + 1)
        .ok_or_else(|| format!("malformed <{axis}Lines>"))?;
    let end = xml[body_start..]
        .find(&close)
        .map(|offset| body_start + offset)
        .ok_or_else(|| format!("missing {close}"))?;
    let body = &xml[body_start..end];
    let mut values = Vec::new();
    for token in body.split(',') {
        let value = token
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("non-numeric {axis} line {token:?}"))?;
        values.push(value);
    }
    if values.len() < 2 {
        return Err(format!("<{axis}Lines> needs at least two values"));
    }
    Ok(values)
}

fn replace_grid_lines(xml: &str, axis: char, values: &[f64]) -> Result<String, String> {
    let open = format!("<{axis}Lines");
    let close = format!("</{axis}Lines>");
    let start = xml
        .find(&open)
        .ok_or_else(|| format!("missing <{axis}Lines>"))?;
    let body_start = xml[start..]
        .find('>')
        .map(|offset| start + offset + 1)
        .ok_or_else(|| format!("malformed <{axis}Lines>"))?;
    let end = xml[body_start..]
        .find(&close)
        .map(|offset| body_start + offset)
        .ok_or_else(|| format!("missing {close}"))?;
    let body = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{}{}{}",
        &xml[..body_start],
        body,
        &xml[end..]
    ))
}

/// Build the base and child operator XMLs from the full model.
///
/// The base keeps the rho lines up to the split and every second alpha line;
/// the child starts at the split and keeps all alpha lines (native
/// `Operator_CylinderMultiGrid::SetGeometryCSX`).
pub fn split_model_xml(xml: &str, split: GridSplit) -> Result<(String, String), String> {
    let rho = grid_lines(xml, 'X')?;
    let alpha = grid_lines(xml, 'Y')?;
    if split.split_pos == 0 || split.split_pos >= rho.len() {
        return Err("split position outside the rho lines".to_string());
    }
    let base_rho = &rho[..split.split_pos];
    let base_alpha: Vec<f64> = alpha.iter().copied().step_by(2).collect();
    let child_rho = &rho[split.split_pos..];
    if base_alpha.len() < 2 || child_rho.len() < 2 {
        return Err("split leaves fewer than two lines on one side".to_string());
    }
    let base = replace_grid_lines(
        &replace_grid_lines(xml, 'X', base_rho)?,
        'Y',
        &base_alpha,
    )?;
    let child = replace_grid_lines(
        &replace_grid_lines(xml, 'X', child_rho)?,
        'Y',
        &alpha,
    )?;
    Ok((base, child))
}

impl MultiGridOperatorPair {
    pub fn from_xml(xml: &str, plan: &CylinderMultiGridPlan) -> Result<Self, String> {
        let rho = grid_lines(xml, 'X')?;
        let alpha = grid_lines(xml, 'Y')?;
        let closed = (alpha[alpha.len() - 1] - alpha[0] - std::f64::consts::TAU).abs()
            < 1e-9 * std::f64::consts::TAU.abs().max(1.0);
        let split = find_grid_split(&rho, plan.split_radius(), alpha.len(), closed)?;
        let (base_xml, child_xml) = split_model_xml(xml, split)?;
        let mut base = crate::operator::Operator::new();
        base.load_csx(&base_xml)
            .map_err(|error| format!("base operator: {error}"))?;
        let mut child = crate::operator::Operator::new();
        child.load_csx(&child_xml)
            .map_err(|error| format!("child operator: {error}"))?;
        Ok(Self {
            plan: plan.clone(),
            split,
            base,
            child,
        })
    }

    /// Cells covered by the base and child grids (native `GetNumberCells`).
    pub fn cell_counts(&self) -> (usize, usize) {
        let base = &self.base.base.num_lines;
        let child = &self.child.base.num_lines;
        (
            (base[0] * base[1] * base[2]) as usize,
            (child[0] * child[1] * child[2]) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_validates_radii_and_cascades() {
        let plan = CylinderMultiGridPlan::new(&[0.2, 0.5]).expect("valid");
        assert_eq!(plan.grid_count(), 3);
        assert_eq!(plan.split_radius(), 0.5);
        assert_eq!(plan.child_radii(), &[0.2]);

        assert!(CylinderMultiGridPlan::new(&[]).is_err());
        assert!(CylinderMultiGridPlan::new(&[0.0]).is_err());
        assert!(CylinderMultiGridPlan::new(&[0.5, 0.2]).is_err());
        assert!(CylinderMultiGridPlan::new(&[f64::NAN]).is_err());
        let too_many = vec![0.1; MULTIGRID_LIMIT + 1];
        assert!(CylinderMultiGridPlan::new(&too_many).is_err());
    }

    #[test]
    fn split_search_matches_the_native_rules() {
        let rho: Vec<f64> = (0..20).map(|v| v as f64 * 0.1).collect();
        let split = find_grid_split(&rho, 0.75, 9, false).expect("split at 0.8");
        assert_eq!(split.split_pos, 8);
        assert!((split.split_radius - 0.8).abs() < 1e-12);

        // Too close to the high end.
        assert!(find_grid_split(&rho, 1.7, 9, false).is_err());
        // Even alpha count is rejected.
        assert!(find_grid_split(&rho, 0.75, 8, false).is_err());
        // Closed alpha subtracts one plane; 9-1=8 is even -> rejected there.
        assert!(find_grid_split(&rho, 0.75, 9, true).is_err());
        // Outside the mesh.
        assert!(find_grid_split(&rho, 5.0, 9, false).is_err());
    }

    #[test]
    fn base_gets_split_rho_and_every_second_alpha() {
        let rho: Vec<f64> = (0..12).map(|v| v as f64).collect();
        let alpha: Vec<f64> = (0..9).map(|v| v as f64).collect();
        let split = find_grid_split(&rho, 3.5, 9, false).expect("split at 4");
        let lines = split_lines(&rho, &alpha, split);
        assert_eq!(lines.base_rho, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(lines.base_alpha, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(lines.child_rho, rho);
        assert_eq!(lines.child_alpha, alpha);
    }

    #[test]
    fn extension_matrix_matches_native_values() {
        for kind in [
            ExtensionKind::ConductingSheet,
            ExtensionKind::Cylinder,
            ExtensionKind::Debye,
            ExtensionKind::Excitation,
            ExtensionKind::Lorentz,
            ExtensionKind::LumpedRlc,
            ExtensionKind::SteadyState,
        ] {
            assert!(kind.multigrid_safe(false), "{} base", kind.name());
            assert!(kind.multigrid_safe(true), "{} child", kind.name());
        }
        assert!(!ExtensionKind::Tfsf.multigrid_safe(false));
        assert!(!ExtensionKind::Tfsf.multigrid_safe(true));
        assert!(ExtensionKind::Upml.multigrid_safe(false));
        assert!(!ExtensionKind::Upml.multigrid_safe(true));
        assert!(!ExtensionKind::Unknown.multigrid_safe(false));

        // Mur: z is always safe; r-top only on the base.
        assert!(ExtensionKind::Mur {
            component: 2,
            top: false
        }
        .multigrid_safe(true));
        assert!(ExtensionKind::Mur {
            component: 0,
            top: true
        }
        .multigrid_safe(false));
        assert!(!ExtensionKind::Mur {
            component: 0,
            top: true
        }
        .multigrid_safe(true));
        assert!(!ExtensionKind::Mur {
            component: 0,
            top: false
        }
        .multigrid_safe(false));
        assert!(!ExtensionKind::Mur {
            component: 1,
            top: true
        }
        .multigrid_safe(false));

        assert!(check_extensions(&[(ExtensionKind::Excitation, true)]).is_ok());
        let error = check_extensions(&[(ExtensionKind::Tfsf, false)]).expect_err("TFSF");
        assert!(error.contains("TFSF") && error.contains("base"), "{error}");
    }

    #[test]
    fn pair_builds_restricted_base_and_child_operators() {
        // 13 rho lines, closed alpha with 10 lines (9 after the wrap).
        let rho = (0..13).map(|v| (v as f64 * 0.1).to_string()).collect::<Vec<_>>().join(",");
        let alpha = (0..10)
            .map(|v| (v as f64 * std::f64::consts::TAU / 9.0).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="1">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{rho}</XLines>
                    <YLines>{alpha}</YLines>
                    <ZLines>0,1,2,3,4</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure>"#
        );
        let plan = CylinderMultiGridPlan::new(&[0.55]).expect("plan");
        let pair = MultiGridOperatorPair::from_xml(&xml, &plan).expect("pair");
        assert_eq!(pair.split.split_pos, 6);
        assert_eq!(pair.base.base.num_lines[0], 6);
        assert_eq!(pair.base.base.num_lines[1], 5);
        assert_eq!(pair.child.base.num_lines[0], 7);
        // CSXCAD pads a closed alpha mesh with the wrap line.
        assert_eq!(pair.child.base.num_lines[1], 11);
        let (base_cells, child_cells) = pair.cell_counts();
        assert!(base_cells > 0 && child_cells > 0);
    }

    #[test]
    fn alpha_interpolation_tables_match_the_native_positions() {
        let base: Vec<f64> = (0..5).map(|i| i as f64 * 2.0).collect();
        let child: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let table = AlphaInterpolation::new(&base, &child, true).expect("table");
        assert_eq!(table.voltage.len(), 5);
        assert_eq!(table.current.len(), 5);
        // Current: p = a/2, pp = a/2 + odd.
        assert_eq!(table.current[0].pos_p, 0);
        assert_eq!(table.current[0].pos_pp, 0);
        assert_eq!(table.current[1].pos_p, 0);
        assert_eq!(table.current[1].pos_pp, 1);
        // Voltage: pp = a/2 + odd - even; index 0 wraps to child_len-3.
        assert_eq!(table.voltage[0].pos_pp, 7);
        assert_eq!(table.voltage[1].pos_pp, 1);
        for pair in table.voltage.iter().chain(table.current.iter()) {
            assert!(pair.weight_p.is_finite() && pair.weight_pp.is_finite());
            assert!(pair.pos_p < child.len() && pair.pos_pp < child.len());
        }
    }
}
