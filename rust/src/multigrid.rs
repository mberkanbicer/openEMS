//! Cylindrical multi-grid operator plan and engine (E-007).
//!
//! Mirrors the native `Operator_CylinderMultiGrid` split/level rules, the
//! `Operator_Extension::IsCylindricalMultiGridSave` compatibility matrix and
//! the `Engine_CylinderMultiGrid` barrier order. The engine is implemented and
//! unit-tested here, but `setup_fdtd` still rejects multi-grid radii until the
//! high-level extension/probe wiring and the `coax_cylindrical` validation are
//! in place.

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

/// Child-to-base interpolation tables (native
/// `Operator_CylinderMultiGrid::SetupInterpolation`).
///
/// The base grid keeps the full (fine) azimuthal mesh; the child grid spans the
/// inner rho range with every second alpha line. Entries are indexed by base
/// alpha line, exactly as in the native tables. The native weights include the
/// dual line-length ratios because the field arrays store line-integrated
/// voltages and loop currents, not the bare field components.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaInterpolation {
    /// alpha component (native table index 1)
    pub voltage: Vec<InterpolationPair>,
    pub current: Vec<InterpolationPair>,
    /// rho/z components (native table index 0)
    pub voltage_rz: Vec<InterpolationPair>,
    pub current_rz: Vec<InterpolationPair>,
}

/// Native `Operator::GetDiscLine` for the azimuthal direction.
fn alpha_line(lines: &[f64], index: usize, dual: bool) -> f64 {
    if !dual {
        lines[index]
    } else if index + 1 < lines.len() {
        0.5 * (lines[index] + lines[index + 1])
    } else {
        lines[index] + 0.5 * (lines[index] - lines[index - 1])
    }
}

/// Native `Operator::GetDiscDelta` for the azimuthal direction.
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

/// Number of physical azimuthal lines. A closed mesh carries one extra padding
/// line for the dual-mesh seam; that plane is not part of the native mesh.
fn logical_alpha_count(lines: &[f64], closed: bool) -> usize {
    if closed {
        lines.len().saturating_sub(1)
    } else {
        lines.len()
    }
}

impl AlphaInterpolation {
    /// Build the native tables from the base and child azimuthal lines.
    ///
    /// `closed` marks a full `2*pi` range; both meshes then carry the padding
    /// line of the seam, which is excluded from the table entries.
    pub fn new(base: &[f64], child: &[f64], closed: bool) -> Result<Self, String> {
        if base.len() < 3 || child.len() < 3 {
            return Err("interpolation needs at least three base and child lines".to_string());
        }
        let n_base = logical_alpha_count(base, closed);
        let n_child = logical_alpha_count(child, closed);
        if n_base < 3 || n_child < 2 {
            return Err(format!(
                "interpolation needs at least three base and two child alpha lines (base {n_base}, child {n_child})"
            ));
        }
        let index = |value: isize, what: &str| -> Result<usize, String> {
            if value < 0 || value as usize >= n_child {
                return Err(format!(
                    "{what} child alpha index {value} is outside 0..{n_child}"
                ));
            }
            Ok(value as usize)
        };

        let mut voltage = Vec::with_capacity(base.len());
        let mut current = Vec::with_capacity(base.len());
        let mut voltage_rz = Vec::with_capacity(base.len());
        let mut current_rz = Vec::with_capacity(base.len());

        for a_n in 0..n_base {
            let is_odd = (a_n % 2) as isize;
            let is_even = 1 - is_odd;

            // --- current, rho/z (native table [0]) ---
            let i_p = (a_n / 2) as isize;
            let mut i_pp = (a_n / 2) as isize + is_odd - is_even;
            if a_n == 0 && closed {
                i_pp = n_child as isize - 3;
            } else if a_n == 0 && !closed {
                i_pp = 0;
            }
            if !closed && a_n + 2 == n_base {
                i_pp = a_n as isize / 2 - 1;
            }
            if !closed && a_n + 1 == n_base {
                i_pp = i_p;
            }
            let ip = index(i_p, "current rho/z p")?;
            let ipp = index(i_pp, "current rho/z pp")?;
            let dl_p = alpha_line(child, ip, true);
            let mut dl_pp = alpha_line(child, ipp, true);
            if a_n == 0 && closed {
                dl_pp -= std::f64::consts::TAU;
            }
            let this_line = alpha_line(base, a_n, true);
            let (wp, wpp) = if ip == ipp {
                (1.0, 0.0)
            } else {
                (
                    (dl_pp - this_line) / (dl_pp - dl_p),
                    (this_line - dl_p) / (dl_pp - dl_p),
                )
            };
            current_rz.push(InterpolationPair {
                pos_p: ip,
                pos_pp: ipp,
                weight_p: wp,
                weight_pp: wpp,
            });

            // --- voltage, rho/z (native table [0]) ---
            let v_p = (a_n / 2) as isize;
            let v_pp = (a_n / 2) as isize + is_odd;
            let vp = index(v_p, "voltage rho/z p")?;
            let vpp = index(v_pp, "voltage rho/z pp")?;
            let dl_p = alpha_line(child, vp, false);
            let dl_pp = alpha_line(child, vpp, false);
            let this_line = alpha_line(base, a_n, false);
            let (wp, wpp) = if vp == vpp {
                (1.0, 0.0)
            } else {
                (
                    (dl_pp - this_line) / (dl_pp - dl_p),
                    (this_line - dl_p) / (dl_pp - dl_p),
                )
            };
            voltage_rz.push(InterpolationPair {
                pos_p: vp,
                pos_pp: vpp,
                weight_p: wp,
                weight_pp: wpp,
            });

            // --- current, alpha (native table [1]) ---
            let mut c_p = (a_n / 2) as isize;
            let mut c_pp = (a_n / 2) as isize + is_odd;
            if !closed && a_n == 1 {
                c_p = 2;
            }
            if !closed && a_n + 2 == n_base {
                c_pp = a_n as isize / 2 - 1;
            }
            let cp = index(c_p, "current alpha p")?;
            let cpp = index(c_pp, "current alpha pp")?;
            let base_delta = alpha_delta(base, a_n, true);
            let (wp, wpp) = if cp == cpp {
                (base_delta / alpha_delta(child, cp, true), 0.0)
            } else {
                let dl_p = alpha_line(child, cp, false);
                let dl_pp = alpha_line(child, cpp, false);
                let this_line = alpha_line(base, a_n, false);
                (
                    (dl_pp - this_line) / (dl_pp - dl_p)
                        * base_delta
                        / alpha_delta(child, cp, true),
                    (this_line - dl_p) / (dl_pp - dl_p)
                        * base_delta
                        / alpha_delta(child, cpp, true),
                )
            };
            current.push(InterpolationPair {
                pos_p: cp,
                pos_pp: cpp,
                weight_p: wp,
                weight_pp: wpp,
            });

            // --- voltage, alpha (native table [1]) ---
            let mut v_p = (a_n / 2) as isize;
            let mut v_pp = (a_n / 2) as isize + is_odd - is_even;
            if a_n == 0 && closed {
                v_pp = n_child as isize - 3;
            } else if a_n == 0 && !closed {
                v_pp = 1;
            }
            if !closed && a_n + 2 == n_base {
                v_pp = a_n as isize / 2 - 1;
            }
            if !closed && a_n + 1 == n_base {
                v_p = 0;
                v_pp = 0;
            }
            let vp = index(v_p, "voltage alpha p")?;
            let vpp = index(v_pp, "voltage alpha pp")?;
            let base_delta = alpha_delta(base, a_n, false);
            let (wp, wpp) = if vp == vpp {
                (0.0, 0.0)
            } else {
                let dl_p = alpha_line(child, vp, true);
                let dl_pp = alpha_line(child, vpp, true);
                let this_line = alpha_line(base, a_n, true);
                (
                    (dl_pp - this_line) / (dl_pp - dl_p)
                        * base_delta
                        / alpha_delta(child, vp, false),
                    (this_line - dl_p) / (dl_pp - dl_p)
                        * base_delta
                        / alpha_delta(child, vpp, false),
                )
            };
            voltage.push(InterpolationPair {
                pos_p: vp,
                pos_pp: vpp,
                weight_p: wp,
                weight_pp: wpp,
            });
        }

        // A closed base mesh carries one extra padding plane beyond the seam.
        // It is not a physical plane, but the field array still has it; mirror
        // the seam entry so a full-array interpolation never reads stale data.
        if base.len() > n_base {
            for table in [&mut voltage, &mut current, &mut voltage_rz, &mut current_rz] {
                if let Some(last) = table.last().copied() {
                    table.push(last);
                }
            }
        }

        Ok(Self {
            voltage,
            current,
            voltage_rz,
            current_rz,
        })
    }
}

/// Apply the child-to-base alpha interpolation for one rho row and z index.
///
/// Mirrors native `Engine_CylinderMultiGrid::InterpolVoltChild2Base` /
/// `InterpolCurrChild2Base`. `alpha_count` is the number of physical base
/// azimuthal lines (the padding plane of a closed mesh is skipped). The rho
/// index is shared: the child holds the same rho lines up to the split.
pub fn interpolate_child_to_base(
    base: &mut crate::array_nijk::FieldArray,
    child: &crate::array_nijk::FieldArray,
    rho: usize,
    z: usize,
    alpha_count: usize,
    table: &AlphaInterpolation,
    voltage: bool,
) -> Result<(), String> {
    let (bj, bk) = (base.nj(), base.nk());
    let (cj, ck) = (child.nj(), child.nk());
    if rho >= base.ni() || rho >= child.ni() || z >= bk || z >= ck {
        return Err("interpolation index outside the grids".to_string());
    }
    if alpha_count > bj {
        return Err("interpolation alpha count outside the base grid".to_string());
    }
    let (alpha, rz) = if voltage {
        (&table.voltage, &table.voltage_rz)
    } else {
        (&table.current, &table.current_rz)
    };
    if alpha.len() < alpha_count || rz.len() < alpha_count {
        return Err("interpolation table too short".to_string());
    }
    for j in 0..alpha_count {
        let pair = alpha[j];
        if pair.pos_p >= cj || pair.pos_pp >= cj {
            return Err("interpolation child alpha index outside the grid".to_string());
        }
        let value = pair.weight_p * *child.get(1, rho, pair.pos_p, z) as f64
            + pair.weight_pp * *child.get(1, rho, pair.pos_pp, z) as f64;
        *base.get_mut(1, rho, j, z) = value as crate::constants::FDTD_FLOAT;

        let pair = rz[j];
        if pair.pos_p >= cj || pair.pos_pp >= cj {
            return Err("interpolation child alpha index outside the grid".to_string());
        }
        for n in [0usize, 2] {
            let value = pair.weight_p * *child.get(n, rho, pair.pos_p, z) as f64
                + pair.weight_pp * *child.get(n, rho, pair.pos_pp, z) as f64;
            *base.get_mut(n, rho, j, z) = value as crate::constants::FDTD_FLOAT;
        }
    }
    Ok(())
}

/// A base/child cylindrical operator pair built from one CSXCAD model.
///
/// The base keeps the full mesh; the child spans the inner rho range with
/// every second alpha line (native `SetGeometryCSX`). The engine that drives
/// the pair is [`MultiGridEngine`].
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
/// Native `Operator_CylinderMultiGrid::SetGeometryCSX`: the base keeps the
/// full rho/alpha mesh, while the child (inner operator) gets the rho lines up
/// to the split and every second alpha line.
pub fn split_model_xml(xml: &str, split: GridSplit) -> Result<(String, String), String> {
    let rho = grid_lines(xml, 'X')?;
    let alpha = grid_lines(xml, 'Y')?;
    if split.split_pos == 0 || split.split_pos >= rho.len() {
        return Err("split position outside the rho lines".to_string());
    }
    let child_rho = &rho[..split.split_pos];
    let child_alpha: Vec<f64> = alpha.iter().copied().step_by(2).collect();
    if child_alpha.len() < 2 || child_rho.len() < 2 {
        return Err("split leaves fewer than two lines on one side".to_string());
    }
    let base = xml.to_string();
    let child = replace_grid_lines(
        &replace_grid_lines(xml, 'X', child_rho)?,
        'Y',
        &child_alpha,
    )?;
    Ok((base, child))
}

impl MultiGridOperatorPair {
    pub fn from_xml(xml: &str, plan: &CylinderMultiGridPlan) -> Result<Self, String> {
        let rho = grid_lines(xml, 'X')?;
        let alpha = grid_lines(xml, 'Y')?;
        let closed = (alpha[alpha.len() - 1] - alpha[0] - std::f64::consts::TAU).abs()
            < 1e-9 * std::f64::consts::TAU.abs().max(1.0);
        // Native validates the operator line count, which includes the extra
        // padding line of a closed azimuth.
        let alpha_lines = alpha.len() + usize::from(closed);
        let split = find_grid_split(&rho, plan.split_radius(), alpha_lines, closed)?;
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

/// The cylindrical multi-grid engine (native `Engine_CylinderMultiGrid`).
///
/// The base engine owns the full mesh but only evolves the rho rows from
/// `split_pos - 1` outwards; the inner engine evolves the child mesh. One time
/// step runs the native barrier order:
///
/// 1. base and child voltage half-steps,
/// 2. the interface voltage sync (base row `split_pos - 1` -> child),
/// 3. base and child current half-steps,
/// 4. the interface current sync (child row `split_pos - 2` -> base),
/// 5. the child-to-base interpolation of the overlap rows.
///
/// Native runs the two engines on separate threads and joins them at the
/// barriers. The updates never read the other engine's fields during a phase,
/// so running the phases sequentially gives the same field results.
pub struct MultiGridEngine {
    pub base: crate::engine::Engine,
    pub inner: crate::engine::Engine,
    pub split_pos: usize,
    pub interpolation: AlphaInterpolation,
    closed_alpha: bool,
    /// Physical base alpha lines (the closed-mesh padding plane is excluded).
    alpha_count: usize,
}

impl MultiGridEngine {
    /// Build the engine pair from a validated operator pair, deriving the
    /// timestep from the base operator (native `CalcTimestep`).
    pub fn from_pair(
        pair: MultiGridOperatorPair,
        num_threads: usize,
    ) -> Result<Self, String> {
        Self::build(pair, None, num_threads)
    }

    /// Build the engine pair with an explicit timestep (native shared `dT`).
    pub fn from_pair_with_timestep(
        pair: MultiGridOperatorPair,
        timestep: f64,
        num_threads: usize,
    ) -> Result<Self, String> {
        if !timestep.is_finite() || timestep <= 0.0 {
            return Err(format!(
                "multi-grid timestep must be positive, got {timestep}"
            ));
        }
        Self::build(pair, Some(timestep), num_threads)
    }

    fn build(
        pair: MultiGridOperatorPair,
        timestep: Option<f64>,
        num_threads: usize,
    ) -> Result<Self, String> {
        let MultiGridOperatorPair {
            plan: _,
            split,
            mut base,
            mut child,
        } = pair;
        let split_pos = split.split_pos;
        if split_pos < 2 {
            return Err("multi-grid split must leave at least two rho lines".to_string());
        }
        let closed = base.closed_alpha();

        let base_rho = base
            .base
            .disc_lines[0]
            .clone()
            .ok_or("base rho lines missing")?;
        let child_rho = child
            .base
            .disc_lines[0]
            .clone()
            .ok_or("child rho lines missing")?;
        if child_rho.len() != split_pos
            || base_rho.len() < split_pos
            || base_rho[..split_pos]
                .iter()
                .zip(child_rho.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12)
        {
            return Err(
                "child rho lines must match the first split_pos base lines".to_string(),
            );
        }

        // Native `Operator_CylinderMultiGrid::SetupCSXGrid` forces the larger
        // variable step: the rho = 0 singularity is never updated by the base.
        base.base.m_time_step_var = 3;

        // The EC coefficients of the first pass only need the materials; the
        // step comes from them, then both operators are recomputed with the
        // shared timestep.
        base.calc_ec_operator();
        match timestep {
            Some(dt) => base.base.d_t = dt,
            None => {
                base.calc_timestep();
            }
        }
        if !base.base.d_t.is_finite() || base.base.d_t <= 0.0 {
            return Err(format!(
                "multi-grid base timestep is invalid: {}",
                base.base.d_t
            ));
        }
        base.calc_ec_operator();
        child.base.d_t = base.base.d_t;
        child.calc_ec_operator();

        // Native `openEMS::SetupFDTD` order: BCs after the coefficients, then
        // the rho = 0 extension coefficients.
        base.apply_electric_bc();
        base.apply_magnetic_bc();
        child.apply_electric_bc();
        child.apply_magnetic_bc();
        base.build_r0_coefficients();
        child.build_r0_coefficients();

        // Native computes the base EC only over rho >= split_pos - 1; the
        // overlap rows are supplied by the child interpolation.
        base.zero_ec_below_rho(split_pos - 1);

        let base_alpha = base
            .base
            .disc_lines[1]
            .clone()
            .ok_or("base alpha lines missing")?;
        let child_alpha = child
            .base
            .disc_lines[1]
            .clone()
            .ok_or("child alpha lines missing")?;
        let interpolation = AlphaInterpolation::new(&base_alpha, &child_alpha, closed)?;
        let alpha_count = logical_alpha_count(&base_alpha, closed);
        if alpha_count < 3 {
            return Err("multi-grid needs at least three alpha lines".to_string());
        }
        if child.base.num_lines[0] as usize != split_pos {
            return Err(format!(
                "child rho count {} does not match the split position {split_pos}",
                child.base.num_lines[0]
            ));
        }

        let base_extension = base.build_cylinder_extension();
        let child_extension = child.build_cylinder_extension();

        let mut base_engine = crate::engine::Engine::new(Box::new(base));
        base_engine.set_number_of_threads(num_threads as i32);
        if let Some(extension) = base_extension {
            base_engine.add_extension(Box::new(extension));
        }
        let mut inner_engine = crate::engine::Engine::new(Box::new(child));
        inner_engine.set_number_of_threads(num_threads as i32);
        if let Some(extension) = child_extension {
            inner_engine.add_extension(Box::new(extension));
        }

        Ok(Self {
            base: base_engine,
            inner: inner_engine,
            split_pos,
            interpolation,
            closed_alpha: closed,
            alpha_count,
        })
    }

    /// Advance the two engines by the requested number of native time steps.
    pub fn iterate_ts(&mut self, steps: u32) -> bool {
        for _ in 0..steps {
            let number = self.base.num_ts;
            for extension in self.base.extensions.iter_mut() {
                extension.set_timestep(number);
            }
            for extension in self.inner.extensions.iter_mut() {
                extension.set_timestep(number);
            }

            // Voltage half-step: both engines, then the interface sync.
            self.base.do_pre_voltage_updates();
            self.base.update_voltages();
            self.base.do_post_voltage_updates();
            self.base.apply2_voltages();

            self.inner.do_pre_voltage_updates();
            self.inner.update_voltages();
            self.inner.do_post_voltage_updates();
            self.inner.apply2_voltages();

            self.sync_voltages();

            // Current half-step.
            self.base.do_pre_current_updates();
            self.base.update_currents();
            self.base.do_post_current_updates();
            self.base.apply2_current();

            self.inner.do_pre_current_updates();
            self.inner.update_currents();
            self.inner.do_post_current_updates();
            self.inner.apply2_current();

            self.sync_currents();

            // End of step: fill the base overlap from the child.
            for rho in 0..self.split_pos - 1 {
                let _ = self.interpolate_row(rho, true);
            }
            for rho in 0..self.split_pos.saturating_sub(2) {
                let _ = self.interpolate_row(rho, false);
            }

            self.base.num_ts += 1;
            self.inner.num_ts += 1;
        }
        true
    }

    /// Native `Engine_Ext_CylinderMultiGrid::SyncVoltages`.
    pub fn sync_voltages(&mut self) {
        if self.split_pos == 0 {
            return;
        }
        let rho = self.split_pos - 1;
        let nz = self.base.num_lines[2].min(self.inner.num_lines[2]);
        let inner_ny = self.inner.num_lines[1];
        // Native loops `a < numLines[1] - 1`, which skips the duplicated seam
        // plane of a closed mesh.
        for a in (0..self.alpha_count.saturating_sub(1)).step_by(2) {
            let h = a / 2;
            if h >= inner_ny {
                break;
            }
            let a_next = (a + 1).min(self.alpha_count - 1);
            for z in 0..nz {
                // Native `SyncVoltages` zeroes the normal (rho) component of
                // the child at the interface and copies the tangential ones.
                *self.inner.volt_ptr.get_mut(0, rho, h, z) = 0.0;
                *self.inner.volt_ptr.get_mut(2, rho, h, z) =
                    *self.base.volt_ptr.get(2, rho, a, z);
                *self.inner.volt_ptr.get_mut(1, rho, h, z) =
                    *self.base.volt_ptr.get(1, rho, a, z)
                        + *self.base.volt_ptr.get(1, rho, a_next, z);
            }
        }
    }

    /// Native `Engine_Ext_CylinderMultiGrid::SyncCurrents`.
    pub fn sync_currents(&mut self) {
        if self.split_pos < 2 {
            return;
        }
        let rho = self.split_pos - 2;
        let _ = self.interpolate_row(rho, false);
    }

    /// Interpolate one rho row of the child into the base overlap.
    pub fn interpolate_row(&mut self, rho: usize, voltage: bool) -> Result<(), String> {
        let nz = self.base.num_lines[2].min(self.inner.num_lines[2]);
        for z in 0..nz {
            if voltage {
                interpolate_child_to_base(
                    &mut self.base.volt_ptr,
                    &self.inner.volt_ptr,
                    rho,
                    z,
                    self.alpha_count,
                    &self.interpolation,
                    true,
                )?;
            } else {
                interpolate_child_to_base(
                    &mut self.base.curr_ptr,
                    &self.inner.curr_ptr,
                    rho,
                    z,
                    self.alpha_count,
                    &self.interpolation,
                    false,
                )?;
            }
        }
        Ok(())
    }

    /// Whether the base mesh closes the full azimuth.
    pub fn closed_alpha(&self) -> bool {
        self.closed_alpha
    }

    /// Timestep shared by both grids.
    pub fn timestep(&self) -> f64 {
        self.base.op.get_timestep()
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

    fn mesh_xml(rho: &[f64], alpha: &[f64], z: &[f64]) -> String {
        let join = |values: &[f64]| {
            values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<openEMS><FDTD NumberOfTimesteps="20"/><ContinuousStructure CoordSystem="1">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure></openEMS>"#,
            join(rho),
            join(alpha),
            join(z)
        )
    }

    #[test]
    fn pair_keeps_the_full_base_and_splits_the_child() {
        // 13 rho lines, closed alpha with 9 input lines (8 intervals).
        let rho: Vec<f64> = (0..13).map(|v| v as f64 * 0.1).collect();
        let alpha: Vec<f64> = (0..9)
            .map(|v| v as f64 * std::f64::consts::TAU / 8.0)
            .collect();
        let z: Vec<f64> = (0..5).map(|v| v as f64).collect();
        let xml = mesh_xml(&rho, &alpha, &z);
        let plan = CylinderMultiGridPlan::new(&[0.55]).expect("plan");
        let pair = MultiGridOperatorPair::from_xml(&xml, &plan).expect("pair");
        assert_eq!(pair.split.split_pos, 6);
        // Base keeps the full mesh (plus the closed-alpha padding line).
        assert_eq!(pair.base.base.num_lines[0], 13);
        assert_eq!(pair.base.base.num_lines[1], 10);
        // Child: rho up to the split and every second alpha line (5 + pad).
        assert_eq!(pair.child.base.num_lines[0], 6);
        assert_eq!(pair.child.base.num_lines[1], 6);
        assert_eq!(pair.child.base.num_lines[2], 5);
        let (base_cells, child_cells) = pair.cell_counts();
        assert!(base_cells > child_cells);
    }

    #[test]
    fn alpha_interpolation_tables_match_the_native_positions() {
        // Open alpha: base 9 lines, child every second line (5).
        let base: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let child: Vec<f64> = (0..9).step_by(2).map(|i| i as f64).collect();
        let table = AlphaInterpolation::new(&base, &child, false).expect("table");
        assert_eq!(table.voltage.len(), 9);

        // Current alpha: p = a/2, pp = a/2 + odd; the open first line maps to
        // p = 2 and the last line collapses onto p = pp.
        assert_eq!((table.current[0].pos_p, table.current[0].pos_pp), (0, 0));
        assert_eq!((table.current[1].pos_p, table.current[1].pos_pp), (2, 1));
        assert_eq!((table.current[8].pos_p, table.current[8].pos_pp), (4, 4));

        // Voltage alpha: pp = a/2 + odd - even, with the open seam rules.
        assert_eq!((table.voltage[0].pos_p, table.voltage[0].pos_pp), (0, 1));
        assert_eq!((table.voltage[1].pos_p, table.voltage[1].pos_pp), (0, 1));
        assert_eq!((table.voltage[8].pos_p, table.voltage[8].pos_pp), (0, 0));

        // rho/z current table: even base lines interpolate between the two
        // neighbouring child lines; odd base lines sit on a child line.
        assert_eq!((table.current_rz[0].pos_p, table.current_rz[0].pos_pp), (0, 0));
        assert_eq!((table.current_rz[2].pos_p, table.current_rz[2].pos_pp), (1, 0));
        // The base dual line 2 sits three quarters of the way from child dual
        // line 1 to child dual line 0, so the weights are 0.75 / 0.25 and sum
        // to one for an offset-free interpolation.
        assert!((table.current_rz[2].weight_p - 0.75).abs() < 1e-12);
        assert!((table.current_rz[2].weight_pp - 0.25).abs() < 1e-12);
        assert!(
            (table.current_rz[2].weight_p + table.current_rz[2].weight_pp - 1.0).abs() < 1e-12
        );

        for pair in table
            .voltage
            .iter()
            .chain(table.current.iter())
            .chain(table.voltage_rz.iter())
            .chain(table.current_rz.iter())
        {
            assert!(pair.weight_p.is_finite() && pair.weight_pp.is_finite());
            assert!(pair.pos_p < child.len() && pair.pos_pp < child.len());
        }
    }

    #[test]
    fn child_to_base_interpolation_applies_the_weights_and_indexing() {
        let pair = InterpolationPair {
            pos_p: 0,
            pos_pp: 1,
            weight_p: 0.25,
            weight_pp: 0.75,
        };
        let table = AlphaInterpolation {
            voltage: vec![pair, pair],
            current: vec![pair, pair],
            voltage_rz: vec![pair, pair],
            current_rz: vec![pair, pair],
        };
        let mut base = crate::array_nijk::FieldArray::new("base", [2, 2, 1]);
        let mut child = crate::array_nijk::FieldArray::new("child", [2, 2, 1]);
        for n in 0..3 {
            for i in 0..2 {
                for j in 0..2 {
                    *child.get_mut(n, i, j, 0) = 2.0;
                }
            }
        }
        interpolate_child_to_base(&mut base, &child, 0, 0, 2, &table, true)
            .expect("interpolate");
        // 0.25 * 2 + 0.75 * 2 = 2 for every component.
        for n in 0..3 {
            for j in 0..2 {
                assert!((*base.get(n, 0, j, 0) - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn sync_voltages_matches_the_native_formula() {
        let rho: Vec<f64> = (0..13).map(|v| v as f64 * 0.1).collect();
        let alpha: Vec<f64> = (0..9)
            .map(|v| v as f64 * std::f64::consts::TAU / 8.0)
            .collect();
        let z: Vec<f64> = (0..5).map(|v| v as f64).collect();
        let xml = mesh_xml(&rho, &alpha, &z);
        let plan = CylinderMultiGridPlan::new(&[0.55]).expect("plan");
        let pair = MultiGridOperatorPair::from_xml(&xml, &plan).expect("pair");
        let mut engine = MultiGridEngine::from_pair(pair, 1).expect("engine");
        let split_pos = engine.split_pos;
        let rho_row = split_pos - 1;

        for a in 0..engine.base.num_lines[1] {
            for k in 0..engine.base.num_lines[2] {
                *engine.base.volt_ptr.get_mut(0, rho_row, a, k) = 10.0 + a as f32;
                *engine.base.volt_ptr.get_mut(1, rho_row, a, k) = 100.0 + a as f32;
                *engine.base.volt_ptr.get_mut(2, rho_row, a, k) = 1000.0 + a as f32;
            }
        }
        engine.sync_voltages();

        let h = 2usize;
        let a = 2 * h;
        let k = 1usize;
        assert_eq!(*engine.inner.volt_ptr.get(0, rho_row, h, k), 0.0);
        assert_eq!(
            *engine.inner.volt_ptr.get(2, rho_row, h, k),
            *engine.base.volt_ptr.get(2, rho_row, a, k)
        );
        assert_eq!(
            *engine.inner.volt_ptr.get(1, rho_row, h, k),
            *engine.base.volt_ptr.get(1, rho_row, a, k)
                + *engine.base.volt_ptr.get(1, rho_row, a + 1, k)
        );
    }

    /// Seed a physical TEM pulse on the stored (line-integrated) field arrays.
    fn seed_tem(
        volt: &mut crate::array_nijk::FieldArray,
        curr: &mut crate::array_nijk::FieldArray,
        op: &dyn crate::operator_base::OperatorBase,
        counts: [usize; 3],
    ) {
        let delta = op.get_grid_delta();
        let rho: Vec<f64> = (0..counts[0])
            .map(|i| op.get_disc_line(0, i as u32, false) * delta)
            .collect();
        let alpha: Vec<f64> = (0..counts[1])
            .map(|j| op.get_disc_line(1, j as u32, false))
            .collect();
        let z: Vec<f64> = (0..counts[2])
            .map(|k| op.get_disc_line(2, k as u32, false) * delta)
            .collect();
        let z_max = *z.last().expect("z lines");
        let z0 = 0.5 * z_max;
        let width = 0.12 * z_max;
        for i in 1..counts[0] {
            let dr = rho[i] - rho[i - 1];
            for j in 0..counts[1] {
                let dual = |index: usize| {
                    if index + 1 < counts[1] {
                        0.5 * (alpha[index] + alpha[index + 1])
                    } else {
                        alpha[index] + 0.5 * (alpha[index] - alpha[index - 1])
                    }
                };
                let da = if j == 0 {
                    alpha[1] - alpha[0]
                } else {
                    dual(j) - dual(j - 1)
                };
                for k in 0..counts[2] {
                    let profile = (-((z[k] - z0) / width).powi(2)).exp();
                    let e_rho = profile / rho[i];
                    *volt.get_mut(0, i, j, k) = (e_rho * dr) as crate::constants::FDTD_FLOAT;
                    *curr.get_mut(1, i, j, k) =
                        (e_rho / crate::constants::Z0 * rho[i] * da)
                            as crate::constants::FDTD_FLOAT;
                }
            }
        }
    }

    #[test]
    fn multi_grid_tem_pulse_matches_the_single_grid_reference() {
        // Closed alpha: the TEM mode is azimuthally uniform, so the coarse
        // child mesh can reproduce it and the seam extension is active.
        let rho: Vec<f64> = (0..10).map(|v| (v as f64 + 1.0) * 0.1).collect();
        let alpha: Vec<f64> = (0..9)
            .map(|v| v as f64 * std::f64::consts::TAU / 8.0)
            .collect();
        let z: Vec<f64> = (0..10).map(|v| v as f64 * 0.1).collect();
        let xml = mesh_xml(&rho, &alpha, &z);

        let plan = CylinderMultiGridPlan::new(&[0.55]).expect("plan");
        let pair = MultiGridOperatorPair::from_xml(&xml, &plan).expect("pair");
        let mut multi = MultiGridEngine::from_pair(pair, 1).expect("multi-grid engine");
        let dt = multi.timestep();

        let mut reference = crate::operator::Operator::new();
        reference.load_csx(&xml).expect("reference operator");
        reference.base.d_t = dt;
        reference.calc_ec_operator();
        reference.apply_electric_bc();
        reference.apply_magnetic_bc();
        reference.build_r0_coefficients();
        let reference_extension = reference.build_cylinder_extension();
        let mut reference_engine = crate::engine::Engine::new(Box::new(reference));
        if let Some(extension) = reference_extension {
            reference_engine.add_extension(Box::new(extension));
        }
        assert_eq!(reference_engine.num_lines, multi.base.num_lines);

        seed_tem(
            &mut multi.base.volt_ptr,
            &mut multi.base.curr_ptr,
            multi.base.op.as_ref(),
            multi.base.num_lines,
        );
        seed_tem(
            &mut multi.inner.volt_ptr,
            &mut multi.inner.curr_ptr,
            multi.inner.op.as_ref(),
            multi.inner.num_lines,
        );
        seed_tem(
            &mut reference_engine.volt_ptr,
            &mut reference_engine.curr_ptr,
            reference_engine.op.as_ref(),
            reference_engine.num_lines,
        );

        // The native interface coupling is first-order; compare the first
        // steps, where the pulse has not yet reached the cavity ends.
        multi.iterate_ts(5);
        reference_engine.iterate_ts(5);

        let [nx, ny, nz] = multi.base.num_lines;
        let mut max_volt_ref: f64 = 0.0;
        let mut max_volt_diff: f64 = 0.0;
        let mut max_curr_ref: f64 = 0.0;
        let mut max_curr_diff: f64 = 0.0;
        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let a = *multi.base.volt_ptr.get(n, i, j, k) as f64;
                        let b = *reference_engine.volt_ptr.get(n, i, j, k) as f64;
                        max_volt_ref = max_volt_ref.max(b.abs());
                        max_volt_diff = max_volt_diff.max((a - b).abs());
                        let a = *multi.base.curr_ptr.get(n, i, j, k) as f64;
                        let b = *reference_engine.curr_ptr.get(n, i, j, k) as f64;
                        max_curr_ref = max_curr_ref.max(b.abs());
                        max_curr_diff = max_curr_diff.max((a - b).abs());
                    }
                }
            }
        }
        let volt_error = max_volt_diff / max_volt_ref.max(f64::MIN_POSITIVE);
        let curr_error = max_curr_diff / max_curr_ref.max(f64::MIN_POSITIVE);
        println!(
            "multi-grid TEM parity after 5 steps: voltage {volt_error:.3e}, current {curr_error:.3e}"
        );
        assert!(
            volt_error < 1.5e-2,
            "multi-grid TEM voltage differs from the reference: {volt_error:.3e}"
        );
        assert!(
            curr_error < 5.0e-2,
            "multi-grid TEM current differs from the reference: {curr_error:.3e}"
        );

        // The TEM pulse travels at the grid speed: the multigrid peak must sit
        // on the reference peak (within one cell) in the outer region.
        let peak = |engine: &crate::engine::Engine| -> usize {
            let mut best_k = 0usize;
            let mut best = 0.0f64;
            for k in 0..nz {
                let value = (*engine.volt_ptr.get(0, multi.split_pos, 0, k) as f64).abs();
                if value > best {
                    best = value;
                    best_k = k;
                }
            }
            best_k
        };
        let multi_peak = peak(&multi.base);
        let reference_peak = peak(&reference_engine);
        println!(
            "multi-grid TEM peak: multi z={multi_peak}, reference z={reference_peak}"
        );
        assert!(
            multi_peak.abs_diff(reference_peak) <= 1,
            "multi-grid TEM peak at z={multi_peak}, reference at z={reference_peak}"
        );

        // The radial (coaxial) voltage profile survives the handover: the
        // cumulative E_rho integral across the interface follows the
        // single-grid reference profile.
        let mut volts_multi = Vec::new();
        let mut volts_reference = Vec::new();
        let mut acc_multi = 0.0f64;
        let mut acc_reference = 0.0f64;
        for i in 1..nx {
            acc_multi += *multi.base.volt_ptr.get(0, i, 0, multi_peak) as f64;
            acc_reference += *reference_engine.volt_ptr.get(0, i, 0, multi_peak) as f64;
            volts_multi.push(acc_multi);
            volts_reference.push(acc_reference);
        }
        let norm_multi = *volts_multi.last().expect("profile");
        let norm_reference = *volts_reference.last().expect("profile");
        let profile_error = volts_multi
            .iter()
            .zip(volts_reference.iter())
            .map(|(a, b)| ((a / norm_multi) - (b / norm_reference)).abs())
            .fold(0.0f64, f64::max);
        println!("multi-grid TEM radial profile error: {profile_error:.3e}");
        assert!(
            profile_error < 1.0e-2,
            "multi-grid TEM radial voltage profile differs from the reference: {profile_error:.3e}"
        );

        // The inner engine advanced with the same timestep.
        assert_eq!(multi.inner.num_ts, 5);
        assert!(multi.inner.volt_ptr.get(0, 1, 1, 1).is_finite());
    }
}
