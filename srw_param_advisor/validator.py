"""
SRW Propagation Quality Validator & Pre-Flight Checker
=======================================================

Two-stage system:
  1. Post-propagation validator: analyses before/after wavefronts to detect
     propagation failures and give actionable fix recommendations.
  2. Pre-flight checker: predicts whether a given set of propagation parameters
     will produce a good result, trained on accumulated validator outcomes.

All diagnostics are computed from the wavefront data alone — no reference
solution needed.

Usage:
    from srw_prop_validator import PropagationValidator

    validator = PropagationValidator()

    # Post-propagation check
    report = validator.validate(wfr_before, wfr_after, prop_params)
    print(report)
    # → "FAIL: beam clipped at output grid edges (right: 3.2% of energy).
    #    → Try increasing pxm from 1.0 to at least 2.5"

    # Pre-flight check (after training on past validations)
    preflight = validator.preflight_check(wfr_before, prop_params)
    print(preflight)
    # → "WARNING: high risk of clipping (78% confidence). Recommend pxm ≥ 2.0"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import json
import os

from srw_param_advisor.wavefront import WavefrontSnapshot


# ============================================================================
# Diagnostic data structures
# ============================================================================


@dataclass
class DiagnosticResult:
    """Single diagnostic check result."""
    name: str
    passed: bool
    severity: float          # 0.0 = fine, 1.0 = catastrophic
    value: float             # the measured quantity
    threshold: float         # what we compared against
    message: str
    recommendation: str      # actionable fix


@dataclass
class ValidationReport:
    """Complete validation report with all diagnostics."""
    passed: bool
    overall_quality: float   # 0.0 = terrible, 1.0 = perfect
    diagnostics: List[DiagnosticResult]
    summary: str

    def __str__(self):
        lines = [
            f"{'PASS' if self.passed else 'FAIL'} — overall quality: {self.overall_quality:.2f}",
            f"  {self.summary}",
            ""
        ]
        for d in self.diagnostics:
            status = "✓" if d.passed else "✗"
            lines.append(f"  {status} {d.name}: {d.message}")
            if not d.passed:
                lines.append(f"    → {d.recommendation}")
        return "\n".join(lines)


@dataclass
class PreflightReport:
    """Pre-flight risk assessment."""
    risk_level: str          # "low", "medium", "high"
    predicted_quality: float
    risk_factors: List[Dict]
    recommendations: List[str]

    def __str__(self):
        lines = [
            f"Pre-flight risk: {self.risk_level.upper()} "
            f"(predicted quality: {self.predicted_quality:.2f})",
            ""
        ]
        for rf in self.risk_factors:
            lines.append(f"  ⚠ {rf['issue']} (confidence: {rf['confidence']:.0%})")
        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    → {r}")
        return "\n".join(lines)


# ============================================================================
# Post-propagation validator
# ============================================================================

class PropagationValidator:
    """
    Validates SRW propagation results by checking for known failure signatures.

    Each diagnostic computes a metric from the before/after wavefronts and
    compares against physics-based thresholds. Failed diagnostics produce
    actionable recommendations.
    """

    # Thresholds (can be tuned)
    ENERGY_TOL = 0.02            # 2% energy change is suspicious
    ENERGY_FAIL = 0.10           # 10% is a clear failure
    EDGE_WARN = 0.005            # 0.5% of energy at edge
    EDGE_FAIL = 0.02             # 2% of energy at edge
    SAMPLING_WARN = 0.7          # phase changes > 0.7π per pixel
    SAMPLING_FAIL = 0.95         # phase changes > 0.95π per pixel
    DISCONTINUITY_WARN = 3.0     # 3σ intensity jump
    DISCONTINUITY_FAIL = 10.0    # 10σ intensity jump

    def validate(
        self,
        wfr_before: WavefrontSnapshot,
        wfr_after: WavefrontSnapshot,
        prop_params: Optional[dict] = None,
    ) -> ValidationReport:
        """
        Run all diagnostics on a before/after wavefront pair.

        Parameters
        ----------
        wfr_before : WavefrontSnapshot
            Wavefront before propagation.
        wfr_after : WavefrontSnapshot
            Wavefront after propagation.
        prop_params : dict, optional
            The propagation parameters used (for context in recommendations).

        Returns
        -------
        ValidationReport
        """
        diagnostics = []

        diagnostics.append(self._check_energy_conservation(wfr_before, wfr_after))
        diagnostics.append(self._check_edge_clipping(wfr_after, 'x'))
        diagnostics.append(self._check_edge_clipping(wfr_after, 'z'))
        diagnostics.append(self._check_sampling_adequacy(wfr_after, 'x'))
        diagnostics.append(self._check_sampling_adequacy(wfr_after, 'z'))
        diagnostics.append(self._check_intensity_discontinuities(wfr_after))
        diagnostics.append(self._check_parseval_consistency(wfr_after))
        diagnostics.append(self._check_negative_intensity(wfr_after))

        passed = all(d.passed for d in diagnostics)
        # Quality: weighted by severity of ALL checks (including warnings within passing)
        # Use root-mean-square so one bad diagnostic dominates
        if diagnostics:
            severities = np.array([d.severity for d in diagnostics])
            rms_severity = float(np.sqrt(np.mean(severities**2)))
        else:
            rms_severity = 0.0
        quality = max(0.0, 1.0 - rms_severity)

        failures = [d for d in diagnostics if not d.passed]
        if not failures:
            summary = "All diagnostics passed."
        else:
            summary = f"{len(failures)} issue(s): " + "; ".join(
                d.name for d in failures
            )

        return ValidationReport(
            passed=passed,
            overall_quality=quality,
            diagnostics=diagnostics,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Individual diagnostics
    # ------------------------------------------------------------------

    def _check_energy_conservation(
        self, wfr_before: WavefrontSnapshot, wfr_after: WavefrontSnapshot
    ) -> DiagnosticResult:
        """
        Check that total intensity is conserved through propagation.

        Free-space propagation of a coherent wavefront must conserve energy
        (Parseval's theorem). Significant changes indicate numerical errors,
        typically from insufficient grid resolution or range.
        """
        E_before = wfr_before.total_energy
        E_after = wfr_after.total_energy

        if E_before == 0:
            return DiagnosticResult(
                name="energy_conservation",
                passed=True, severity=0.0,
                value=0.0, threshold=self.ENERGY_TOL,
                message="Input wavefront has zero energy (trivial case).",
                recommendation=""
            )

        relative_change = abs(E_after - E_before) / E_before
        direction = "gained" if E_after > E_before else "lost"

        if relative_change < self.ENERGY_TOL:
            return DiagnosticResult(
                name="energy_conservation",
                passed=True, severity=relative_change / self.ENERGY_FAIL,
                value=relative_change, threshold=self.ENERGY_TOL,
                message=f"Energy conserved within {relative_change:.1%}.",
                recommendation=""
            )

        severity = min(1.0, relative_change / self.ENERGY_FAIL)

        # Diagnose cause and recommend fix
        if E_after < E_before:
            rec = (
                f"Energy {direction} by {relative_change:.1%}. "
                f"Beam likely extends beyond output grid. "
                f"Increase pxm and/or pzm (range magnification) to expand the grid."
            )
        else:
            rec = (
                f"Energy {direction} by {relative_change:.1%}. "
                f"Possible numerical artifact from insufficient resolution. "
                f"Try increasing pxd and/or pzd (resolution increase), "
                f"or switch to AnalTreatment=1."
            )

        return DiagnosticResult(
            name="energy_conservation",
            passed=(relative_change < self.ENERGY_FAIL),
            severity=severity,
            value=relative_change,
            threshold=self.ENERGY_TOL,
            message=f"Energy {direction} by {relative_change:.1%}.",
            recommendation=rec,
        )

    def _check_edge_clipping(
        self, wfr: WavefrontSnapshot, axis: str
    ) -> DiagnosticResult:
        """
        Check if significant intensity reaches the grid edges.

        If the beam is clipped, the propagation result is wrong because the
        FFT assumes periodic boundary conditions — clipped energy wraps around.
        """
        I = wfr.intensity
        I_total = I.sum()
        if I_total == 0:
            return DiagnosticResult(
                name=f"edge_clipping_{axis}",
                passed=True, severity=0.0,
                value=0.0, threshold=self.EDGE_WARN,
                message=f"No intensity (trivial).",
                recommendation=""
            )

        strip = max(3, min(10, wfr.nx // 50 if axis == 'x' else wfr.nz // 50))

        if axis == 'x':
            edge_low = I[:, :strip].sum() / I_total
            edge_high = I[:, -strip:].sum() / I_total
            param_range = "pxm"
            param_res = "pxd"
            dim_name = "horizontal"
            grid_range = wfr.nx * wfr.x_step
        else:
            edge_low = I[:strip, :].sum() / I_total
            edge_high = I[-strip:, :].sum() / I_total
            param_range = "pzm"
            param_res = "pzd"
            dim_name = "vertical"
            grid_range = wfr.nz * wfr.z_step

        max_edge = max(edge_low, edge_high)
        which_edge = ("left" if axis == 'x' else "bottom") if edge_low > edge_high else ("right" if axis == 'x' else "top")

        if max_edge < self.EDGE_WARN:
            return DiagnosticResult(
                name=f"edge_clipping_{axis}",
                passed=True, severity=max_edge / self.EDGE_FAIL,
                value=max_edge, threshold=self.EDGE_WARN,
                message=f"No {dim_name} clipping ({max_edge:.2%} at {which_edge} edge).",
                recommendation=""
            )

        severity = min(1.0, max_edge / self.EDGE_FAIL)

        # Estimate how much to increase the range
        # Crude: if fraction f of energy is at edge, need roughly 1/(1-f)^0.5 more range
        suggested_pxm = max(1.5, 1.0 / max(1.0 - 5 * max_edge, 0.1)**0.5)

        return DiagnosticResult(
            name=f"edge_clipping_{axis}",
            passed=(max_edge < self.EDGE_FAIL),
            severity=severity,
            value=max_edge,
            threshold=self.EDGE_WARN,
            message=f"{dim_name.capitalize()} clipping: {max_edge:.2%} of energy at {which_edge} edge.",
            recommendation=(
                f"Increase {param_range} to at least {suggested_pxm:.1f} "
                f"to expand the {dim_name} grid range (currently {grid_range*1e3:.2f} mm)."
            ),
        )

    def _check_sampling_adequacy(
        self, wfr: WavefrontSnapshot, axis: str
    ) -> DiagnosticResult:
        """
        Check if the output wavefront phase is adequately sampled.

        Measures the maximum phase change between adjacent pixels using the
        complex conjugate product (no unwrapping needed):
            Δφ = arg(E[i+1] · conj(E[i]))

        If max|Δφ| approaches π, the phase is undersampled and the
        propagation result contains aliasing artifacts.
        """
        # Use the dominant polarisation
        I_x = np.sum(np.abs(wfr.Ex)**2)
        I_z = np.sum(np.abs(wfr.Ez)**2)
        E = wfr.Ex if I_x >= I_z else wfr.Ez

        intensity = np.abs(E)**2
        peak = intensity.max()
        if peak == 0:
            return DiagnosticResult(
                name=f"sampling_{axis}",
                passed=True, severity=0.0,
                value=0.0, threshold=self.SAMPLING_WARN,
                message=f"No intensity (trivial).",
                recommendation=""
            )

        mask_threshold = 0.01 * peak

        if axis == 'x':
            # Phase difference between adjacent pixels in x
            E_prod = E[:, 1:] * np.conj(E[:, :-1])
            I_mid = 0.5 * (intensity[:, 1:] + intensity[:, :-1])
            param = "pxd"
            dim_name = "horizontal"
            step = wfr.x_step
        else:
            E_prod = E[1:, :] * np.conj(E[:-1, :])
            I_mid = 0.5 * (intensity[1:, :] + intensity[:-1, :])
            param = "pzd"
            dim_name = "vertical"
            step = wfr.z_step

        mask = I_mid > mask_threshold
        if not mask.any():
            return DiagnosticResult(
                name=f"sampling_{axis}",
                passed=True, severity=0.0,
                value=0.0, threshold=self.SAMPLING_WARN,
                message=f"Insufficient intensity for {dim_name} sampling check.",
                recommendation=""
            )

        phase_diffs = np.abs(np.angle(E_prod))
        max_dphi = float(np.max(phase_diffs[mask]))
        max_dphi_frac = max_dphi / np.pi  # fraction of π

        if max_dphi_frac < self.SAMPLING_WARN:
            return DiagnosticResult(
                name=f"sampling_{axis}",
                passed=True,
                severity=max_dphi_frac / self.SAMPLING_FAIL,
                value=max_dphi_frac,
                threshold=self.SAMPLING_WARN,
                message=f"{dim_name.capitalize()} phase well sampled (max Δφ = {max_dphi_frac:.2f}π per pixel).",
                recommendation=""
            )

        severity = min(1.0, max_dphi_frac / self.SAMPLING_FAIL)

        # Estimate needed resolution increase
        suggested_pxd = max(1.5, max_dphi_frac / 0.5)  # target: 0.5π per pixel

        # Find WHERE the worst sampling occurs (edge vs center)
        # to give more specific advice
        worst_loc = _find_worst_sampling_location(phase_diffs, mask, axis, wfr)

        return DiagnosticResult(
            name=f"sampling_{axis}",
            passed=(max_dphi_frac < self.SAMPLING_FAIL),
            severity=severity,
            value=max_dphi_frac,
            threshold=self.SAMPLING_WARN,
            message=(
                f"{dim_name.capitalize()} phase undersampled: max Δφ = {max_dphi_frac:.2f}π "
                f"per pixel ({worst_loc})."
            ),
            recommendation=(
                f"Increase {param} to at least {suggested_pxd:.1f} to improve {dim_name} "
                f"resolution (current step: {step*1e6:.2f} μm). "
                f"Alternatively, AnalTreatment=1 can subtract the quadratic phase "
                f"before FFT, reducing the sampling requirement."
            ),
        )

    def _check_intensity_discontinuities(
        self, wfr: WavefrontSnapshot
    ) -> DiagnosticResult:
        """
        Check for unphysical intensity discontinuities (jumps between
        adjacent pixels much larger than expected from smooth beam profiles).

        These indicate ringing from FFT edge effects or numerical instability.
        """
        I = wfr.intensity
        if I.max() == 0:
            return DiagnosticResult(
                name="discontinuities",
                passed=True, severity=0.0,
                value=0.0, threshold=self.DISCONTINUITY_WARN,
                message="No intensity.",
                recommendation=""
            )

        # Compute intensity gradient magnitude
        grad_x = np.diff(I, axis=1)
        grad_z = np.diff(I, axis=0)

        # Normalise by local intensity to get relative jumps
        I_mid_x = 0.5 * (I[:, 1:] + I[:, :-1])
        I_mid_z = 0.5 * (I[1:, :] + I[:-1, :])

        # Only check where there's meaningful signal
        mask_x = I_mid_x > 0.001 * I.max()
        mask_z = I_mid_z > 0.001 * I.max()

        rel_jumps = []
        if mask_x.any():
            rel_x = np.abs(grad_x[mask_x]) / I_mid_x[mask_x]
            rel_jumps.append(rel_x)
        if mask_z.any():
            rel_z = np.abs(grad_z[mask_z]) / I_mid_z[mask_z]
            rel_jumps.append(rel_z)

        if not rel_jumps:
            return DiagnosticResult(
                name="discontinuities",
                passed=True, severity=0.0,
                value=0.0, threshold=self.DISCONTINUITY_WARN,
                message="Insufficient data for discontinuity check.",
                recommendation=""
            )

        all_jumps = np.concatenate(rel_jumps)
        median_jump = float(np.median(all_jumps))
        max_jump = float(np.max(all_jumps))

        # How many σ above median is the max?
        if median_jump > 0:
            sigma_jump = max_jump / median_jump
        else:
            sigma_jump = 0.0

        if sigma_jump < self.DISCONTINUITY_WARN:
            return DiagnosticResult(
                name="discontinuities",
                passed=True,
                severity=sigma_jump / self.DISCONTINUITY_FAIL,
                value=sigma_jump, threshold=self.DISCONTINUITY_WARN,
                message=f"Intensity profile smooth (max relative jump: {sigma_jump:.1f}× median).",
                recommendation=""
            )

        severity = min(1.0, sigma_jump / self.DISCONTINUITY_FAIL)

        return DiagnosticResult(
            name="discontinuities",
            passed=(sigma_jump < self.DISCONTINUITY_FAIL),
            severity=severity,
            value=sigma_jump, threshold=self.DISCONTINUITY_WARN,
            message=f"Intensity discontinuity: max jump is {sigma_jump:.1f}× the median gradient.",
            recommendation=(
                "Sharp intensity features suggest ringing or FFT edge artifacts. "
                "Try AnalTreatment=1 (reduces edge effects by subtracting quadratic phase), "
                "or increase grid range (pxm/pzm) so beam edges don't interact with grid boundaries."
            ),
        )

    def _check_parseval_consistency(
        self, wfr: WavefrontSnapshot
    ) -> DiagnosticResult:
        """
        Check Parseval's theorem: the total power in coordinate and angular
        representations should match.

        Violation indicates the grid doesn't properly support the FFT
        (e.g., significant signal at Nyquist frequency).
        """
        E = wfr.Ex if np.sum(np.abs(wfr.Ex)**2) >= np.sum(np.abs(wfr.Ez)**2) else wfr.Ez

        power_coord = float(np.sum(np.abs(E)**2))
        if power_coord == 0:
            return DiagnosticResult(
                name="parseval_consistency",
                passed=True, severity=0.0,
                value=0.0, threshold=0.01,
                message="No intensity.",
                recommendation=""
            )

        E_fft = np.fft.fft2(E)
        power_freq = float(np.sum(np.abs(E_fft)**2)) / E.size

        relative_diff = abs(power_coord - power_freq) / power_coord

        # Parseval's theorem should hold to machine precision for any grid.
        # Significant violation means something is very wrong with the data.
        threshold = 1e-6
        if relative_diff < threshold:
            return DiagnosticResult(
                name="parseval_consistency",
                passed=True, severity=0.0,
                value=relative_diff, threshold=threshold,
                message=f"Parseval's theorem satisfied (error: {relative_diff:.2e}).",
                recommendation=""
            )

        return DiagnosticResult(
            name="parseval_consistency",
            passed=False,
            severity=min(1.0, relative_diff / 0.01),
            value=relative_diff, threshold=threshold,
            message=f"Parseval's theorem violated by {relative_diff:.2e}.",
            recommendation=(
                "This indicates corrupted E-field data. Check that the wavefront "
                "arrays are not truncated or that memory was not exhausted during propagation."
            ),
        )

    def _check_negative_intensity(
        self, wfr: WavefrontSnapshot
    ) -> DiagnosticResult:
        """
        Check for regions where the computed intensity is anomalously structured
        in ways suggesting numerical artifacts (e.g., regular grid-aligned patterns
        that indicate FFT wraparound).
        
        Note: |E|² is always ≥ 0 by definition, so we check for the subtler
        signature of FFT wraparound: periodic intensity modulation at the
        Nyquist frequency.
        """
        I = wfr.intensity
        if I.max() == 0:
            return DiagnosticResult(
                name="nyquist_artifacts",
                passed=True, severity=0.0,
                value=0.0, threshold=0.01,
                message="No intensity.",
                recommendation=""
            )

        # Check for checkerboard pattern (Nyquist artifact)
        # A checkerboard has the signature that alternating pixels differ
        # systematically. Measure by comparing even/odd pixel sums.
        even_x = I[:, 0::2].sum()
        odd_x = I[:, 1::2].sum()
        total = even_x + odd_x

        if total == 0:
            checker_metric = 0.0
        else:
            checker_metric = abs(even_x - odd_x) / total

        threshold = 0.01  # 1% even/odd asymmetry

        if checker_metric < threshold:
            return DiagnosticResult(
                name="nyquist_artifacts",
                passed=True,
                severity=checker_metric / 0.05,
                value=checker_metric, threshold=threshold,
                message=f"No Nyquist-frequency artifacts detected (metric: {checker_metric:.4f}).",
                recommendation=""
            )

        return DiagnosticResult(
            name="nyquist_artifacts",
            passed=(checker_metric < 0.05),
            severity=min(1.0, checker_metric / 0.05),
            value=checker_metric, threshold=threshold,
            message=f"Nyquist-frequency artifact detected (even/odd asymmetry: {checker_metric:.2%}).",
            recommendation=(
                "Checkerboard intensity pattern indicates signal at the Nyquist frequency, "
                "typically from FFT wraparound of a clipped beam. "
                "Increase grid range (pxm/pzm) and resolution (pxd/pzd)."
            ),
        )

    # ------------------------------------------------------------------
    # Pre-flight checker
    # ------------------------------------------------------------------

    def preflight_check(
        self,
        wfr_before: WavefrontSnapshot,
        drift_length: float,
        prop_params: Optional[dict] = None,
    ) -> PreflightReport:
        """
        Predict whether propagation will succeed based on input wavefront
        analysis and geometric arguments.

        This uses physics-based heuristics, not a trained model (yet).
        The trained model would replace/augment these heuristics after
        accumulating enough validator results as training data.
        """
        risks = []
        recommendations = []

        lambda_m = wfr_before.wavelength
        L = drift_length

        # --- Risk 1: Grid range adequate for beam expansion? ---
        risk_clip = self._preflight_clipping_risk(wfr_before, L, prop_params)
        if risk_clip:
            risks.append(risk_clip)
            recommendations.append(risk_clip['fix'])

        # --- Risk 2: Resolution adequate for phase after propagation? ---
        risk_sampling = self._preflight_sampling_risk(wfr_before, L, prop_params)
        if risk_sampling:
            risks.append(risk_sampling)
            recommendations.append(risk_sampling['fix'])

        # --- Risk 3: Already clipped at input? ---
        risk_input_clip = self._preflight_input_clipping(wfr_before)
        if risk_input_clip:
            risks.append(risk_input_clip)
            recommendations.append(risk_input_clip['fix'])

        # --- Risk 4: Near waist / sign flip? ---
        risk_waist = self._preflight_waist_risk(wfr_before, L)
        if risk_waist:
            risks.append(risk_waist)
            recommendations.append(risk_waist['fix'])

        # Determine overall risk
        if not risks:
            risk_level = "low"
            predicted_quality = 0.95
        else:
            max_conf = max(r['confidence'] for r in risks)
            if max_conf > 0.8:
                risk_level = "high"
                predicted_quality = 0.3
            elif max_conf > 0.5:
                risk_level = "medium"
                predicted_quality = 0.6
            else:
                risk_level = "low"
                predicted_quality = 0.8

        return PreflightReport(
            risk_level=risk_level,
            predicted_quality=predicted_quality,
            risk_factors=risks,
            recommendations=recommendations,
        )

    def _preflight_clipping_risk(self, wfr, L, prop_params):
        """Estimate if beam will overflow the output grid."""
        I = wfr.intensity
        if I.max() == 0:
            return None

        # Estimate beam size from intensity
        sig_x = _estimate_beam_sigma(I, axis=1) * wfr.x_step
        sig_z = _estimate_beam_sigma(I, axis=0) * wfr.z_step

        grid_half_x = 0.5 * wfr.nx * wfr.x_step
        grid_half_z = 0.5 * wfr.nz * wfr.z_step

        # Estimate beam growth through drift
        # Without knowing R exactly, use conservative estimate from beam size
        # and divergence (diffraction limit as lower bound)
        div_x = max(wfr.wavelength / (4 * np.pi * sig_x), 1e-10) if sig_x > 0 else 1e-3
        div_z = max(wfr.wavelength / (4 * np.pi * sig_z), 1e-10) if sig_z > 0 else 1e-3

        # Projected beam size at output
        sig_x_out = np.sqrt(sig_x**2 + (div_x * L)**2)
        sig_z_out = np.sqrt(sig_z**2 + (div_z * L)**2)

        # Apply resize factors if provided
        pxm = prop_params.get('pxm', 1.0) if prop_params else 1.0
        pzm = prop_params.get('pzm', 1.0) if prop_params else 1.0
        eff_half_x = grid_half_x * pxm
        eff_half_z = grid_half_z * pzm

        # Risk: beam extends beyond 3σ of grid half-width
        fill_x = 3 * sig_x_out / eff_half_x if eff_half_x > 0 else 10
        fill_z = 3 * sig_z_out / eff_half_z if eff_half_z > 0 else 10
        max_fill = max(fill_x, fill_z)

        if max_fill > 0.8:
            confidence = min(1.0, (max_fill - 0.5) / 0.5)
            suggested_pxm = max(pxm * fill_x / 0.5, 1.5) if fill_x > fill_z else pxm
            suggested_pzm = max(pzm * fill_z / 0.5, 1.5) if fill_z >= fill_x else pzm
            axis_name = "horizontal" if fill_x > fill_z else "vertical"
            param_name = "pxm" if fill_x > fill_z else "pzm"
            suggested = suggested_pxm if fill_x > fill_z else suggested_pzm
            return {
                'issue': f"Beam likely to clip {axis_name} grid edges after propagation "
                         f"(estimated fill: {max_fill:.0%})",
                'confidence': confidence,
                'fix': f"Increase {param_name} to at least {suggested:.1f}.",
            }
        return None

    def _preflight_sampling_risk(self, wfr, L, prop_params):
        """Estimate if phase sampling will be adequate after propagation."""
        lambda_m = wfr.wavelength

        # After drift, the new curvature adds phase.
        # The most demanding phase variation comes from the quadratic term.
        # In angular representation (mode 0), the phase is:
        #   φ = -π·L·λ·(q_x² + q_z²)
        # The max frequency q is limited by the grid: q_max = 1/(2·dx)
        # Phase at q_max: π·L·λ/(4·dx²)
        # Phase per pixel in angular domain: this divided by (N/2)

        max_phase_per_pixel_x = abs(L) * lambda_m / (2 * wfr.x_step * wfr.nx * wfr.x_step)
        max_phase_per_pixel_z = abs(L) * lambda_m / (2 * wfr.z_step * wfr.nz * wfr.z_step)

        max_frac = max(max_phase_per_pixel_x, max_phase_per_pixel_z) / np.pi

        if max_frac > 0.5:
            confidence = min(1.0, (max_frac - 0.3) / 0.7)
            axis = "horizontal" if max_phase_per_pixel_x > max_phase_per_pixel_z else "vertical"
            param = "pxd" if axis == "horizontal" else "pzd"
            suggested = max(1.5, max_frac / 0.3)
            return {
                'issue': f"Phase may be undersampled in {axis} direction after propagation "
                         f"(estimated max Δφ: {max_frac:.2f}π per pixel)",
                'confidence': confidence,
                'fix': (
                    f"Increase {param} to at least {suggested:.1f}, or use "
                    f"AnalTreatment=1 to subtract quadratic phase before FFT."
                ),
            }
        return None

    def _preflight_input_clipping(self, wfr):
        """Check if input beam is already clipped."""
        I = wfr.intensity
        I_total = I.sum()
        if I_total == 0:
            return None

        strip = max(3, min(10, wfr.nx // 50))
        edges = [
            I[:, :strip].sum() / I_total,
            I[:, -strip:].sum() / I_total,
            I[:strip, :].sum() / I_total,
            I[-strip:, :].sum() / I_total,
        ]
        max_edge = max(edges)

        if max_edge > 0.005:
            confidence = min(1.0, max_edge / 0.02)
            return {
                'issue': f"Input beam already clipped ({max_edge:.2%} of energy at grid edge)",
                'confidence': confidence,
                'fix': (
                    "The input wavefront is already truncated. Re-run the upstream "
                    "propagation with larger grid range, or increase pxm/pzm to "
                    "expand the grid before this drift."
                ),
            }
        return None

    def _preflight_waist_risk(self, wfr, L):
        """Check if propagation passes through or near a waist."""
        if wfr.Robs_x is None or wfr.Robs_z is None:
            return None  # can't check without curvature info

        Rx, Rz = wfr.Robs_x, wfr.Robs_z
        new_Rx = Rx + L
        new_Rz = Rz + L

        # Near-waist: R flips sign, meaning |R+L| is small compared to |R|
        ratio_x = abs(new_Rx) / max(abs(Rx), 1e-30) if Rx != 0 else 1.0
        ratio_z = abs(new_Rz) / max(abs(Rz), 1e-30) if Rz != 0 else 1.0
        min_ratio = min(ratio_x, ratio_z)

        sign_flip_x = (Rx != 0) and (np.sign(new_Rx) != np.sign(Rx))
        sign_flip_z = (Rz != 0) and (np.sign(new_Rz) != np.sign(Rz))

        if sign_flip_x or sign_flip_z or min_ratio < 0.3:
            axis = "horizontal" if ratio_x < ratio_z else "vertical"
            confidence = min(1.0, (0.5 - min_ratio) / 0.5) if min_ratio < 0.5 else 0.5
            return {
                'issue': (
                    f"Propagation passes through or near {axis} waist "
                    f"(R ratio: {min_ratio:.2f})"
                ),
                'confidence': confidence,
                'fix': (
                    "Near-waist propagation requires special handling. "
                    "Use AnalTreatment=4 (propagation to waist) or AnalTreatment=1 "
                    "(analytical quad-phase treatment). Standard mode 0 may fail."
                ),
            }
        return None


# ============================================================================
# Pre-flight model (trainable, learns from past validation outcomes)
# ============================================================================

class PreflightModel:
    """
    Learns to predict validation outcomes from input wavefront features.

    Trained on (input_features, validation_report) pairs accumulated
    from past propagation runs. Uses a simple feature-based classifier
    (logistic regression or small MLP) — the physics-based preflight
    heuristics handle the common cases, this learns the edge cases.

    Training loop:
        model = PreflightModel()
        for wfr_before, wfr_after, params in past_propagations:
            features = model.extract_features(wfr_before, params)
            report = validator.validate(wfr_before, wfr_after, params)
            model.add_training_example(features, report)
        model.train()

    Inference:
        features = model.extract_features(wfr_new, params_new)
        prediction = model.predict(features)
    """

    def __init__(self):
        self.training_features = []
        self.training_labels = []
        self.weights = None
        self.feature_mean = None
        self.feature_std = None

    def extract_features(
        self, wfr: WavefrontSnapshot, drift_length: float, prop_params: dict
    ) -> np.ndarray:
        """
        Extract a fixed-size feature vector from a wavefront + propagation config.

        All features are computed from the INPUT wavefront — this is what we
        have available at pre-flight time.
        """
        I = wfr.intensity
        I_total = I.sum()
        lambda_m = wfr.wavelength

        # Beam size (clipped RMS, robust to noise)
        sig_x_px = _estimate_beam_sigma(I, axis=1)
        sig_z_px = _estimate_beam_sigma(I, axis=0)
        sig_x = sig_x_px * wfr.x_step
        sig_z = sig_z_px * wfr.z_step

        # Grid fill fraction
        fill_x = (2 * sig_x) / (wfr.nx * wfr.x_step) if wfr.nx * wfr.x_step > 0 else 0
        fill_z = (2 * sig_z) / (wfr.nz * wfr.z_step) if wfr.nz * wfr.z_step > 0 else 0

        # Edge clipping
        strip = max(3, min(10, wfr.nx // 50))
        edge_max = 0.0
        if I_total > 0:
            edges = [
                I[:, :strip].sum() / I_total,
                I[:, -strip:].sum() / I_total,
                I[:strip, :].sum() / I_total,
                I[-strip:, :].sum() / I_total,
            ]
            edge_max = max(edges)

        # Phase sampling (from input wavefront)
        dphi_x, dphi_z = _measure_phase_sampling(wfr)

        # Propagation geometry
        L = drift_length
        Nf_x = (wfr.nx * wfr.x_step)**2 / (lambda_m * abs(L)) if L != 0 else 1e10
        Nf_z = (wfr.nz * wfr.z_step)**2 / (lambda_m * abs(L)) if L != 0 else 1e10

        # Prop params
        pxm = prop_params.get('pxm', 1.0)
        pzm = prop_params.get('pzm', 1.0)
        pxd = prop_params.get('pxd', 1.0)
        pzd = prop_params.get('pzd', 1.0)
        analyt_treat = prop_params.get('analyt_treat', 0)

        features = np.array([
            np.log10(max(wfr.photon_energy_eV, 1.0)),   # 0
            np.log10(max(abs(L), 1e-10)),                # 1
            np.sign(L),                                   # 2
            np.log10(max(wfr.nx, 1)),                    # 3
            np.log10(max(wfr.nz, 1)),                    # 4
            np.log10(max(wfr.x_step, 1e-20)),            # 5
            np.log10(max(wfr.z_step, 1e-20)),            # 6
            fill_x,                                       # 7
            fill_z,                                       # 8
            edge_max,                                     # 9
            dphi_x / np.pi,                               # 10
            dphi_z / np.pi,                               # 11
            np.log10(max(Nf_x, 1e-5)),                   # 12
            np.log10(max(Nf_z, 1e-5)),                   # 13
            np.log10(max(pxm, 0.01)),                    # 14
            np.log10(max(pzm, 0.01)),                    # 15
            np.log10(max(pxd, 0.01)),                    # 16
            np.log10(max(pzd, 0.01)),                    # 17
            float(analyt_treat),                          # 18
            np.log10(max(sig_x, 1e-20)),                 # 19
            np.log10(max(sig_z, 1e-20)),                 # 20
        ])
        return features

    def add_training_example(self, features: np.ndarray, report: ValidationReport):
        """Store a (features, outcome) pair for training."""
        # Label: vector of per-diagnostic severities
        label = np.array([d.severity for d in report.diagnostics])
        self.training_features.append(features)
        self.training_labels.append(label)

    def train(self, epochs=500, lr=0.01, reg=1e-4):
        """
        Train a simple linear model: features → predicted severity per diagnostic.

        In production, replace with a small MLP. For proof of concept,
        regularised linear regression is transparent and debuggable.
        """
        if len(self.training_features) < 10:
            print(f"Only {len(self.training_features)} examples — need more data.")
            return

        X = np.array(self.training_features)
        Y = np.array(self.training_labels)

        # Normalise features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-10] = 1.0
        X_norm = (X - self.feature_mean) / self.feature_std

        # Add bias
        X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])

        # Ridge regression: closed-form solution
        # W = (X^T X + λI)^{-1} X^T Y
        n_features = X_aug.shape[1]
        I_reg = reg * np.eye(n_features)
        I_reg[-1, -1] = 0  # don't regularise bias
        self.weights = np.linalg.solve(
            X_aug.T @ X_aug + I_reg,
            X_aug.T @ Y
        )

        # Report training accuracy
        Y_pred = X_aug @ self.weights
        mse = np.mean((Y_pred - Y)**2)
        print(f"Trained on {len(X)} examples, MSE: {mse:.4f}")

    def predict_severities(self, features: np.ndarray) -> np.ndarray:
        """Predict per-diagnostic severity from features."""
        if self.weights is None:
            return np.zeros(8)  # untrained, return optimistic default

        x = (features - self.feature_mean) / self.feature_std
        x_aug = np.append(x, 1.0)
        pred = x_aug @ self.weights
        return np.clip(pred, 0.0, 1.0)

    def save(self, path):
        np.savez(path,
                 weights=self.weights,
                 feature_mean=self.feature_mean,
                 feature_std=self.feature_std)

    def load(self, path):
        data = np.load(path)
        self.weights = data['weights']
        self.feature_mean = data['feature_mean']
        self.feature_std = data['feature_std']


# ============================================================================
# Helper functions
# ============================================================================

def _estimate_beam_sigma(intensity_2d, axis):
    """
    Estimate beam RMS size along an axis (in pixels) from intensity.
    Uses clipped moments (1% threshold) for robustness.
    """
    profile = intensity_2d.sum(axis=axis)
    threshold = 0.01 * profile.max()
    mask = profile > threshold
    if not mask.any():
        return 1.0

    coords = np.arange(len(profile), dtype=float)
    p = profile * mask
    total = p.sum()
    if total == 0:
        return 1.0

    mean = np.sum(coords * p) / total
    var = np.sum((coords - mean)**2 * p) / total
    return max(np.sqrt(var), 0.5)


def _measure_phase_sampling(wfr):
    """
    Measure maximum inter-pixel phase change in the wavefront.
    Returns (max_dphi_x, max_dphi_z) in radians.
    """
    I_x = np.sum(np.abs(wfr.Ex)**2)
    I_z = np.sum(np.abs(wfr.Ez)**2)
    E = wfr.Ex if I_x >= I_z else wfr.Ez

    intensity = np.abs(E)**2
    peak = intensity.max()
    if peak == 0:
        return 0.0, 0.0

    mask_thresh = 0.01 * peak

    # X direction
    E_prod_x = E[:, 1:] * np.conj(E[:, :-1])
    I_mid_x = 0.5 * (intensity[:, 1:] + intensity[:, :-1])
    mask_x = I_mid_x > mask_thresh
    if mask_x.any():
        max_dphi_x = float(np.max(np.abs(np.angle(E_prod_x[mask_x]))))
    else:
        max_dphi_x = 0.0

    # Z direction
    E_prod_z = E[1:, :] * np.conj(E[:-1, :])
    I_mid_z = 0.5 * (intensity[1:, :] + intensity[:-1, :])
    mask_z = I_mid_z > mask_thresh
    if mask_z.any():
        max_dphi_z = float(np.max(np.abs(np.angle(E_prod_z[mask_z]))))
    else:
        max_dphi_z = 0.0

    return max_dphi_x, max_dphi_z


def _find_worst_sampling_location(phase_diffs, mask, axis, wfr):
    """Find where the worst phase sampling occurs (edge vs center)."""
    if not mask.any():
        return "unknown location"

    worst_idx = np.unravel_index(
        np.argmax(phase_diffs * mask), phase_diffs.shape
    )

    if axis == 'x':
        pos = worst_idx[1]
        total = phase_diffs.shape[1]
    else:
        pos = worst_idx[0]
        total = phase_diffs.shape[0]

    frac = pos / max(total - 1, 1)
    if frac < 0.15 or frac > 0.85:
        return "at grid edge"
    elif 0.35 < frac < 0.65:
        return "near beam center"
    else:
        return "in beam body"


# ============================================================================
# Synthetic wavefront generation (for testing)
# ============================================================================

def generate_test_wavefront(
    nx=512, nz=256,
    dx=1e-6, dz=1e-6,
    photon_energy_eV=12000.0,
    R_x=30.0, R_z=30.0,
    beam_sigma_x=100e-6, beam_sigma_z=50e-6,
    center_x=0.0, center_z=0.0,
    tilt_x=0.0, tilt_z=0.0,
    noise_level=0.0,
) -> WavefrontSnapshot:
    """Generate a synthetic Gaussian beam wavefront with known parameters."""
    x = (np.arange(nx) - nx // 2) * dx + center_x
    z = (np.arange(nz) - nz // 2) * dz + center_z
    X, Z = np.meshgrid(x, z)

    wavelength = 1.239842e-06 / photon_energy_eV
    k = 2 * np.pi / wavelength

    # Amplitude
    amp = np.exp(-X**2 / (2 * beam_sigma_x**2) - Z**2 / (2 * beam_sigma_z**2))

    # Phase: curvature + tilt
    phase = np.zeros_like(X)
    if R_x != 0 and np.isfinite(R_x):
        phase += (k / (2 * R_x)) * X**2
    if R_z != 0 and np.isfinite(R_z):
        phase += (k / (2 * R_z)) * Z**2
    phase += k * (tilt_x * X + tilt_z * Z)

    E = amp * np.exp(1j * phase)

    if noise_level > 0:
        noise = noise_level * amp.max() * (
            np.random.randn(nz, nx) + 1j * np.random.randn(nz, nx)
        )
        E += noise

    return WavefrontSnapshot(
        Ex=E,
        Ez=np.zeros_like(E),
        x_start=x[0], x_step=dx,
        z_start=z[0], z_step=dz,
        nx=nx, nz=nz,
        photon_energy_eV=photon_energy_eV,
        Robs_x=R_x, Robs_z=R_z,
    )


def simulate_drift_propagation(
    wfr: WavefrontSnapshot,
    drift_length: float,
    method: str = 'angular',
) -> WavefrontSnapshot:
    """
    Simple FFT-based drift propagation for testing.
    This is NOT SRW — it's a basic angular-spectrum method for generating
    test cases with known failure modes.
    """
    wavelength = wfr.wavelength
    k = 2 * np.pi / wavelength
    L = drift_length

    # Frequency grids
    qx = np.fft.fftfreq(wfr.nx, wfr.x_step)
    qz = np.fft.fftfreq(wfr.nz, wfr.z_step)
    Qx, Qz = np.meshgrid(qx, qz)

    # Angular spectrum transfer function
    # φ = -π·L·λ·(qx² + qz²) with higher-order correction
    qr2 = Qx**2 + Qz**2
    lam_qr2 = wavelength**2 * qr2
    # Avoid sqrt of negative for evanescent waves
    mask_prop = lam_qr2 < 1.0
    phase_tf = np.zeros_like(qr2)
    phase_tf[mask_prop] = k * L * (np.sqrt(1.0 - lam_qr2[mask_prop]) - 1.0)
    transfer = np.exp(1j * phase_tf) * mask_prop

    # Propagate
    Ex_f = np.fft.fft2(wfr.Ex)
    Ez_f = np.fft.fft2(wfr.Ez)
    Ex_prop = np.fft.ifft2(Ex_f * transfer)
    Ez_prop = np.fft.ifft2(Ez_f * transfer)

    return WavefrontSnapshot(
        Ex=Ex_prop, Ez=Ez_prop,
        x_start=wfr.x_start, x_step=wfr.x_step,
        z_start=wfr.z_start, z_step=wfr.z_step,
        nx=wfr.nx, nz=wfr.nz,
        photon_energy_eV=wfr.photon_energy_eV,
        Robs_x=(wfr.Robs_x + L) if wfr.Robs_x is not None else None,
        Robs_z=(wfr.Robs_z + L) if wfr.Robs_z is not None else None,
    )


# ============================================================================
# Demo
# ============================================================================

def demo():
    print("=" * 72)
    print("SRW Propagation Validator — Demo")
    print("=" * 72)

    validator = PropagationValidator()

    # ------------------------------------------------------------------
    # Scenario 1: Good propagation (well-resolved, beam fits grid)
    # ------------------------------------------------------------------
    print("\n" + "—" * 72)
    print("Scenario 1: Well-behaved propagation")
    print("—" * 72)

    wfr_before = generate_test_wavefront(
        nx=512, nz=256, dx=1e-6, dz=1e-6,
        photon_energy_eV=12000, R_x=50, R_z=50,
        beam_sigma_x=80e-6, beam_sigma_z=40e-6,
    )
    wfr_after = simulate_drift_propagation(wfr_before, drift_length=2.0)
    report = validator.validate(wfr_before, wfr_after)
    print(report)

    # Pre-flight
    preflight = validator.preflight_check(
        wfr_before, drift_length=2.0, prop_params={'pxm': 1.0, 'pzm': 1.0}
    )
    print(f"\n{preflight}")

    # ------------------------------------------------------------------
    # Scenario 2: Beam clips the grid (too small range for divergence)
    # ------------------------------------------------------------------
    print("\n" + "—" * 72)
    print("Scenario 2: Beam clips grid edges")
    print("—" * 72)

    wfr_before = generate_test_wavefront(
        nx=256, nz=256, dx=2e-6, dz=2e-6,
        photon_energy_eV=500, R_x=5, R_z=5,
        beam_sigma_x=200e-6, beam_sigma_z=200e-6,
    )
    wfr_after = simulate_drift_propagation(wfr_before, drift_length=3.0)
    report = validator.validate(wfr_before, wfr_after)
    print(report)

    preflight = validator.preflight_check(
        wfr_before, drift_length=3.0, prop_params={'pxm': 1.0, 'pzm': 1.0}
    )
    print(f"\n{preflight}")

    # ------------------------------------------------------------------
    # Scenario 3: Undersampled phase (tight grid, large curvature change)
    # ------------------------------------------------------------------
    print("\n" + "—" * 72)
    print("Scenario 3: Phase undersampled after propagation")
    print("—" * 72)

    wfr_before = generate_test_wavefront(
        nx=128, nz=128, dx=5e-6, dz=5e-6,
        photon_energy_eV=20000, R_x=3, R_z=3,
        beam_sigma_x=100e-6, beam_sigma_z=100e-6,
    )
    wfr_after = simulate_drift_propagation(wfr_before, drift_length=5.0)
    report = validator.validate(wfr_before, wfr_after)
    print(report)

    preflight = validator.preflight_check(
        wfr_before, drift_length=5.0,
        prop_params={'pxm': 1.0, 'pzm': 1.0, 'pxd': 1.0, 'pzd': 1.0}
    )
    print(f"\n{preflight}")

    # ------------------------------------------------------------------
    # Scenario 4: Near-waist propagation
    # ------------------------------------------------------------------
    print("\n" + "—" * 72)
    print("Scenario 4: Propagation through waist")
    print("—" * 72)

    wfr_before = generate_test_wavefront(
        nx=512, nz=512, dx=1e-6, dz=1e-6,
        photon_energy_eV=8000, R_x=-3.0, R_z=-3.0,
        beam_sigma_x=60e-6, beam_sigma_z=60e-6,
    )
    wfr_after = simulate_drift_propagation(wfr_before, drift_length=3.5)
    report = validator.validate(wfr_before, wfr_after)
    print(report)

    preflight = validator.preflight_check(
        wfr_before, drift_length=3.5,
        prop_params={'pxm': 1.0, 'pzm': 1.0, 'analyt_treat': 0}
    )
    print(f"\n{preflight}")

    # ------------------------------------------------------------------
    # Scenario 5: Pre-flight model training demo
    # ------------------------------------------------------------------
    print("\n" + "—" * 72)
    print("Scenario 5: Training pre-flight model on accumulated results")
    print("—" * 72)

    model = PreflightModel()
    np.random.seed(42)

    # Generate training data from random configurations
    n_train = 200
    print(f"Generating {n_train} training propagations...")
    for i in range(n_train):
        energy = 10 ** np.random.uniform(2, 5)
        R = np.random.choice([-1, 1]) * 10 ** np.random.uniform(0, 3)
        L = 10 ** np.random.uniform(-1, 2)
        nx = np.random.choice([128, 256, 512])
        dx = 10 ** np.random.uniform(-7, -4)
        sig = 10 ** np.random.uniform(-5, -3)

        wfr_b = generate_test_wavefront(
            nx=nx, nz=nx, dx=dx, dz=dx,
            photon_energy_eV=energy, R_x=R, R_z=R,
            beam_sigma_x=sig, beam_sigma_z=sig,
        )
        wfr_a = simulate_drift_propagation(wfr_b, L)

        params = {
            'pxm': np.random.choice([0.5, 1.0, 1.5, 2.0]),
            'pzm': np.random.choice([0.5, 1.0, 1.5, 2.0]),
            'pxd': 1.0, 'pzd': 1.0,
            'analyt_treat': np.random.choice([0, 1]),
        }

        features = model.extract_features(wfr_b, L, params)
        report = validator.validate(wfr_b, wfr_a, params)
        model.add_training_example(features, report)

    model.train()

    # Test on a new configuration
    print("\nTesting pre-flight model on new configuration:")
    wfr_test = generate_test_wavefront(
        nx=256, nz=256, dx=2e-6, dz=2e-6,
        photon_energy_eV=10000, R_x=10, R_z=10,
        beam_sigma_x=150e-6, beam_sigma_z=150e-6,
    )
    test_params = {'pxm': 1.0, 'pzm': 1.0, 'pxd': 1.0, 'pzd': 1.0, 'analyt_treat': 0}
    test_features = model.extract_features(wfr_test, drift_length=5.0, prop_params=test_params)
    predicted_severities = model.predict_severities(test_features)

    diag_names = [
        "energy_conservation", "edge_clipping_x", "edge_clipping_z",
        "sampling_x", "sampling_z", "discontinuities",
        "parseval_consistency", "nyquist_artifacts"
    ]
    print("  Predicted diagnostic severities:")
    for name, sev in zip(diag_names, predicted_severities):
        status = "⚠" if sev > 0.3 else "✓"
        print(f"    {status} {name}: {sev:.3f}")

    # Compare with actual propagation
    wfr_test_after = simulate_drift_propagation(wfr_test, 5.0)
    actual_report = validator.validate(wfr_test, wfr_test_after, test_params)
    print("\n  Actual validation:")
    print(f"    {actual_report}")


if __name__ == "__main__":
    demo()
