"""
Analytical propagation parameter estimator.

Stage 1 of the two-stage approach: traces Gaussian beam parameters through
the optical layout using ABCD matrices and produces rule-based parameter
suggestions. No ML — deterministic physics.

Based on the logic in SRW's srTDriftSpace::ChooseLocalPropMode and
TuneRadForPropMeth_1.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalyticalEstimate:
    """Result of the analytical parameter estimation."""
    AT: int
    pxm: float
    pxd: float
    pzm: float
    pzd: float
    expected_quality: float
    R_x: float
    R_z: float
    sigma_x: float
    sigma_z: float
    waist_dist_x: float
    waist_dist_z: float

    def to_prior_vector(self, drift_length: float) -> np.ndarray:
        """Convert to the 12-element prior vector for the RL agent."""
        def signed_log(x):
            return np.sign(x) * np.log10(max(abs(x), 1e-20))

        return np.array([
            signed_log(self.R_x),
            signed_log(self.R_z),
            np.log10(max(self.sigma_x, 1e-20)),
            np.log10(max(self.sigma_z, 1e-20)),
            signed_log(self.waist_dist_x),
            signed_log(self.waist_dist_z),
            float(self.AT),
            np.log(max(self.pxm, 0.1)),
            np.log(max(self.pxd, 0.1)),
            np.log(max(self.pzm, 0.1)),
            np.log(max(self.pzd, 0.1)),
            signed_log(drift_length),
        ], dtype=np.float32)


class AnalyticalDriftEstimator:
    """
    Estimates propagation parameters from beam parameters at a drift entrance.

    Parameters come from ABCD matrix trace, SRW's own Robs, or user input.
    """

    def estimate(
        self,
        R_x: float,
        R_z: float,
        sigma_x: float,
        sigma_z: float,
        drift_length: float,
        photon_energy_eV: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
    ) -> AnalyticalEstimate:
        """
        Compute analytical parameter suggestion.

        Parameters
        ----------
        R_x, R_z : float
            Wavefront curvature radii (m), signed. >0 diverging, <0 converging.
        sigma_x, sigma_z : float
            RMS beam sizes (m).
        drift_length : float
            Drift space length (m), signed.
        photon_energy_eV : float
        dx, dz : float
            Pixel sizes (m).
        nx, nz : int
            Grid points.

        Returns
        -------
        AnalyticalEstimate
        """
        lambda_m = 1.239842e-06 / photon_energy_eV
        L = drift_length

        new_Rx = R_x + L
        new_Rz = R_z + L

        grid_half_x = nx * dx / 2
        grid_half_z = nz * dz / 2

        # --- AnalTreatment selection ---
        phase_cycles_x = abs(grid_half_x ** 2 / (lambda_m * R_x)) if R_x != 0 else 0
        phase_cycles_z = abs(grid_half_z ** 2 / (lambda_m * R_z)) if R_z != 0 else 0

        crit_dist = lambda_m * 5
        radii_ok = abs(R_x) >= crit_dist and abs(R_z) >= crit_dist
        waist_x = radii_ok and abs(new_Rx) < 0.3 * abs(R_x) if R_x != 0 else False
        waist_z = radii_ok and abs(new_Rz) < 0.3 * abs(R_z) if R_z != 0 else False

        if waist_x and waist_z:
            AT = 4  # to waist
        elif phase_cycles_x > 2 or phase_cycles_z > 2:
            AT = 1  # analytical quad-phase
        else:
            AT = 0  # standard angular

        # --- Resize: range magnification ---
        growth_x = abs(new_Rx / R_x) if R_x != 0 else self._diffraction_growth(
            sigma_x, lambda_m, L)
        growth_z = abs(new_Rz / R_z) if R_z != 0 else self._diffraction_growth(
            sigma_z, lambda_m, L)

        margin = 1.15
        pxm = margin * growth_x if growth_x > 1.15 else 1.0
        pzm = margin * growth_z if growth_z > 1.15 else 1.0

        # Shrink if beam is much smaller than grid
        fill_x = 4 * sigma_x / (nx * dx) if nx * dx > 0 else 1
        fill_z = 4 * sigma_z / (nz * dz) if nz * dz > 0 else 1
        if fill_x < 0.2 and pxm >= 1.0:
            pxm = max(0.5, fill_x * 3)
        if fill_z < 0.2 and pzm >= 1.0:
            pzm = max(0.5, fill_z * 3)

        # --- Resize: resolution ---
        phase_pp_x = abs(L) * lambda_m / (2 * dx ** 2 * nx)
        phase_pp_z = abs(L) * lambda_m / (2 * dz ** 2 * nz)
        target = 0.5 * np.pi

        if AT == 1:
            # Quad-phase subtraction reduces sampling need
            pxd = max(phase_pp_x * 0.3 / target, 1.0)
            pzd = max(phase_pp_z * 0.3 / target, 1.0)
        else:
            pxd = max(phase_pp_x / target, 1.0)
            pzd = max(phase_pp_z / target, 1.0)

        # --- Expected quality ---
        quality = 1.0
        if fill_x * pxm * growth_x > 0.8:
            quality -= 0.3
        if fill_z * pzm * growth_z > 0.8:
            quality -= 0.3
        if phase_pp_x / max(pxd, 1) > 0.8 * np.pi:
            quality -= 0.2
        if phase_pp_z / max(pzd, 1) > 0.8 * np.pi:
            quality -= 0.2
        quality = max(0.0, min(1.0, quality))

        waist_dist_x = -R_x if R_x != 0 else 1e23
        waist_dist_z = -R_z if R_z != 0 else 1e23

        return AnalyticalEstimate(
            AT=AT,
            pxm=round(pxm, 3), pxd=round(pxd, 3),
            pzm=round(pzm, 3), pzd=round(pzd, 3),
            expected_quality=round(quality, 3),
            R_x=R_x, R_z=R_z,
            sigma_x=sigma_x, sigma_z=sigma_z,
            waist_dist_x=waist_dist_x, waist_dist_z=waist_dist_z,
        )

    @staticmethod
    def _diffraction_growth(sigma, lambda_m, L):
        """Beam growth for a waist beam (R=inf) via diffraction."""
        div = lambda_m / (4 * np.pi * max(sigma, 1e-15))
        return np.sqrt(1 + (L * div / max(sigma, 1e-15)) ** 2)
