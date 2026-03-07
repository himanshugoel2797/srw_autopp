"""
Wavefront data structures and SRW interface utilities.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WavefrontSnapshot:
    """
    Minimal wavefront representation for the parameter advisor.
    Can be constructed from SRW wavefront or from raw arrays.
    """
    Ex: np.ndarray              # complex, shape (nz, nx)
    Ez: np.ndarray              # complex, shape (nz, nx)
    x_start: float              # metres
    x_step: float               # metres
    z_start: float              # metres
    z_step: float               # metres
    nx: int
    nz: int
    photon_energy_eV: float
    Robs_x: Optional[float] = None
    Robs_z: Optional[float] = None

    @classmethod
    def from_srw(cls, wfr):
        """
        Construct from an SRWLWfr object.

        Parameters
        ----------
        wfr : srwpy.srwlib.SRWLWfr
        """
        mesh = wfr.mesh
        nx, nz = mesh.nx, mesh.ny
        n_pts = nx * nz

        if hasattr(wfr, 'arEx') and wfr.arEx is not None:
            ex_flat = np.array(wfr.arEx[:2 * n_pts])
            Ex = (ex_flat[0::2] + 1j * ex_flat[1::2]).reshape(nz, nx)
        else:
            Ex = np.zeros((nz, nx), dtype=complex)

        if hasattr(wfr, 'arEy') and wfr.arEy is not None:
            ez_flat = np.array(wfr.arEy[:2 * n_pts])
            Ez = (ez_flat[0::2] + 1j * ez_flat[1::2]).reshape(nz, nx)
        else:
            Ez = np.zeros((nz, nx), dtype=complex)

        return cls(
            Ex=Ex, Ez=Ez,
            x_start=mesh.xStart,
            x_step=(mesh.xFin - mesh.xStart) / max(nx - 1, 1),
            z_start=mesh.yStart,
            z_step=(mesh.yFin - mesh.yStart) / max(nz - 1, 1),
            nx=nx, nz=nz,
            photon_energy_eV=mesh.eStart,
            Robs_x=getattr(wfr, 'Rx', None),
            Robs_z=getattr(wfr, 'Ry', None),
        )

    def to_srw(self):
        """
        Convert back to an SRWLWfr object for use with SRW propagation.

        Returns
        -------
        srwpy.srwlib.SRWLWfr
        """
        from srwpy.srwlib import SRWLWfr, SRWLRadMesh, array

        mesh = SRWLRadMesh(
            _eStart=self.photon_energy_eV,
            _eFin=self.photon_energy_eV,
            _ne=1,
            _xStart=self.x_start,
            _xFin=self.x_start + (self.nx - 1) * self.x_step,
            _nx=self.nx,
            _yStart=self.z_start,
            _yFin=self.z_start + (self.nz - 1) * self.z_step,
            _ny=self.nz,
        )

        wfr = SRWLWfr()
        wfr.mesh = mesh
        wfr.allocate(1, self.nx, self.nz)

        # Pack Ex: interleaved [Re, Im, Re, Im, ...] as float32
        n = self.nx * self.nz
        ex_flat = np.empty(2 * n, dtype=np.float32)
        ex_vals = self.Ex.ravel()
        ex_flat[0::2] = ex_vals.real
        ex_flat[1::2] = ex_vals.imag
        wfr.arEx = array('f', ex_flat.tolist())

        # Pack Ez as float32
        ez_flat = np.empty(2 * n, dtype=np.float32)
        ez_vals = self.Ez.ravel()
        ez_flat[0::2] = ez_vals.real
        ez_flat[1::2] = ez_vals.imag
        wfr.arEy = array('f', ez_flat.tolist())

        if self.Robs_x is not None:
            wfr.Rx = self.Robs_x
        if self.Robs_z is not None:
            wfr.Ry = self.Robs_z

        return wfr

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.Ex) ** 2 + np.abs(self.Ez) ** 2

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.intensity) * self.x_step * self.z_step)

    @property
    def x_coords(self) -> np.ndarray:
        return self.x_start + np.arange(self.nx) * self.x_step

    @property
    def z_coords(self) -> np.ndarray:
        return self.z_start + np.arange(self.nz) * self.z_step

    @property
    def wavelength(self) -> float:
        return 1.239842e-06 / self.photon_energy_eV
