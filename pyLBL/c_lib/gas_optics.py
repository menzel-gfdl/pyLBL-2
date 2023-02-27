"""Manages API or the molecular lines calcultion."""
import ctypes
from os.path import dirname, join, realpath

from cuda.cuda import cuLaunchKernel, cuMemAlloc, cuMemcpyDtoH, cuMemcpyHtoD, cuMemFree
from numpy import asarray, dtype, float64, zeros

from .cuda_utils import compiled_cuda_module, cuda_error_check, cuda_kernel


class Gas(object):
    """API for gas optics calculation.

    Attributes:
        database: String path to the spectral sqlite3 database.
        formula: String chemical formula.
    """
    def __init__(self, lines_database, formula):
        """Initializes the object.

        Args:
            lines_database: Database object.
            formula: String chemical formula.
        """
        self.database = lines_database
        self.formula = formula

        # Prepare cuda library and compile code.
        path = join(dirname(realpath(__file__)), "absorption.cu")
        self.kernel = cuda_kernel(compiled_cuda_module(path, 0), "calc_absorption")

        # Get the line parameters from the database.
        _, mass, transitions, _ = self.database.gas(self.formula)
        self.num_transitions = len(transitions)
        if not self.num_transitions > 0:
            return

        # Convert line parameters to arrays.
        parameters = ["nu", "sw", "gamma_air", "gamma_self", "n_air", "elower",
                      "delta_air", "local_iso_id"]
        host = {name: [] for name in parameters}
        host["mass"] = []
        for transition in transitions:
            for parameter in parameters:
                host[parameter].append(getattr(transition, parameter))
            if host["local_iso_id"][-1] == 0:
                host["local_iso_id"][-1] = 10
            host["mass"].append(mass[host["local_iso_id"][-1] - 1])
        for parameter in parameters + ["mass",]:
            host[parameter] = asarray(host[parameter], dtype=float64)

        # Get the tips data from the database.
        tips_temperature, tips_data = self.database.tips(self.formula)
        host["tips_temperature"] = asarray(tips_temperature, dtype=float64)
        host["tips_data"] = asarray(tips_data, dtype=float64)
        self.tips_num_t = host["tips_temperature"].size

        # Allocate space on the device and copy the data over.
        self.device = {}
        for key, value in host.items():
            self.device[key] = cuda_error_check(cuMemAlloc(value.nbytes))[0]
            cuda_error_check(cuMemcpyHtoD(self.device[key], value, value.nbytes))

    def absorption_coefficient(self, temperature, pressure, volume_mixing_ratio, grid,
                               remove_pedestal=False, cut_off=25):
        """Calculates absorption coefficient.

        Args:
            temperature: Temperature [K].
            pressure: Pressure [Pa].
            volume_mixing_ratio: Volume mixing ratio [mol mol-1].
            grid: Numpy array defining the spectral grid [cm-1].
            remove_pedestal: Flag specifying if a pedestal should be subtracted.
            cut_off: Wavenumber cut-off distance [cm-1] from line centers.

        Returns:
            Numpy array of absorption coefficients [m2].
        """
        # Set up grid and output array.
        host = {}
        v0 = int(round(grid[0]))
        vn = int(round(grid[-1]) + 1)
        n_per_v = int(round(1./(grid[1] - grid[0])))
        n = (vn - v0)*n_per_v
        host["v"] = asarray([v0 + i*vn for i in range(n)], dtype=float64)
        host["absorption"] = zeros(host["v"].size, dtype=float64)

        device = {}
        for key, value in host.items():
            device[key] = cuda_error_check(cuMemAlloc(value.nbytes))[0]
            if key != "absorption":
                cuda_error_check(cuMemcpyHtoD(device[key], value, value.nbytes))

        # Set the grid and threadblock configuration.
        threads_per_block = 256
        blocks_per_grid = (self.num_transitions + threads_per_block - 1)/threads_per_block

        # Launch the kernel.
        kernel_args = (
            (
                self.device["nu"], self.device["sw"], self.device["gamma_air"], self.device["gamma_self"],
                self.device["n_air"], self.device["elower"], self.device["delta_air"], self.device["local_iso_id"],
                self.device["mass"], int(self.num_transitions), int(self.tips_num_t),
                self.device["tips_temperature"], self.device["tips_data"], temperature, pressure,
                volume_mixing_ratio, device["v"], n, n_per_v, device["absorption"],
                cut_off, remove_pedestal
            ),
            (
                None, None, None, None, None, None, None, None, None,
                ctypes.c_int, ctypes.c_int, None, None, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, None, ctypes.c_int, ctypes.c_int, None,
                ctypes.c_int, ctypes.c_int
            )
        )
        cuda_error_check(cuLaunchKernel(self.kernel, blocks_per_grid, 1, 1,
                                        threads_per_block, 1, 1, 0, 0, kernel_args, 0))

        # Copy the result back to the host.
        cuda_error_check(cuMemcpyDtoH(host["absorption"], device["absorption"],
                                      host["absorption"].nbytes))

        # Free the device memory.
        for value in device.values():
            cuda_error_check(cuMemFree(value))

        return host["absorption"]
