"""Manages API or the molecular lines calcultion."""
from ctypes import c_double, c_int, c_void_p
from os.path import dirname, join, realpath

from cuda.cuda import cuLaunchKernel
from cuda.cudart import cudaDeviceSynchronize, cudaFree, cudaMalloc, \
                        cudaMemcpy, cudaMemcpyKind
from numpy import asarray, double, dtype, intc, zeros

from .cuda_utils import compiled_cuda_module, cuda_error_check, cuda_kernel


# Initialize the cuda library and compile code.
path = join(dirname(realpath(__file__)), "absorption.cu")
kernel = cuda_kernel(compiled_cuda_module(path, 0), "calc_absorption")
host_to_device = cudaMemcpyKind(1)
device_to_host = cudaMemcpyKind(2)


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
            dtype = intc if parameter == "local_iso_id" else double
            host[parameter] = asarray(host[parameter], dtype=dtype, order="C")

        # Get the tips data from the database.
        tips_temperature, tips_data = self.database.tips(self.formula)
        host["tips_temperature"] = tips_temperature.astype(dtype=double, order="C")
        host["tips_data"] = tips_data.astype(dtype=double, order="C")
        self.tips_num_t = host["tips_temperature"].size

        # Allocate space on the device and copy the data over.
        self.device = {}
        for key, value in host.items():
            self.device[key] = cuda_error_check(cudaMalloc(value.nbytes))[0]
            cuda_error_check(cudaMemcpy(self.device[key], value.data, value.nbytes,
                                        host_to_device))
        cuda_error_check(cudaDeviceSynchronize())

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
        n_per_v = int(round(1./(grid[1] - grid[0])))
        vn = int(round(grid[-1]) + 1)
        n = (vn - v0)*n_per_v
        dv = 1./n_per_v
        host["v"] = asarray([v0 + i*dv for i in range(n)], dtype=double, order="C")
        host["absorption"] = zeros(host["v"].size, dtype=double, order="C")
        if not self.num_transitions > 0:
            return host["absorption"][:grid.size]

        device = {}
        for key, value in host.items():
            device[key] = cuda_error_check(cudaMalloc(value.nbytes))[0]
            cuda_error_check(cudaMemcpy(device[key], value.data, value.nbytes,
                                        host_to_device))
        cuda_error_check(cudaDeviceSynchronize())

        # Set the grid and threadblock configuration.
        threads_per_block = 256
        blocks_per_grid = (self.num_transitions + threads_per_block - 1)//threads_per_block

        # Launch the kernel.
        remove_pedestal = 1 if remove_pedestal else 0
        args = (
            (self.device["nu"], c_void_p),
            (self.device["sw"], c_void_p),
            (self.device["gamma_air"], c_void_p),
            (self.device["gamma_self"], c_void_p),
            (self.device["n_air"], c_void_p),
            (self.device["elower"], c_void_p),
            (self.device["delta_air"], c_void_p),
            (self.device["local_iso_id"], c_void_p),
            (self.device["mass"], c_void_p),
            (c_int(self.num_transitions), c_int),
            (c_int(self.tips_num_t), c_int),
            (self.device["tips_temperature"], c_void_p),
            (self.device["tips_data"], c_void_p),
            (c_double(temperature), c_double),
            (c_double(pressure), c_double),
            (c_double(volume_mixing_ratio), c_double),
            (device["v"], c_void_p),
            (c_int(n), c_int),
            (c_int(n_per_v), c_int),
            (device["absorption"], c_void_p),
            (c_int(cut_off), c_int),
            (c_int(remove_pedestal), c_int),
        )
        kernel_args = (tuple([x[0] for x in args]), tuple([x[1] for x in args]))
        cuda_error_check(cuLaunchKernel(kernel, blocks_per_grid, 1, 1,
                                        threads_per_block, 1, 1, 0, 0, kernel_args, 0))
        cuda_error_check(cudaDeviceSynchronize())

        # Copy the result back to the host.
        cuda_error_check(cudaMemcpy(host["absorption"].data, device["absorption"],
                                    host["absorption"].nbytes, device_to_host))
        cuda_error_check(cudaDeviceSynchronize())

        # Free the device memory.
        for value in device.values():
            cuda_error_check(cudaFree(value))
        cuda_error_check(cudaDeviceSynchronize())

        return host["absorption"][:grid.size]
