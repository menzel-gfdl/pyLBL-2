from os import getenv
from os.path import join
from pathlib import Path

from cuda.cuda import cuGetErrorName, cuModuleGetFunction, cuModuleLoadData, CUresult
from cuda.cudart import cudaDeviceAttr, cudaFree, cudaDeviceGetAttribute, \
                        cudaGetDeviceProperties, cudaGetErrorName, cudaError_t, \
                        cudaSetDevice
from cuda.nvrtc import nvrtcCompileProgram, nvrtcCreateProgram, nvrtcGetCUBIN, \
                       nvrtcGetCUBINSize, nvrtcGetErrorString, nvrtcGetProgramLog, \
                       nvrtcGetProgramLogSize, nvrtcGetPTX, nvrtcGetPTXSize, \
                       nvrtcResult, nvrtcVersion
from numpy import char


error_function = {
    CUresult: cuGetErrorName,
    cudaError_t: cudaGetErrorName,
    nvrtcResult: nvrtcGetErrorString,
}
major_cc_version = cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
minor_cc_version = cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor


def cuda_error_string(error):
    """Returns a string describing which CUDA error has occurred."""
    return error_function[type(error)](error)[1].decode("utf-8")


def cuda_error_check(result):
    result = list(result)
    error = result.pop(0)
    if error.value:
        raise RuntimeError(f"cuda error {error.value} ({cuda_error_string(error)})")
    return result if result else None


class GpuDevice(object):
    def __init__(self, device_id):
        self.device_id = device_id
        cuda_error_check(cudaFree(0))  # Initialize cuda.
        self.activate()
        self.major_cc = cuda_error_check(cudaDeviceGetAttribute(major_cc_version, device_id))[0]
        self.minor_cc = cuda_error_check(cudaDeviceGetAttribute(minor_cc_version, device_id))[0]
        self.use_cubin = cuda_error_check(nvrtcVersion())[1] >= 1
        if self.use_cubin:
            self.program_size_func = nvrtcGetCUBINSize
            self.program_bin_func = nvrtcGetCUBIN
        else:
            self.program_size_func = nvrtcGetPTXSize
            self.program_bin_func = nvrtcGetPTX

    def architecture_flag(self):
        """Construct the nvidia architecture compile argument."""
        cc = "sm" if self.use_cubin else "compute"
        return f"--gpu-architecture={cc}_{self.major_cc}{self.minor_cc}"

    def activate(self):
        cuda_error_check(cudaSetDevice(self.device_id))

    def display_properties(self):
        properties = cuda_error_check(cudaGetDeviceProperties(self.device_id));
        print(f"CUDA device [{properties.name}]:")
        print(f"\t# of streaming multi-processors {properties.multiProcessorCount}")
        print(f"\tCompute capacity: {properties.major}{properties.minor}")


def cuda_include_directory():
    """Find the cuda include directory.

    Returns:
        The path to the cuda include directory
    """
    path = getenv("CUDA_HOME") or getenv("CUDA_PATH")
    if path == None:
        raise ValueError("Cannot find the cuda prefix.")
    return join(path, "include")


def compile_flags(device):
    flags = [
        f"{device.architecture_flag()}",
        "-default-device",
        "--fmad=true",  #  Fuse-multiply add.
        f"--include-path={cuda_include_directory()}",  # Include directory.
        "--std=c++11",
    ]
    return [bytes(flag, "utf-8") for flag in flags]


def compile_cuda_library(source_file, device):
    code = bytes(Path(source_file).read_text(), "utf-8")
    flags = compile_flags(device)
    program = cuda_error_check(nvrtcCreateProgram(code, b"defautlprogram.cu", 0, [], []))[0]
    try:
        cuda_error_check(nvrtcCompileProgram(program, len(flags), flags))
    except RuntimeError as err:
        log = b" "*cuda_error_check(nvrtcGetProgramLogSize(program))[0]
        cuda_error_check(nvrtcGetProgramLog(program, log))
        print(log.decode("utf-8"))
        raise
    return program


def compiled_cuda_module(source_file, device_id=0):
    device = GpuDevice(device_id)
    program = compile_cuda_library(source_file, device)
    data = b" "*cuda_error_check(device.program_size_func(program))[0]
    cuda_error_check(device.program_bin_func(program, data))
    return cuda_error_check(cuModuleLoadData(char.array(data)))[0]


def cuda_kernel(module, name):
    return cuda_error_check(cuModuleGetFunction(module, bytes(name, "utf-8")))[0]
