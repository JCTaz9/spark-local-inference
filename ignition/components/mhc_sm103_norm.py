"""Standard CuTe host ABI for FlashInfer's unchanged BF16 RMSNorm kernel."""

import cuda.bindings.driver as cuda
import cutlass
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from cutlass import cute
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel


class BF16NormHost:
    def __init__(self, hidden, wide_rows):
        self.hidden, self.wide_rows = hidden, wide_rows
        self.kernel = RMSNormKernel(cutlass.BFloat16, hidden, 0.0, sm_version=103)

    @cute.jit
    def __call__(self, source: cute.Pointer, weight: cute.Pointer, output: cute.Pointer,
                 rows: cutlass.Int64, epsilon: cutlass.Float32, stream: cuda.CUstream):
        row_stride = cutlass.Int64(self.hidden) if cutlass.const_expr(self.wide_rows) else self.hidden
        layout = cute.make_layout((rows, self.hidden), stride=(row_stride, 1))
        self.kernel(cute.make_tensor(source, layout),
                    cute.make_tensor(weight, cute.make_layout(self.hidden)),
                    cute.make_tensor(output, layout), rows, epsilon, True, stream)


def pointer(address):
    return make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=16)


def launch(program, source, weight, output, rows, epsilon):
    program(pointer(source.data_ptr()), pointer(weight.data_ptr()), pointer(output.data_ptr()),
            rows, epsilon, current_cuda_stream())
