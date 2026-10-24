"""Finish lagged mHC coefficients alongside consumers of its normalized input."""

import torch

from ignition.components.mhc import PreparedMHC


@torch.library.custom_op("flare::start_mhc_coefficients", mutates_args=("coefficients",))
def _start(partials: torch.Tensor, residual: torch.Tensor, normalized: torch.Tensor,
           coefficients: list[torch.Tensor], scale: torch.Tensor, base: torch.Tensor,
           plan_handle: int, stream: int) -> None:
    from b12x.norm.mhc._ops import _coefficients

    auxiliary = torch.cuda.ExternalStream(stream, device=partials.device)
    auxiliary.wait_stream(torch.cuda.current_stream(partials.device))
    with torch.cuda.stream(auxiliary):
        _coefficients(partials, residual, normalized, coefficients, scale, base, plan_handle)


@_start.register_fake
def _fake_start(partials, residual, normalized, coefficients, scale, base, plan_handle, stream):
    pass


@torch.library.custom_op("flare::join_mhc_coefficients", mutates_args=("partials", "coefficients"))
def _join(partials: torch.Tensor, dependency: torch.Tensor,
          coefficients: list[torch.Tensor], stream: int) -> None:
    torch.cuda.current_stream(partials.device).wait_stream(
        torch.cuda.ExternalStream(stream, device=partials.device))


@_join.register_fake
def _fake_join(partials, dependency, coefficients, stream):
    pass


class ConcurrentMHC:
    """Keep projection partials private until the coefficient branch joins.

    The input/projection producer and normalization execute on the main stream.
    Coefficients are independent of downstream normalized-input consumers.
    Their private partials and outputs must remain disjoint from every consumer
    workspace. Join before reading coefficients or overwriting those partials.
    The stream must differ from any concurrently active projection/expert stream.
    """

    def __init__(self, operation: PreparedMHC, *, stream):
        if operation.composed is not None or operation.row_calls is not None:
            raise ValueError("coefficient overlap requires one prepared native invocation")
        if not operation.plan.query.split_lagged:
            raise ValueError("coefficient overlap requires a split lagged mHC declaration")
        self.operation, self.stream = operation, stream
        self.stream_handle = stream.cuda_stream
        self.plan, self.scratch = operation.plan, operation.scratch

    def start(self, *inputs):
        self.operation.input(*inputs)
        binding = self.operation.binding
        _start(binding.partials, binding.out, binding.y,
               [binding.post_buffer, binding.comb_buffer, binding.pre_out],
               self.operation.weights.scale, self.operation.weights.base,
               self.plan.handle, self.stream_handle)
        return binding.y

    def finish(self, dependency):
        binding = self.operation.binding
        _join(binding.partials, dependency,
              [binding.post_buffer, binding.comb_buffer, binding.pre_out], self.stream_handle)
