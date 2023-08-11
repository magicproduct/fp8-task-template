from typing import Optional

import torch


@torch.jit.script
def round_to_fp8_represented_as_int8(
        t: torch.Tensor,
        n_mantissa: int,
        out: Optional[torch.Tensor] = None,
        scaling_factor: float = 1,
) -> torch.Tensor:
    raise NotImplementedError("round_to_fp8_represented_as_int8")


@torch.jit.script
def undo_int8_fp8(
        fp8_tensor: torch.Tensor,
        n_mantissa: int,
        target_dt: torch.dtype,
        out: Optional[torch.Tensor] = None,
        scaling_factor: float = 1,
) -> torch.Tensor:
    raise NotImplementedError("undo_int8_fp8")
