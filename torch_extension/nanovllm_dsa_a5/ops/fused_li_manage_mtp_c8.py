"""Public bindings for the one-kernel A5 C8 MTP LIDU path."""

import torch


# PrivateUse1 and Meta implementations are registered directly by the C++
# extension. Each public invocation launches exactly one repository-local MIX
# kernel; Python no longer composes official LI with a second manager op.
fused_li_manage_mtp_c8 = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8
fused_li_manage_mtp_c8_out = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8_out


__all__ = ["fused_li_manage_mtp_c8", "fused_li_manage_mtp_c8_out"]
