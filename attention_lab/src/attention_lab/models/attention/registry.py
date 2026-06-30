from attention_lab.models.attention.cp_bilinear import CPBilinearCausalSelfAttention
from attention_lab.models.attention.cp_trilinear import CPTrilinearCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.attention.trilinear_cp import TrilinearCPCausalSelfAttention


def build_attention(config):
    attention_type = getattr(config, "attention_type", "standard")
    if attention_type == "standard":
        return StandardCausalSelfAttention(config)
    if attention_type == "cp_bilinear":
        return CPBilinearCausalSelfAttention(config)
    if attention_type == "cp_trilinear":
        return CPTrilinearCausalSelfAttention(config)
    if attention_type == "trilinear_cp":
        return TrilinearCPCausalSelfAttention(config)
    raise ValueError(f"Unknown attention_type: {attention_type}")
