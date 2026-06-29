from attention_lab.models.attention_standard import StandardCausalSelfAttention
from attention_lab.models.attention_trilinear_cp import TrilinearCPCausalSelfAttention


def build_attention(config):
    attention_type = getattr(config, "attention_type", "standard")
    if attention_type == "standard":
        return StandardCausalSelfAttention(config)
    if attention_type == "trilinear_cp":
        return TrilinearCPCausalSelfAttention(config)
    raise ValueError(f"Unknown attention_type: {attention_type}")

