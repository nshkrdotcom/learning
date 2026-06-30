from attention_lab.models.attention.cp_bilinear import CPBilinearCausalSelfAttention
from attention_lab.models.attention.cp_trilinear import CPTrilinearCausalSelfAttention
from attention_lab.models.attention.multi_qkv_common import MultiQKVSharedBank
from attention_lab.models.attention.multi_qkv_position_rotation import MultiQKVPositionRotationGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_static import MultiQKVStaticGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_train_rotation import MultiQKVTrainRotationGlobalCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.attention.trilinear_cp import TrilinearCPCausalSelfAttention


def build_attention(config, *, layer_idx: int | None = None, shared_qkv_bank: MultiQKVSharedBank | None = None):
    attention_type = getattr(config, "attention_type", "standard")
    if attention_type == "standard":
        return StandardCausalSelfAttention(config)
    if attention_type == "cp_bilinear":
        return CPBilinearCausalSelfAttention(config)
    if attention_type == "cp_trilinear":
        return CPTrilinearCausalSelfAttention(config)
    if attention_type == "multi_qkv_static_3track_global":
        if layer_idx is None or shared_qkv_bank is None:
            raise ValueError("multi_qkv_static_3track_global requires layer_idx and shared_qkv_bank")
        return MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=layer_idx, shared_qkv_bank=shared_qkv_bank)
    if attention_type == "multi_qkv_train_rotation_3track_global":
        if layer_idx is None or shared_qkv_bank is None:
            raise ValueError("multi_qkv_train_rotation_3track_global requires layer_idx and shared_qkv_bank")
        return MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=layer_idx, shared_qkv_bank=shared_qkv_bank)
    if attention_type == "multi_qkv_position_rotation_3track_global":
        if layer_idx is None or shared_qkv_bank is None:
            raise ValueError("multi_qkv_position_rotation_3track_global requires layer_idx and shared_qkv_bank")
        return MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=layer_idx, shared_qkv_bank=shared_qkv_bank)
    if attention_type == "trilinear_cp":
        return TrilinearCPCausalSelfAttention(config)
    raise ValueError(f"Unknown attention_type: {attention_type}")
