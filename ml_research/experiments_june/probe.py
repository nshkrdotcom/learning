# probe.py
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Only load the model if it isn't already in our interactive memory
if "model" not in globals():
    print("Loading Gemma-2-2B into GPU (bfloat16)...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", 
        device="cuda",
        dtype=torch.bfloat16  # Force bfloat16 to save 50% VRAM!
    )

# Only load the SAE if it isn't already in our interactive memory
if "sae" not in globals():
    print("Loading Gemma-Scope SAE into GPU...")
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device="cuda"
    )

prompt = "JSON: {\"name\": \"Ada\", \"age\": 42"
hook_name = "blocks.12.hook_resid_post"

# 1. Convert prompt to tokens
tokens = model.to_tokens(prompt)

# 2. Capture the activations at Layer 12
_, cache = model.run_with_cache(tokens, names_filter=[hook_name])
resid_activations = cache[hook_name]

# 3. Encode to get sparse feature activations
sae_activations = sae.encode(resid_activations)
final_token_features = sae_activations[0, -1, :]

# 4. Find the top 10 most active features on this final token
top_features = final_token_features.topk(10)

print("\n--- Top 10 Firing SAE Features at Layer 12 ---")
for val, idx in zip(top_features.values.tolist(), top_features.indices.tolist()):
    if val > 0:
        print(f"Feature ID: {idx:<6} (Activation Strength: {val:.4f})")
