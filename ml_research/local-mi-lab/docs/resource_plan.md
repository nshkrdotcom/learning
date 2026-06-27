# Resource Plan

Default resource limits:

```yaml
resources:
  max_disk_cache_gb: 100
  max_activation_cache_gb_per_run: 20
  max_ram_gb_per_run: 64
  max_gpu_vram_fraction: 0.80
  max_examples_initial: 128
  max_examples_full: 1024
  activation_cache_dtype: float16
  batch_size_auto: true
```

Use the available workstation resources sensibly. The machine can support repeated local runs, selected activation caches, and plots, but the default workflow should not dump every activation from every layer.

First-pass defaults:

- GPT-2 small before Pythia-410M.
- Selected layers before all layers.
- Final token position before all positions.
- Selected attention patterns before broad attention sweeps.
- Initial prompt sets before full prompt sets.
- Script artifacts before notebook inspection.

Attention pattern CSVs are small for GPT-2 small selected-layer runs, but they should stay scoped to the current prompt set and selected layers. Do not cache every attention pattern for every layer and prompt by default.

The controls workflow uses 192 initial examples by default, balanced across six families. This is still a small GPT-2 small run and should remain selected-layer/final-token by default.

The controlled patching follow-up should stay tiny by default: selected candidates only, four prompt families, eight examples per family, at most twelve candidates, and final-position patching. Candidate rows may request heads, but the first pass can patch full layer-level `attn_out`; in that case artifacts must record that the patch was not head-specific. Do not expand this into path patching or an exhaustive component search.
