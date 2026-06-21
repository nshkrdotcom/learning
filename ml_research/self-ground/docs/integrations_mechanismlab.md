# MechanismLab Integrations

MechanismLab uses optional integration manifests to record availability without
importing or requiring heavy packages.

## Execution And Representation

- TransformerLens: local activation capture and activation patching backend.
- SAELens: SAE representation backend for load, encode, and decode.
- nnsight/NDIF: optional remote execution backend for larger model runs.
- pyvene: optional intervention abstraction for serializable/trainable
  interventions and non-TransformerLens architectures.

## Evaluation

- SAEBench/RAVEL: optional benchmark/evaluation backend. Current repo includes a
  bounded probe at `scripts/probe_saebench_ravel_bridge.py`; it does not claim
  integration unless the upstream package is importable and API-compatible.

## Tracking And Artifacts

- Local JSON tracker: always available, writes `tracker_events.jsonl`.
- W&B: optional tracking/dashboard/artifact mirror. It is not a source of claim
  status.
- MLflow: optional local/self-hosted tracking. It is not a source of claim
  status.
- DVC: optional data/artifact versioning for large files.
- Hydra: optional config composition and sweeps.

MechanismLab core tests do not require or contact optional services.
