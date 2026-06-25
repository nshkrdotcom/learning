# Spec Alignment

The implementation follows the `0430_revised_v6.md` center of gravity:

- Canonical records are flat files under `research/`.
- Local run data lives under `.mechledger/runs/` and is gitignored.
- SQLite is rebuilt state only.
- Draft Guard is deterministic and only checks explicit claim tags.
- Artifacts are explicit registrations or run-local auto-collect files.
- Run ledger appends remain human-approved; wrapped runs create proposals.
- Scientific debt is visible and attached to run evidence, not hidden behind pass/fail.

The original `self-ground` project informed the debt/evidence policy style, especially conservative claim promotion and telemetry-style blockers. Heavy ML execution dependencies are intentionally excluded from this core package.
