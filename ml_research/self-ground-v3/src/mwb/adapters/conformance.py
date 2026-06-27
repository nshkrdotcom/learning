from __future__ import annotations

from mwb.adapters.manifests import AdapterConformanceResult, ClaimBearingGateResult


def claim_bearing_gate(
    results: list[AdapterConformanceResult],
    *,
    required_adapters: list[str],
    required_refs: dict[str, str | None],
) -> ClaimBearingGateResult:
    blockers: list[str] = []
    result_by_name = {result.adapter_name: result for result in results}

    for adapter_name in required_adapters:
        result = result_by_name.get(adapter_name)
        if result is None:
            blockers.append(f"missing adapter conformance: {adapter_name}")
            continue
        if result.status != "pass":
            blockers.append(f"adapter conformance not pass: {adapter_name}={result.status}")

    for name, value in sorted(required_refs.items()):
        if not value:
            blockers.append(f"missing required ref: {name}")

    return ClaimBearingGateResult(supported=not blockers, blockers=blockers)

