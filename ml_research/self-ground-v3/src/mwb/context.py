from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import torch
from ruamel.yaml import YAML

from mwb.adapters.saelens import SAELensAdapter
from mwb.adapters.transformer_lens import TransformerLensAdapter
from mwb.artifacts import ArtifactRegistry
from mwb.domain.objects import (
    ActivationSet,
    ControlBundle,
    DictionaryIdentity,
    DomainBundle,
    ExampleBundle,
    FeatureRanking,
    Hypothesis,
    ModelIdentity,
    PredictionLock,
    WorkbenchObject,
)
from mwb.git_state import capture_git_state
from mwb.hashing import sha256_text
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.session import WorkbenchSession


@dataclass
class ObjectFactory:
    ctx: RunContext

    def create(
        self,
        wb_type: str,
        *,
        metadata: dict[str, Any] | None = None,
        parents: list[str] | None = None,
    ) -> WorkbenchObject:
        self.ctx.object_counter += 1
        object_ref = stable_ref(
            "obj",
            self.ctx.session.session_ref,
            self.ctx.object_counter,
            wb_type,
            metadata or {},
            parents or [],
        )
        return WorkbenchObject(
            wb_ref=object_ref,
            wb_type=wb_type,
            parents=list(parents or []),
            metadata=dict(metadata or {}),
        )


@dataclass
class ModelFactory:
    ctx: RunContext

    def load_tl(self, model_name: str, *, device: str = "cpu") -> ModelIdentity:
        adapter = TransformerLensAdapter()
        model = adapter.load_model(model_name, device=device)
        identity = adapter.model_identity_for_name(
            model_name,
            backend_version=adapter.backend_version_manifest(device=device).package_versions.get(
                "transformer_lens"
            ),
            config=getattr(model.cfg, "to_dict", lambda: {})(),
        )
        identity._backend_model = model
        self.ctx.last_model = identity
        return identity


@dataclass
class SAEFactory:
    ctx: RunContext

    def load(
        self,
        release: str,
        *,
        hook: str,
        sae_id: str | None = None,
        device: str = "cpu",
    ) -> DictionaryIdentity:
        if self.ctx.last_model is None:
            raise ValueError("ctx.saes.load requires ctx.models.load_tl to run first")
        adapter = SAELensAdapter()
        raw_sae = adapter.load_sae(release=release, sae_id=sae_id or hook, device=device)
        feature_count = None
        if hasattr(raw_sae, "cfg"):
            feature_count = getattr(raw_sae.cfg, "d_sae", None)
        identity = adapter.dictionary_identity_for_name(
            release=release,
            sae_id=sae_id or hook,
            hook_point=hook,
            model_ref=self.ctx.last_model.wb_ref,
            feature_count=int(feature_count) if feature_count is not None else None,
        )
        identity._backend_sae = raw_sae
        self.ctx.last_sae = identity
        return identity


@dataclass
class NegationDomain:
    ctx: RunContext

    def load(self, name: str) -> DomainBundle:
        if name != "phase3_calibrated":
            raise ValueError(f"unknown built-in negation bundle: {name}")
        bundle_file = resources.files("mwb.resources.bundles").joinpath(
            "negation_phase3_calibrated.yaml"
        )
        data = YAML(typ="safe").load(bundle_file.read_text(encoding="utf-8"))
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
        bundle_hash = sha256_text(encoded)
        source = str(bundle_file)
        targets = ExampleBundle(
            wb_ref=stable_ref("bundle", "negation", name, "targets", bundle_hash),
            name=f"{name}_targets",
            domain="negation",
            examples=list(data["examples"]),
            source=source,
            bundle_hash=bundle_hash,
        )
        controls = ControlBundle(
            wb_ref=stable_ref("ctrl", "negation", name, "controls", bundle_hash),
            name=f"{name}_controls",
            domain="negation",
            control_families=dict(data["controls"]),
            source=source,
            bundle_hash=bundle_hash,
            parents=[targets.wb_ref],
        )
        return DomainBundle(
            wb_ref=stable_ref("bundle", "negation", name, bundle_hash),
            name=name,
            domain="negation",
            targets=targets,
            controls=controls,
            source=source,
            bundle_hash=bundle_hash,
            parents=[targets.wb_ref, controls.wb_ref],
        )


@dataclass
class DomainRegistry:
    ctx: RunContext

    def __post_init__(self) -> None:
        self.negation = NegationDomain(self.ctx)


@dataclass
class CaptureBuilder:
    ctx: RunContext
    model: ModelIdentity
    bundle: DomainBundle

    def at(self, hook_point: str) -> ActivationSet:
        raw_model = self.model._backend_model
        if raw_model is None:
            raise ValueError("model has no loaded TransformerLens backend object")
        prompts = [str(example["prompt"]) for example in self.bundle.targets.examples]
        tokens = raw_model.to_tokens(prompts)
        with torch.no_grad():
            _, cache = raw_model.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_point,
            )
        activation = cache[hook_point].detach().cpu()
        d_model = int(activation.shape[-1])
        tensor_space_ref = stable_ref("space", self.model.wb_ref, hook_point, d_model)
        activation_ref = stable_ref(
            "obj",
            "ActivationSet",
            self.model.wb_ref,
            self.bundle.wb_ref,
            hook_point,
            list(activation.shape),
        )
        obj = ActivationSet(
            wb_ref=activation_ref,
            model_ref=self.model.wb_ref,
            bundle_ref=self.bundle.wb_ref,
            hook_point=hook_point,
            tensor_space_ref=tensor_space_ref,
            activation_summary={
                "shape": [int(dim) for dim in activation.shape],
                "dtype": str(activation.dtype),
                "prompt_count": len(prompts),
            },
            parents=[self.model.wb_ref, self.bundle.wb_ref],
        )
        obj._activation = activation
        self.ctx.last_activation = obj
        return obj


@dataclass
class FeatureFactory:
    ctx: RunContext

    def rank(
        self,
        sae: DictionaryIdentity,
        acts: ActivationSet,
        *,
        contrast: str,
        top_k: int = 20,
    ) -> FeatureRanking:
        raw_sae = sae._backend_sae
        if raw_sae is None:
            raise ValueError("SAE has no loaded SAELens backend object")
        if acts._activation is None:
            raise ValueError("activation set has no captured activation tensor")
        activation = acts._activation
        with torch.no_grad():
            encoded = raw_sae.encode(activation)
        reduce_dims = tuple(range(encoded.ndim - 1))
        scores = encoded.float().abs().mean(dim=reduce_dims)
        top = torch.topk(scores, k=min(top_k, int(scores.shape[0])))
        rows = [
            {"feature_index": int(index), "score": float(score)}
            for score, index in zip(top.values, top.indices, strict=True)
        ]
        return FeatureRanking(
            wb_ref=stable_ref("obj", "FeatureRanking", sae.wb_ref, acts.wb_ref, contrast, rows),
            dictionary_ref=sae.wb_ref,
            activation_ref=acts.wb_ref,
            contrast=contrast,
            rows=rows,
            parents=[sae.wb_ref, acts.wb_ref],
        )


@dataclass
class ArtifactFactory:
    ctx: RunContext

    def register(self, path: str, *, role: str, parents: list[str] | None = None):
        return ArtifactRegistry(self.ctx.project).register_path(
            self.ctx.project.root / path,
            role=role,
            parents=parents,
        )


@dataclass
class HypothesisFactory:
    ctx: RunContext

    def create(
        self,
        *,
        title: str,
        units: list[str],
        example_bundle: ExampleBundle,
        control_bundle: ControlBundle,
        expected_effect: str,
        required_controls: list[str],
        alternative_explanations: list[str] | None = None,
        requested_evidence_tier: str = "causal_necessity",
    ) -> Hypothesis:
        return Hypothesis(
            wb_ref=stable_ref(
                "hyp",
                title,
                units,
                example_bundle.wb_ref,
                control_bundle.wb_ref,
                expected_effect,
            ),
            title=title,
            units=units,
            example_bundle_ref=example_bundle.wb_ref,
            control_bundle_ref=control_bundle.wb_ref,
            expected_effect=expected_effect,
            required_controls=required_controls,
            alternative_explanations=list(alternative_explanations or []),
            requested_evidence_tier=requested_evidence_tier,
            parents=[example_bundle.wb_ref, control_bundle.wb_ref, *units],
        )


@dataclass
class PredictionFactory:
    ctx: RunContext

    def lock(
        self,
        hypothesis: Hypothesis,
        *,
        expected_direction: str,
        expected_controls: dict[str, str],
    ) -> PredictionLock:
        git_state = capture_git_state(self.ctx.project.root)
        environment = {
            "python": "uv",
            "project_root": str(self.ctx.project.root),
        }
        return PredictionLock(
            wb_ref=stable_ref(
                "lock",
                hypothesis.wb_ref,
                hypothesis.wb_fingerprint(),
                expected_direction,
                expected_controls,
            ),
            hypothesis_ref=hypothesis.wb_ref,
            hypothesis_spec_hash=hypothesis.wb_fingerprint(),
            expected_direction=expected_direction,
            expected_controls=expected_controls,
            git_state=git_state,
            environment=environment,
            parents=[hypothesis.wb_ref],
        )


@dataclass
class RunContext:
    project: Project
    session: WorkbenchSession
    object_counter: int = 0
    last_model: ModelIdentity | None = None
    last_sae: DictionaryIdentity | None = None
    last_activation: ActivationSet | None = None

    def __post_init__(self) -> None:
        self.objects = ObjectFactory(self)
        self.models = ModelFactory(self)
        self.saes = SAEFactory(self)
        self.domains = DomainRegistry(self)
        self.features = FeatureFactory(self)
        self.artifact = ArtifactFactory(self)
        self.hypotheses = HypothesisFactory(self)
        self.predictions = PredictionFactory(self)

    def capture(self, model: ModelIdentity, bundle: DomainBundle) -> CaptureBuilder:
        return CaptureBuilder(self, model, bundle)

    def record(self, obj: WorkbenchObject, *, name: str | None = None) -> WorkbenchObject:
        metadata = dict(obj.metadata)
        if name:
            metadata["label"] = name
        return obj.model_copy(update={"metadata": metadata})

    def note(self, text: str) -> WorkbenchObject:
        return self.objects.create("Note", metadata={"text": text})
