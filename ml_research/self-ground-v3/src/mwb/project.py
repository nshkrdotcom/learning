from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from mwb.events import append_event, read_events
from mwb.git_state import discover_git_root
from mwb.refs import ref_from_name
from mwb.sqlite_index import initialize_schema, insert_event, insert_project
from mwb.time import utc_now

MECHANISM_DIR = ".mechanism"
SCHEMA_VERSION = 1
PROJECT_DIRS = [
    "sessions",
    "runs",
    "artifacts",
    "cards",
    "claims",
    "hypotheses",
    "exports",
    "cache",
    "adapters",
    "logs",
    "redactions",
]


@dataclass(frozen=True)
class Project:
    root: Path
    name: str
    project_ref: str
    mechanism_dir: Path
    sqlite_path: Path
    events_path: Path
    schema_version: int


def default_project_toml(name: str, created_at: str) -> str:
    return f"""[project]
name = "{name}"
created_at = "{created_at}"
mechanism_dir = ".mechanism"
schema_version = {SCHEMA_VERSION}

[backend]
default_profile = "local"
model_backend = "transformer_lens"
sae_backend = "saelens"

[capture]
auto_capture = true
capture_display = false
capture_stdout = true
max_stdout_bytes = 65536

[artifacts]
root = ".mechanism/artifacts"
hash_algorithm = "sha256"

[domains]
bundle_dirs = ["research/bundles", ".mechanism/bundles"]
builtin = ["negation"]

[thresholds]
control_leaky_ratio = 0.8
off_manifold_kl_drift = 0.1
off_manifold_norm_drift = 0.5
density_matching_max_ratio = 1.5
"""


def read_project_config(project_toml: Path) -> dict:
    try:
        config = tomllib.loads(project_toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid project.toml: {exc}") from exc
    project = config.get("project", {})
    required = ["name", "created_at", "mechanism_dir", "schema_version"]
    missing = [key for key in required if key not in project]
    if missing:
        raise ValueError(f"missing project.toml fields: {', '.join(missing)}")
    if project["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported project schema_version {project['schema_version']}; "
            f"expected {SCHEMA_VERSION}"
        )
    return config


class ProjectManager:
    @staticmethod
    def init(root: Path | None = None, *, name: str = "self-ground") -> Project:
        resolved_root = (root or Path.cwd()).resolve()
        repo_root = discover_git_root(resolved_root) or resolved_root
        mechanism = repo_root / MECHANISM_DIR
        mechanism.mkdir(parents=True, exist_ok=True)
        for directory in PROJECT_DIRS:
            (mechanism / directory).mkdir(parents=True, exist_ok=True)

        created_at = utc_now()
        project_toml = mechanism / "project.toml"
        if not project_toml.exists():
            project_toml.write_text(default_project_toml(name, created_at), encoding="utf-8")

        config = read_project_config(project_toml)
        project_config = config["project"]
        project_name = project_config["name"]
        project_ref = ref_from_name("proj", project_name)
        sqlite_path = mechanism / "workbench.sqlite"
        events_path = mechanism / "events.jsonl"
        events_path.touch(exist_ok=True)
        initialize_schema(sqlite_path)

        project_record = {
            "project_ref": project_ref,
            "name": project_name,
            "root": str(repo_root),
            "mechanism_dir": project_config["mechanism_dir"],
            "schema_version": project_config["schema_version"],
            "created_at": project_config["created_at"],
        }
        insert_project(sqlite_path, project_record)

        events = read_events(events_path)
        if not any(event.get("event_type") == "project_created" for event in events):
            event = append_event(
                events_path,
                "project_created",
                {
                    "project_ref": project_ref,
                    "project_name": project_name,
                    "root": str(repo_root),
                    "mechanism_dir": MECHANISM_DIR,
                    "schema_version": SCHEMA_VERSION,
                },
            )
            insert_event(sqlite_path, event)

        return Project(
            root=repo_root,
            name=project_name,
            project_ref=project_ref,
            mechanism_dir=mechanism,
            sqlite_path=sqlite_path,
            events_path=events_path,
            schema_version=SCHEMA_VERSION,
        )

    @staticmethod
    def discover(start: Path | None = None) -> Project:
        current = (start or Path.cwd()).resolve()
        candidates = [current, *current.parents]
        for candidate in candidates:
            project_toml = candidate / MECHANISM_DIR / "project.toml"
            if project_toml.exists():
                config = read_project_config(project_toml)
                project_config = config["project"]
                mechanism = candidate / project_config["mechanism_dir"]
                return Project(
                    root=candidate,
                    name=project_config["name"],
                    project_ref=ref_from_name("proj", project_config["name"]),
                    mechanism_dir=mechanism,
                    sqlite_path=mechanism / "workbench.sqlite",
                    events_path=mechanism / "events.jsonl",
                    schema_version=project_config["schema_version"],
                )
        raise FileNotFoundError("no .mechanism/project.toml found")

    @staticmethod
    def discover_or_create(start: Path | None = None, *, name: str = "self-ground") -> Project:
        try:
            return ProjectManager.discover(start)
        except FileNotFoundError:
            return ProjectManager.init(start, name=name)

