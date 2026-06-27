from __future__ import annotations

import os

import mwb
from mwb.context import RunContext
from mwb.ipython.capture import IPythonCapture
from mwb.ipython.display import display_card, display_features, display_graph, display_run
from mwb.project import ProjectManager
from mwb.session import SessionManager


def load_ipython_extension(ipython) -> None:
    start_workbench_ipython(ipython, resume=os.environ.get("MWB_RESUME_SESSION") or None)


def start_workbench_ipython(ipython, *, resume: str | None = None) -> None:
    project = ProjectManager.discover_or_create()
    session = SessionManager.start(project, surface="ipython", resume=resume)
    ctx = RunContext(project=project, session=session)
    capture = IPythonCapture(ipython=ipython, session=session)

    ipython.push(
        {
            "ctx": ctx,
            "mwb": mwb,
            "display_card": display_card,
            "display_run": display_run,
            "display_features": display_features,
            "display_graph": display_graph,
            "_mwb_capture": capture,
            "_mwb_session": session,
        }
    )
    capture.install_hooks()


def unload_ipython_extension(ipython) -> None:
    capture = ipython.user_ns.get("_mwb_capture")
    session = ipython.user_ns.get("_mwb_session")
    if capture is not None:
        capture.uninstall_hooks()
    if session is not None:
        session.close()
