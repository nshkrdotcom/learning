from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from typing import Any

from IPython.core.interactiveshell import ExecutionResult, InteractiveShell

from mwb.domain.objects import WorkbenchObject, is_workbench_object
from mwb.session import CellState, WorkbenchSession


@dataclass(frozen=True)
class NamespaceValue:
    python_id: int
    object_ref: str
    object_type: str
    version: str
    fingerprint: str
    obj: WorkbenchObject


class IPythonCapture:
    def __init__(self, *, ipython: InteractiveShell, session: WorkbenchSession) -> None:
        self.ipython = ipython
        self.session = session
        self.current_cell: CellState | None = None
        self._stdout_redirect: contextlib.redirect_stdout | None = None
        self._stderr_redirect: contextlib.redirect_stderr | None = None
        self._stdout_buffer: io.StringIO | None = None
        self._stderr_buffer: io.StringIO | None = None

    def install_hooks(self) -> None:
        self.ipython.events.register("pre_run_cell", self.pre_run_cell)
        self.ipython.events.register("post_run_cell", self.post_run_cell)
        self.ipython.events.register("post_execute", self.post_execute)

    def uninstall_hooks(self) -> None:
        for event_name, callback in [
            ("pre_run_cell", self.pre_run_cell),
            ("post_run_cell", self.post_run_cell),
            ("post_execute", self.post_execute),
        ]:
            try:
                self.ipython.events.unregister(event_name, callback)
            except ValueError:
                pass

    def pre_run_cell(self, info: Any) -> None:
        self.current_cell = self.session.begin_cell(
            source=info.raw_cell,
            execution_index=int(getattr(self.ipython, "execution_count", 0)),
            namespace_before=self.snapshot_namespace(),
        )
        self._start_stream_capture()

    def post_run_cell(self, result: ExecutionResult) -> None:
        if self.current_cell is None:
            return
        stdout_text, stderr_text = self._stop_stream_capture()
        error = result.error_in_exec or result.error_before_exec
        status = "error" if error else "ok"
        self.session.finish_cell_execution(status=status, error=error)
        self.session.record_cell_streams(
            self.current_cell,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        before = self.current_cell.namespace_before
        after = self.snapshot_namespace()
        self.register_workbench_changes(before, after, self.current_cell)
        self.session.commit_cell(self.current_cell)
        self.current_cell = None

    def post_execute(self) -> None:
        return

    def _start_stream_capture(self) -> None:
        self._stop_stream_capture()
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()
        self._stdout_redirect = contextlib.redirect_stdout(
            _TeeStream(sys.stdout, self._stdout_buffer)
        )
        self._stderr_redirect = contextlib.redirect_stderr(
            _TeeStream(sys.stderr, self._stderr_buffer)
        )
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()

    def _stop_stream_capture(self) -> tuple[str, str]:
        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(None, None, None)
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(None, None, None)
        stdout_text = self._stdout_buffer.getvalue() if self._stdout_buffer else ""
        stderr_text = self._stderr_buffer.getvalue() if self._stderr_buffer else ""
        self._stdout_redirect = None
        self._stderr_redirect = None
        self._stdout_buffer = None
        self._stderr_buffer = None
        return stdout_text, stderr_text

    def snapshot_namespace(self) -> dict[str, NamespaceValue]:
        snapshot: dict[str, NamespaceValue] = {}
        for name, value in self.ipython.user_ns.items():
            if name.startswith("_"):
                continue
            if not is_workbench_object(value):
                continue
            obj = value
            snapshot[name] = NamespaceValue(
                python_id=id(obj),
                object_ref=obj.wb_ref,
                object_type=obj.wb_type,
                version=obj.wb_version,
                fingerprint=obj.wb_fingerprint(),
                obj=obj,
            )
        return snapshot

    def register_workbench_changes(
        self,
        before: dict[str, NamespaceValue],
        after: dict[str, NamespaceValue],
        cell: CellState,
    ) -> None:
        before_refs = {value.object_ref for value in before.values()}
        for name, after_value in after.items():
            before_value = before.get(name)
            if before_value is None:
                event = (
                    "alias_bound"
                    if after_value.object_ref in before_refs
                    else "object_registered"
                )
                self.session.register_object_event(
                    event_type=event,
                    variable_name=name,
                    obj=after_value.obj,
                    cell=cell,
                )
                continue
            if after_value.object_ref != before_value.object_ref:
                event = (
                    "alias_bound"
                    if after_value.object_ref in before_refs
                    else "object_registered"
                )
                self.session.register_object_event(
                    event_type=event,
                    variable_name=name,
                    obj=after_value.obj,
                    cell=cell,
                )
                continue
            if (
                after_value.version != before_value.version
                or after_value.fingerprint != before_value.fingerprint
            ):
                self.session.register_object_event(
                    event_type="object_mutated",
                    variable_name=name,
                    obj=after_value.obj,
                    cell=cell,
                )

        for name, before_value in before.items():
            if name not in after:
                self.session.register_object_event(
                    event_type="alias_deleted",
                    variable_name=name,
                    obj=before_value.obj,
                    cell=cell,
                )


class _TeeStream:
    def __init__(self, original: Any, buffer: io.StringIO) -> None:
        self.original = original
        self.buffer = buffer

    def write(self, value: str) -> int:
        self.buffer.write(value)
        return self.original.write(value)

    def flush(self) -> None:
        self.buffer.flush()
        self.original.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())
