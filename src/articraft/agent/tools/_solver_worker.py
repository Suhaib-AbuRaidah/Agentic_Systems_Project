from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

from articraft.agent._child_process import child_environment


async def _relay(stream: asyncio.StreamReader) -> str:
    tail: list[str] = []
    while line := await stream.readline():
        text = line.decode(errors="replace").rstrip()
        print(text, file=sys.stderr, flush=True)
        tail.append(text[:2000])
        tail = tail[-30:]
    return "\n".join(tail)


async def run_solver_worker(
    module: str, workspace: Path, result_path: Path, timeout: float, *args: str
) -> tuple[int, bytes, str] | None:
    """Run a solver and remove partial results on timeout or cancellation."""
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            module,
            *args,
            cwd=workspace,
            env=child_environment(),
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    )
    cancelled = False
    while True:
        try:
            proc = await asyncio.shield(spawning)
            break
        except asyncio.CancelledError:
            # Finish spawning so cancellation can kill the child we own.
            cancelled = True
    assert proc.stdout is not None and proc.stderr is not None
    stdout = asyncio.create_task(proc.stdout.read())
    stderr = asyncio.create_task(_relay(proc.stderr))

    async def collect() -> tuple[bytes, str]:
        # Drain both pipes before waiting for exit. In particular, never leave
        # the child waiting on its own verbose stderr while we await proc.wait().
        raw, errors = await asyncio.gather(stdout, stderr)
        await proc.wait()
        return raw, errors

    collected = asyncio.create_task(collect())
    try:
        if cancelled:
            raise asyncio.CancelledError
        raw, errors = await asyncio.wait_for(asyncio.shield(collected), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        cancelled = cancelled or isinstance(exc, asyncio.CancelledError)
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        while True:
            try:
                await asyncio.shield(collected)
                break
            except asyncio.CancelledError:
                cancelled = True
        result_path.unlink(missing_ok=True)
        if cancelled:
            raise asyncio.CancelledError from None
        return None
    assert proc.returncode is not None
    return proc.returncode, raw, errors
