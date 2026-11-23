from __future__ import annotations

import json
import math
import sys
import uuid
from typing import Any

from articraft.agent.tools._core import Tool, ToolContext, schema, workspace_path
from articraft.agent.tools._solver_worker import run_solver_worker


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    root = context.workspace.resolve()
    task = workspace_path(root, str(args["task_path"]))
    timeout = float(args.get("timeout_seconds", 180))
    if not math.isfinite(timeout) or not 0 < timeout <= 900:
        raise ValueError("timeout_seconds must be positive and at most 900")
    if not task.is_file():
        raise ValueError("Planar mechanism task JSON does not exist")
    output = workspace_path(root, f"planar_output/{uuid.uuid4().hex[:12]}")
    output.mkdir(parents=True)
    result = output / "result.json"
    request = {
        "kind": "planar",
        "task": task.relative_to(root).as_posix(),
        "output": output.relative_to(root).as_posix(),
    }
    (context.run_dir / "trajectory_request.json").write_text(json.dumps(request) + "\n")
    context.successful_compile_digest = None

    print(
        f"[trajectory] Planar tool invoked: {request['task']} (timeout {timeout:g}s)",
        file=sys.stderr,
        flush=True,
    )
    response = await run_solver_worker(
        "articraft.agent.workspace.planar_synthesis_worker",
        root,
        result,
        timeout,
        str(task),
        str(result),
    )
    if response is None:
        print("[trajectory] Planar solver timed out", file=sys.stderr, flush=True)
        return {"status": "error", "error": "planar synthesis timed out", "timed_out": True}
    returncode, _, errors = response
    if returncode != 0:
        result.unlink(missing_ok=True)
        return {"status": "error", "error": "planar synthesis worker failed", "details": errors}
    report = json.loads(result.read_text())
    status = report["status"]
    if status == "solver_failed":
        return {"status": "error", "error": report.get("error", "planar solver failed")}
    print(
        "[trajectory] Returning optimized planar geometry and validation to LLM",
        file=sys.stderr,
        flush=True,
    )
    return {
        "status": "success" if status == "success" else "needs_revision",
        "design_changes": report["design_changes"],
        "validation": report["validation"],
        "optimization": report["optimization"],
        "result_path": result.relative_to(root).as_posix(),
        "next_step": (
            "Use result_path to build geometry and joints. In main.py define planar_targets "
            "as one BodyFrame per task target, in order. Use the actual CAD features. "
            "Omit it for driver-only tasks. Check clearance along the saved motion and compile."
            if status == "success"
            else "Inspect failed constraints and revise the linkage or its initial geometry."
        ),
    }


TOOL = Tool(
    "synthesize_planar_mechanism",
    schema(
        "synthesize_planar_mechanism",
        "Fit a closed-loop planar joint graph to sampled point/pose paths or prescribed motion. "
        "Uses a workspace JSON in metres and radians. Returns optimized local geometry, motion "
        "and validation. Read docs/sdk/common/43_planar_synthesis.md. Run by itself.",
        {
            "task_path": {"type": "string", "description": "Mechanism task JSON in the workspace."},
            "timeout_seconds": {"type": "number", "description": "Default 180, maximum 900."},
        },
        ["task_path"],
    ),
    run,
)
