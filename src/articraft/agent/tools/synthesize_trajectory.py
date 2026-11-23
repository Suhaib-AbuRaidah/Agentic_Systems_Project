from __future__ import annotations

import json
import math
import sys
import uuid
from typing import Any

from articraft.agent.tools._core import Tool, ToolContext, schema, workspace_path
from articraft.agent.tools._solver_worker import run_solver_worker


async def run(context: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    task = workspace_path(context.workspace, str(args["task_path"]))
    spec = json.loads(task.read_text())
    # All declared inputs must stay in the run workspace, including symlinks.
    workspace_path(context.workspace, str(task.parent / spec["urdf"]))
    if "trajectory" in spec:
        workspace_path(context.workspace, str(task.parent / spec["trajectory"]["module"]))
    tcp = spec.get("tcp", {})
    description = tcp.get("description") if isinstance(tcp, dict) else None
    axes = tcp.get("axes") if isinstance(tcp, dict) else None
    if (
        not isinstance(description, str)
        or not description.strip()
        or not isinstance(axes, dict)
        or any(not isinstance(axes.get(axis), str) or not axes[axis].strip() for axis in "xyz")
    ):
        raise ValueError(
            "Task tcp must describe the functional working point in description "
            "and its x, y, z directions in axes. Read docs/sdk/common/42_trajectory_synthesis.md."
        )
    timeout = float(args.get("timeout_seconds", 180))
    subdivisions = args.get("subdivisions", 10)
    if not math.isfinite(timeout) or not 0 < timeout <= 900:
        raise ValueError("timeout_seconds must be positive and at most 900")
    if not isinstance(subdivisions, int) or not 1 <= subdivisions <= 100:
        raise ValueError("subdivisions must be an integer between 1 and 100")
    root = context.workspace.resolve()
    output = workspace_path(root, f"trajectory_output/{uuid.uuid4().hex[:12]}")
    output.mkdir(parents=True)
    request = {
        "task": task.relative_to(root).as_posix(),
        "output": output.relative_to(root).as_posix(),
    }
    (context.run_dir / "trajectory_request.json").write_text(json.dumps(request) + "\n")
    context.successful_compile_digest = None
    print(
        f"[trajectory] Tool invoked: {request['task']} (timeout {timeout:g}s)",
        file=sys.stderr,
        flush=True,
    )
    response = await run_solver_worker(
        "articraft.agent.workspace.synthesis_worker",
        root,
        output / "result.json",
        timeout,
        str(task),
        str(output),
        str(subdivisions),
    )
    if response is None:
        print(
            "[trajectory] Solver timed out; final compile remains blocked",
            file=sys.stderr,
            flush=True,
        )
        return {"status": "error", "error": "trajectory synthesis timed out", "timed_out": True}
    returncode, raw, errors = response
    if returncode != 0:
        (output / "result.json").unlink(missing_ok=True)
        return {"status": "error", "error": "trajectory worker failed", "details": errors}
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        (output / "result.json").unlink(missing_ok=True)
        raise ValueError("trajectory worker returned invalid JSON") from exc
    print(
        "[trajectory] Returning TCP, dimensions, joint-path location, and validation feedback to LLM",
        file=sys.stderr,
        flush=True,
    )
    return {
        "status": "success" if report["functional_success"] else "needs_revision",
        "waypoint_count": report["waypoint_count"],
        "position_tolerance_m": report["position_tolerance_m"],
        "design": report["design"],
        "tcp": report["tcp"],
        "joint_order": report["joint_order"],
        "joint_configurations": report["joint_configurations"],
        "initial_max_error_m": report["initial_max_error_m"],
        "waypoint_max_error_m": report["waypoint_max_error_m"],
        "waypoint_constraints_pass": report["waypoint_constraints_pass"],
        "dense_validation": report["dense_validation"],
        "inner_message": report["inner_message"],
        "outer_message": report["outer_message"],
        "early_stopped": report["early_stopped"],
        "geometry_evaluations": report["geometry_evaluations"],
        "result_dir": request["output"],
        "joint_path": f"{request['output']}/joint_path.csv",
        "next_step": "Apply accepted dimensions to geometry, joints, and trajectory_tip, then "
        "compile. If needs_revision, correct the task and rerun without weakening requirements.",
    }


TOOL = Tool(
    "synthesize_trajectory",
    schema(
        "synthesize_trajectory",
        "Optimize a requested TCP trajectory on an open serial chain. "
        "Task JSON format: docs/sdk/common/42_trajectory_synthesis.md. "
        "Input: initial URDF, bounded parameters, targets, and TCP description/axes. "
        "Returns dimensions, TCP, joint-path files, and validation. "
        "Later compiles verify rebuilt geometry. Run without concurrent workspace tools.",
        {
            "task_path": {"type": "string", "description": "Task JSON inside the workspace."},
            "timeout_seconds": {"type": "number", "description": "Default 180, maximum 900."},
            "subdivisions": {
                "type": "integer",
                "description": "Dense checks per segment, 1..100; default 10.",
            },
        },
        ["task_path"],
    ),
    run,
)
