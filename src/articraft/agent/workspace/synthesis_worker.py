"""Isolated, bounded trajectory solve. stdout is one JSON feedback object."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from articraft.sdk._synthesis import load_problem, save_outputs, task_digest
from articraft.sdk.trajectory import _read_tcp


def main() -> None:
    task, output = Path(sys.argv[1]), Path(sys.argv[2])
    subdivisions = int(sys.argv[3])
    with contextlib.redirect_stdout(sys.stderr):
        print(
            "[trajectory] Loading URDF, design bounds, and trajectory", file=sys.stderr, flush=True
        )
        digest = task_digest(task)
        solver, spec = load_problem(task)
        print(
            f"[trajectory] Functional TCP before optimization: "
            f"{_read_tcp(spec, task.parent / spec['urdf'])}",
            file=sys.stderr,
            flush=True,
        )
        cfg = solver.settings
        if not (
            1 <= cfg.inner_maxiter <= 1000
            and 1 <= cfg.outer_maxeval <= 1000
            and 1 <= cfg.multistarts <= 8
        ):
            raise ValueError(
                "Solver budgets: inner_maxiter/outer_maxeval 1..1000, multistarts 1..8"
            )
        if solver.n > 200 or solver.dof > 20 or len(solver.initial) > 30:
            raise ValueError("Task budget: at most 200 waypoints, 20 joints, 30 design parameters")
        print(
            f"[trajectory] Position tolerance: {solver.task.position_tolerance:.6g} m",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[trajectory] Optimizing {len(solver.initial)} dimensions, {solver.dof} joints, "
            f"{solver.n} waypoints; free endpoints="
            f"{solver.task.q_start is None}/{solver.task.q_end is None}; "
            f"inner_ftol={cfg.inner_ftol:g}, outer_tol={cfg.outer_tol:g}; "
            f"geometry evaluation limit={min(cfg.outer_maxeval, 10)}",
            file=sys.stderr,
            flush=True,
        )
        best, outer = solver.solve(subdivisions=subdivisions)
        print(
            "[trajectory] Checking dense path and saving numerical outputs",
            file=sys.stderr,
            flush=True,
        )
        report = save_outputs(solver, best, outer, spec, output, subdivisions)
        if task_digest(task) != digest:
            raise ValueError("Task changed during optimization; rerun with stable inputs")
        report.update(
            task_digest=digest,
            subdivisions=subdivisions,
            tcp=_read_tcp(spec, output / "optimized_kinematics.urdf"),
        )
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[trajectory] Optimized dimensions: {report['design']}", file=sys.stderr, flush=True)
        print(
            f"[trajectory] Optimized functional TCP: {report['tcp']}", file=sys.stderr, flush=True
        )
        print(
            f"[trajectory] Joint configurations: {report['joint_configurations']}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[trajectory] Accepted: {report['functional_success']}; "
            f"dense error={report['dense_validation']['max_error_m']:.6g} m",
            file=sys.stderr,
            flush=True,
        )
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
