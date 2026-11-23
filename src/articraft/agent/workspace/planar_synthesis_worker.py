from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


def main(task: Path, output: Path) -> None:
    from articraft.sdk._planar_synthesis import PlanarModel

    raw = task.read_bytes()
    spec = json.loads(raw)
    solver = PlanarModel(spec)
    if spec.get("units", "m") != "m" or solver.bodies[solver.ground]["pose"] != [0, 0, 0]:
        raise ValueError("Use metres and radians, with the ground pose at [0, 0, 0]")
    if len(solver.joints) < len(solver.bodies):
        raise ValueError("Planar synthesis requires a closed loop")
    if solver.K > 200 or len(solver.bodies) > 20 or len(solver.joints) > 40:
        raise ValueError("At most 200 samples, 20 bodies, and 40 joints are supported")
    options = solver.options
    options.setdefault("trajectory_tolerance", 0.001)
    for name in (
        "closure_tolerance",
        "driver_tolerance",
        "angular_closure_tolerance",
        "angular_driver_tolerance",
    ):
        options.setdefault(name, 1e-6)
    for name in (
        "trajectory_tolerance",
        "orientation_tolerance",
        "closure_tolerance",
        "driver_tolerance",
        "angular_closure_tolerance",
        "angular_driver_tolerance",
    ):
        if name in options and (not math.isfinite(options[name]) or options[name] <= 0):
            raise ValueError(f"{name} must be positive and finite")
    for name, default, limit in (("verify_substeps", 2, 20), ("max_iter", 700, 2000)):
        n = options.get(name, default)
        if type(n) is not int or not 1 <= n <= limit:
            raise ValueError(f"{name} must be an integer from 1 to {limit}")
    print(
        f"[trajectory] Solving planar linkage with {len(solver.bodies)} bodies, "
        f"{len(solver.variables)} design variables, and {solver.K} samples",
        file=sys.stderr,
        flush=True,
    )
    report = solver.solve()
    if task.read_bytes() != raw:
        raise ValueError("Planar task changed during synthesis")
    report["task_digest"] = hashlib.sha256(raw).hexdigest()
    output.write_text(json.dumps(report, allow_nan=False) + "\n")
    stats = report["optimization"]
    print(
        f"[trajectory] Planar result: {report['status']}, "
        f"{stats.get('iterations', 0)} IPOPT iterations; "
        f"validation={report.get('validation', {})}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
