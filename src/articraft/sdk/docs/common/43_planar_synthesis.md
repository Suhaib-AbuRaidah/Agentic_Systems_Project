# Closed-loop planar synthesis

Use `synthesize_planar_mechanism` when the user specifies a path, pose sequence, or motion for a planar linkage. Choose the topology first. The solver fits that graph and returns optimized joint attachments and assembled poses. It supports revolute, prismatic and fixed joints.

## Task JSON

Write a workspace JSON object with these fields:

| Field | Format |
| --- | --- |
| `units` | `"m"`. Lengths are metres, angles radians. The mechanism moves in ground XY. |
| `bodies` | `[{"id":"ground","ground":true,"pose":[0,0,0],"frames":{"pivot":[0,0,0]}}, ...]`. Give each moving body an approximate assembled `pose:[x,y,yaw]` near the intended branch. |
| `joints` | `[{"id":"hinge","type":"revolute","a":["ground","pivot"],"b":["crank","prox"],"limits":[-1,2]}, ...]`. Include every closing joint. Revolute coordinates are B yaw minus A yaw. Prismatic coordinates are along frame A's local +X. |
| `targets` | `[{"body":"coupler","frame":"tip","positions":[[0.1,0.04],[0.11,0.05]],"orientations":[0.2,0.2]}]`. Positions can use a named `point` instead of `frame`. A body's `points` map uses local `[x,y]`. Orientation without a frame refers to the body. |
| `drivers` | Optional `[{"joint":"hinge","values":[0.2,0.4]}]`. Motion-only tasks can omit targets and provide at least two prescribed driver values. A driver without values can specify `"monotonic":"increasing"` or `"decreasing"` when targets exist. |
| `design` | Prefer `{"mode":"explicit"}`. Mark adjustable attachments with `{"pose":[0.08,0,0],"free":["x"],"bounds":{"x":[0.06,0.1]}}`. For points use `xy` instead of `pose`. |
| `geometry_constraints` | Optional body-local distances, e.g. `{"type":"distance","body":"crank","a":["frame","prox"],"b":["frame","dist"],"min":0.04,"max":0.09}` inside the list. |
| `options` | `trajectory_tolerance` (default 0.001 m), `orientation_tolerance` (default 0.05 rad), `verify_substeps` (default 2). Preserve the user's limits. |

Sample equations into ordered target rows before calling the tool. Use matching counts for targets and prescribed drives, repeat a fixed yaw at each sample, and use unwrapped angles. For mechanisms with multiple independent inputs, prescribe enough drives to determine the intended motion between samples.

## Solve, build and check

Call `synthesize_planar_mechanism` with `{"task_path":"planar_task.json"}`. If accepted, read its `result_path`. Use `mechanism.bodies` for CAD dimensions and attachment locations. Keep positions local to each body, and use solved offsets when placing holes, pins and rails.

In `main.py`, after creating the named rigid bodies and their shapes:

```python
import json
from pathlib import Path
from articraft.sdk import TestContext
from articraft.sdk.planar import apply_planar_joints, planar_motion_states

result = json.loads(Path("planar_output/<id>/result.json").read_text())
apply_planar_joints(object_model, result, layers={"coupler": 0.008})
planar_targets = [coupler.at((tip_x, tip_y, 0))]  # Actual CAD feature, in task target order.
```

For an orientation target, give the BodyFrame the feature's local yaw. Omit `planar_targets` for driver-only tasks. The joint helper keeps all physical joints, selects a spanning tree and rebases SDK joint coordinates to zero at the first solved pose. Optional layers offset bodies along Z.

In `run_tests`, iterate `planar_motion_states(object_model, result)` and use `with ctx.state(state):` for mesh clearance checks at solved and intermediate poses. Compile validates the coupled motion using the saved poses, plus position and yaw at the target waypoints. Use authored mesh checks for clearance and joint contact along those poses. Intermediate checks cover closure and drive continuity. Use enough target samples for the requested curve accuracy. Repair mesh-only issues without changing the synthesis task or rerunning the optimizer.
