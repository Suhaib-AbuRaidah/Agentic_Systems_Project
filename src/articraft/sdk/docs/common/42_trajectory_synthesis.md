# End-effector trajectory synthesis

Use `synthesize_trajectory` for explicit end-effector paths. It optimizes dimensions and joint motion on an open serial chain.
Supported joints are fixed, revolute, continuous, and prismatic, with at most one axis per joint.
Preserve the requested mechanism and part roles. Closed loops, topology synthesis, and dynamics are unsupported.

## Model and TCP

Share dimension parameters between geometry, joints, and the tool center point (TCP).
Define these globals in `main.py` using the current geometry:

```python
from articraft.sdk import JointFrame

trajectory_base = object_model.get_rigid_body("base")
tool = object_model.get_rigid_body("end_effector")
trajectory_tip = tool.at(JointFrame(xyz=(tool_length_m, 0, 0), rpy=(0, 0, 0)))
```

The TCP origin is the functional working point, such as the midpoint between gripper tips.
Its right-handed axes describe tool function. For example, +X follows the fingers, +Y crosses the jaw gap, and +Z completes the frame.
A body origin is a valid TCP only when it is the working point.
Both position and orientation targets describe this frame relative to `trajectory_base`.

Compile a simple model first. Export once from a separate setup script with `export_kinematic_urdf`:

```python
import runpy
from articraft.sdk import export_kinematic_urdf

initial = runpy.run_path("main.py")
fields = export_kinematic_urdf(
    initial["object_model"], "initial.urdf",
    base=initial["trajectory_base"], tip=initial["trajectory_tip"],
)
```

The returned fields identify the URDF, base link, and TCP link.
The fixed joint `__trajectory_tip_mount` places `__trajectory_tip` on the tool body.
Auxiliary `__trajectory_body_NAME` joints account for child frames. Joint names and coordinates remain unchanged.
Keep the export outside `main.py` so imports cannot overwrite synthesis inputs.

## Task JSON

Adapt this example to the exported model's dimensions, joints, and target.

```json
{
  "urdf": "initial.urdf",
  "base_link": "base",
  "tip_link": "__trajectory_tip",
  "tcp": {
    "description": "Working point between the finger tips",
    "axes": {"x": "Along fingers", "y": "Across jaw gap", "z": "Palm normal"}
  },
  "trajectory": {
    "module": "trajectory.py", "function": "positions",
    "parameter_range": [0, 1], "samples": 21
  },
  "waypoints": {"quaternions_xyzw": [0, 0, 0, 1]},
  "design_parameters": [{
    "name": "upper_length_m", "initial": 0.2, "lower": 0.1, "upper": 0.3,
    "bindings": [{"joint": "elbow", "xyz": [1, 0, 0]}]
  }],
  "position_tolerance_ratio": 0.03,
  "orientation_tolerance_rad": 0.03490658503988659,
  "max_joint_step": [0.2, 0.2, 0.2]
}
```

The target function accepts a parameter array and returns finite positions with shape `(N, 3)` in metres:

```python
import numpy as np

def positions(t):
    return np.column_stack((0.1 + 0.1*t, 0.05*t, np.full_like(t, 0.2)))
```

For sampled targets, replace `trajectory` with `waypoints.positions_m`. Dense checks use the original equation or the waypoint polyline.
A binding applies `origin = initial_origin + binding * (parameter - initial)`.
Bindings support `xyz` and `rpy` vectors. Initial parameter values must match the exported URDF.
The task requires `tcp.description` and x/y/z axis descriptions. The TCP pose comes from the fixed URDF joint.

## Controls

| Field | Meaning |
| --- | --- |
| `trajectory.samples` | Default 21 optimization waypoints, including endpoints. |
| `position_tolerance_ratio` | Default `0.03` times the target bounding-box diagonal. Equation bounds use 201 target samples. |
| `position_tolerance_m` | Explicit absolute limit, overriding the relative tolerance. Required for zero-span targets. |
| `waypoints.quaternions_xyzw` | One `[x,y,z,w]` quaternion for fixed orientation, or one row per position sample for changing orientation. Dense rotations use SLERP. |
| `orientation_tolerance_rad` | Default 5 degrees in radians. Requires an orientation target. The example uses 2 degrees. |
| `q_start`, `q_end` | Optional fixed joint configurations in movable-joint order. Leave free unless requested. |
| `max_joint_step` | Positive increments per sample in radians or metres. This is not a speed limit. |
| `joint_bounds` | Finite `[lower, upper]` overrides by joint name. Required for unbounded joints. SDK limits still apply. |
| `sphere_obstacles` | `[x,y,z,radius]` rows with optional `capsule_radius_m` and `clearance_m`. These approximate chain clearance only. |
| `length_scale_m` | Position normalization, default 1 metre. |
| `settings` | Solver budgets: `outer_maxeval` (default and cap 10), `inner_maxiter`, `multistarts`. |
| `settings.inner_ftol` | Inner numerical convergence tolerance, default `1e-7`. |
| `settings.outer_tol` | Normalized dimension convergence tolerance, default `1e-3`. |

Limits are 200 samples, 20 joints, 30 dimensions, 1000 inner iterations, 10 geometry evaluations (including the initial design), and 8 starts.
Optimization stops early when waypoint and dense position errors are both at most 10% of the position tolerance.
Configured constraints must also pass. Otherwise, optimization runs to numerical convergence or its evaluation limit.
Final acceptance uses the full position tolerance for both waypoint and dense maximum errors.
Keep requested orientation targets and tolerances when correcting input errors.

## Solve and verify

Call the tool with `task_path: "trajectory_task.json"` and no concurrent workspace tools.
Optional `timeout_seconds` defaults to 180, maximum 900. `subdivisions` defaults to 10, range 1 to 100.
Feedback contains dimensions, TCP, joint summaries, validation, and file paths.
Full outputs stay under the returned `result_dir`:

| File | Contents |
| --- | --- |
| `result.json` | Dimensions, acceptance, and validation. |
| `joint_path.csv` | Waypoint index and joint coordinates in reported order. |
| `history.json` | Evaluated designs. |
| `optimized_kinematics.urdf` | Updated joint origins, without redesigned geometry. |

Apply accepted values to actual geometry, joint frames, and `trajectory_tip`, then compile.
For `needs_revision`, fix the numerical cause and rerun without weakening requirements.
Compile verifies the rebuilt TCP and saved motion. Add `TestContext` mesh collision checks along that motion.
`verify_trajectory` provides the same numerical check for local scripts.
Any task JSON edit, including descriptions, or a URDF or curve edit invalidates the saved solve.
Keep these inputs unchanged during geometry-only repairs. Recompile to check the saved motion against repaired geometry.
Compile saves paths and TCP data for the viewer automatically. Keep numerical paths in files, outside the model conversation.
Finite samples do not prove continuous tracking or collision freedom. Refine sampling when intermediate checks fail.
