"""Build and verify SDK joints from a saved planar linkage solution."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from articraft.sdk.assembly import (
    JointAxis,
    JointDOF,
    PhysicsState,
    RigidBodyAssembly,
    _frame_matrix,
)
from articraft.sdk.frames import BodyFrame, JointFrame
from articraft.sdk.testing import TestContext

_AXES = {"revolute": JointAxis.ROT_Z, "prismatic": JointAxis.TRANS_X}


def apply_planar_joints(
    model: RigidBodyAssembly,
    result: Mapping[str, Any],
    *,
    layers: Mapping[str, float] | None = None,
) -> None:
    """Connect existing bodies using optimized XY frames and optional body Z offsets."""
    if result["status"] != "success":
        raise ValueError("Apply only a successful planar synthesis result")
    mechanism = result["mechanism"]
    bodies = {body["id"]: body for body in mechanism["bodies"]}
    ground = next(name for name, body in bodies.items() if body.get("ground"))
    if model.joints or model.articulations or {b.name for b in model.rigid_bodies} != bodies.keys():
        raise ValueError("Create exactly the solver's named bodies before applying planar joints")
    heights = dict(layers or {})
    if heights.keys() - bodies.keys() or heights.get(ground, 0) != 0:
        raise ValueError("Layers must name solver bodies; the ground layer must be zero")
    initial = result["motion"]["joint_coordinates"][0]
    components = {name: {name} for name in bodies}
    tree = []
    for joint in mechanism["joints"]:
        name, kind = joint["id"], joint["type"]
        a, af = joint["a"]
        b, bf = joint["b"]
        x, y, yaw = bodies[b]["frames"][bf]["pose"]
        ax, ay, ayaw = bodies[a]["frames"][af]["pose"]
        zero = initial[name]
        # The SDK assembles at zero. Rebase the B frame and limits to sample 0.
        if kind == "revolute":
            yaw -= zero
        elif kind == "prismatic":
            x -= zero * math.cos(yaw)
            y -= zero * math.sin(yaw)
        dofs = []
        if kind in _AXES:
            limits = None
            if "limits" in joint:
                lo, hi = joint["limits"]
                limits = min(0.0, lo - zero), max(0.0, hi - zero)
            dofs.append(JointDOF(_AXES[kind], limits=limits))
        model.joint(
            name,
            model.get_rigid_body(a).at(JointFrame((ax, ay, -heights.get(a, 0)), (0, 0, ayaw))),
            model.get_rigid_body(b).at(JointFrame((x, y, -heights.get(b, 0)), (0, 0, yaw))),
            dofs=dofs,
        )
        if b not in components[a]:
            tree.append(name)
            group = components[a] | components[b]
            for body in group:
                components[body] = group
    model.articulation("planar", root=ground, joints=tree)


def _planar_states(
    model: RigidBodyAssembly, result: Mapping[str, Any]
) -> Iterator[tuple[int | None, PhysicsState]]:
    mechanism = result["mechanism"]
    expected = {j["id"]: j for j in mechanism["joints"]}
    resolved = model.resolve()
    joints = {item.joint.name: item.joint for item in resolved.joints}
    if not resolved.has_closed_loops or joints.keys() != expected.keys():
        raise ValueError("Rebuilt model must contain the original planar joint graph and its loops")
    for name, joint in joints.items():
        spec = expected[name]
        axis = _AXES.get(spec["type"])
        ends = [model.get_rigid_body(spec[side][0]) for side in ("a", "b")]
        if (joint.body0, joint.body1) != tuple(ends) or tuple(dof.axis for dof in joint.dofs) != (
            () if axis is None else (axis,)
        ):
            raise ValueError(f"Rebuilt planar joint {name!r} changed endpoints or axis")

    motion = result["motion"]
    waypoints = motion["body_poses"]
    initial = motion["joint_coordinates"][0]
    reference = resolved.reference_state
    ground = next(body["id"] for body in mechanism["bodies"] if body.get("ground"))
    base = reference.matrix(ground)
    base_inverse = np.linalg.inv(base)
    offsets = {
        body.name: float((base_inverse @ reference.matrix(body))[2, 3])
        for body in resolved.rigid_bodies
    }
    intermediate = iter(motion["intermediate"])
    pending = next(intermediate, None)

    def state(poses: Mapping[str, Sequence[float]], coords: Mapping[str, float]) -> PhysicsState:
        matrices = {
            body: base @ _frame_matrix(JointFrame((x, y, offsets[body]), (0, 0, yaw)))
            for body, (x, y, yaw) in poses.items()
        }
        positions = {
            joint.dof_id(dof): coords[name] - initial[name]
            for name, joint in joints.items()
            for dof in joint.dofs
        }
        return resolved.validate_state(PhysicsState(matrices, dof_positions=positions))

    for index, (sample, coordinates) in enumerate(
        zip(waypoints, motion["joint_coordinates"], strict=True)
    ):
        checked = state(sample["bodies"], coordinates)
        if index == 0 and any(
            not np.allclose(checked.matrix(name), reference.matrix(name), atol=1e-6, rtol=0)
            for name in offsets
        ):
            raise ValueError("Rebuilt reference pose must match the first solved planar pose")
        yield index, checked
        while pending is not None and pending["interval"] == index:
            yield None, state(pending["bodies"], pending["joint_coordinates"])
            pending = next(intermediate, None)


def planar_motion_states(
    model: RigidBodyAssembly, result: Mapping[str, Any]
) -> Iterator[PhysicsState]:
    """Yield checked waypoint and intermediate states for authored mesh tests."""
    for _, state in _planar_states(model, result):
        yield state


def verify_planar_motion(
    ctx: TestContext,
    task: Path,
    result_path: Path,
    targets: Sequence[BodyFrame],
    *,
    preview_path: Path | None = None,
) -> None:
    """Check closed-loop motion and target poses on the rebuilt SDK model."""
    raw = task.read_bytes()
    spec = json.loads(raw)
    result = json.loads(result_path.read_text())
    if hashlib.sha256(raw).hexdigest() != result["task_digest"]:
        raise ValueError("Planar task changed since synthesis")
    if not ctx.check("planar synthesis accepted", result["status"] == "success"):
        return
    declared = spec.get("targets", [])
    if len(targets) != len(declared):
        features = ", ".join(
            f"{target['body']}.{target.get('frame', target.get('point', 'body origin'))}"
            for target in declared
        )
        raise ValueError(
            f"main.py planar_targets has {len(targets)} entries; expected {len(declared)}. "
            f"Declare the actual CAD BodyFrames in task order ({features or 'no targets'}), "
            "then recompile using the saved synthesis result."
        )
    for target, frame in zip(declared, targets, strict=True):
        if not isinstance(frame, BodyFrame) or frame.body is not ctx.model.get_rigid_body(
            target["body"]
        ):
            raise ValueError("Each planar target frame must belong to its declared body")
    ground = next(body["id"] for body in spec["bodies"] if body.get("ground"))
    position_error = angle_error = 0.0
    count = 0
    frames = []
    for index, state in _planar_states(ctx.model, result):
        count += 1
        if preview_path is not None:
            bodies = {}
            for name in state.body_poses:
                matrix = state.matrix(name)
                bodies[name] = {
                    "pos": matrix[:3, 3].tolist(),
                    "quat": Rotation.from_matrix(matrix[:3, :3])
                    .as_quat(scalar_first=True)
                    .tolist(),
                }
            frames.append({"bodies": bodies, "dofs": dict(state.dof_positions)})
        if index is None:
            continue
        base_inverse = np.linalg.inv(state.matrix(ground))
        for target, frame in zip(declared, targets, strict=True):
            pose = base_inverse @ state.matrix(target["body"]) @ _frame_matrix(frame.frame)
            if "positions" in target:
                position_error = max(
                    position_error,
                    float(np.linalg.norm(pose[:2, 3] - target["positions"][index])),
                )
            if "orientations" in target:
                angle = math.atan2(pose[1, 0], pose[0, 0]) - target["orientations"][index]
                angle_error = max(angle_error, abs(math.remainder(angle, math.tau)))
    options = spec.get("options", {})
    ctx.record_metric("planar waypoint position error", position_error, unit="m")
    ctx.record_metric("planar waypoint orientation error", angle_error, unit="rad")
    ctx.check(
        "planar waypoint position", position_error <= options.get("trajectory_tolerance", 0.001)
    )
    ctx.check(
        "planar waypoint orientation", angle_error <= options.get("orientation_tolerance", 0.05)
    )
    ctx.check("planar loop motion", count > 1)
    if preview_path is not None and ctx.report().passed:
        preview = {"schema_version": 1, "body_frames": frames}
        preview_path.write_text(json.dumps(preview, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"[trajectory] Checked {count} rebuilt planar states: "
        f"waypoint error={position_error:.6g} m, orientation error={angle_error:.6g} rad",
        file=sys.stderr,
        flush=True,
    )
