"""A small serial-chain bridge between authored assemblies and synthesis."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from articraft.sdk._values import _as_vec3
from articraft.sdk.assembly import Joint, JointAxis, RigidBodyAssembly, _frame_matrix
from articraft.sdk.bodies import RigidBody, RigidBodyRef
from articraft.sdk.frames import BodyFrame, JointFrame, _matrix_rpy
from articraft.sdk.testing import TestContext


def _chain(
    model: RigidBodyAssembly, base: RigidBodyRef, tip: BodyFrame
) -> tuple[RigidBody, list[tuple[Joint, bool]]]:
    if not isinstance(tip.body, RigidBody):
        raise ValueError("The trajectory tip must belong to a rigid body")
    base = model.get_rigid_body(base)
    if model.get_rigid_body(tip.body) is not tip.body:
        raise ValueError("The trajectory tip belongs to another assembly")
    if model.resolve().has_closed_loops:
        raise ValueError("Trajectory synthesis currently supports open serial chains, not loops")
    pending: list[tuple[RigidBody, list[tuple[Joint, bool]]]] = [(base, [])]
    seen = {base.name}
    for body, path in pending:
        if body is tip.body:
            if not any(joint.dofs for joint, _ in path):
                raise ValueError("The base-to-tip chain must contain a movable joint")
            for joint, _ in path:
                if len(joint.dofs) > 1:
                    raise ValueError(f"Multi-axis joint {joint.name!r} is unsupported")
            return base, path
        for joint in model.joints:
            if joint.body0 is body:
                child, forward = joint.body1, True
            elif joint.body1 is body:
                child, forward = joint.body0, False
            else:
                continue
            if isinstance(child, RigidBody) and child.name not in seen:
                seen.add(child.name)
                pending.append((child, [*path, (joint, forward)]))
    raise ValueError("No base-to-tip chain exists")


def _origin(element: ET.Element, matrix: np.ndarray) -> None:
    ET.SubElement(
        element,
        "origin",
        xyz=" ".join(f"{v:.16g}" for v in matrix[:3, 3]),
        rpy=" ".join(f"{v:.16g}" for v in _matrix_rpy(matrix[:3, :3])),
    )


def _read_tcp(spec: dict[str, Any], urdf_path: Path) -> dict[str, Any]:
    """Read the declared working frame from the URDF, never duplicate its pose in JSON."""
    metadata = spec.get("tcp", {})
    description = metadata.get("description") if isinstance(metadata, dict) else None
    axes = metadata.get("axes") if isinstance(metadata, dict) else None
    if (
        not isinstance(description, str)
        or not description.strip()
        or not isinstance(axes, dict)
        or any(not isinstance(axes.get(axis), str) or not axes[axis].strip() for axis in "xyz")
    ):
        raise ValueError(
            "Task tcp must include a description of the functional working point and "
            "axes with nonempty x, y, z descriptions. Place the URDF tip frame there."
        )
    mount = None
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") == spec["tip_link"]:
            mount = joint
            break
    if mount is None or mount.get("type") != "fixed":
        raise ValueError("The trajectory tip_link must be a TCP frame attached by a fixed joint")
    parent, origin = mount.find("parent"), mount.find("origin")
    if parent is None:
        raise ValueError("The TCP fixed joint must name its parent body")
    frame = JointFrame(
        xyz=_as_vec3(
            list(map(float, origin.get("xyz", "0 0 0").split()))
            if origin is not None
            else (0, 0, 0),
            field_name="TCP xyz",
        ),
        rpy=_as_vec3(
            list(map(float, origin.get("rpy", "0 0 0").split()))
            if origin is not None
            else (0, 0, 0),
            field_name="TCP rpy",
        ),
    )
    return {
        "body": parent.get("link"),
        "xyz_m": list(frame.xyz),
        "rpy_rad": list(frame.rpy),
        "description": description.strip(),
        "axes": {axis: axes[axis].strip() for axis in "xyz"},
    }


def export_kinematic_urdf(
    model: RigidBodyAssembly,
    path: Path | str,
    *,
    base: RigidBodyRef,
    tip: BodyFrame,
) -> dict[str, str]:
    """Export the base-to-tip kinematic chain; no meshes, collision, or inertia.

    The returned fields can be merged into a synthesis task JSON. Joint names
    are preserved. Each joint has an auxiliary fixed ``__trajectory_body_NAME``
    joint to account for the child-side frame. ``__trajectory_tip_mount`` places
    the tracked frame, whose URDF link is ``__trajectory_tip``.
    """
    base, chain = _chain(model, base, tip)
    if any(body.name.startswith("__trajectory_") for body in model.rigid_bodies) or any(
        joint.name.startswith("__trajectory_") for joint in model.joints
    ):
        raise ValueError("Names beginning __trajectory_ are reserved by the URDF exporter")
    root = ET.Element("robot", name=model.name)
    ET.SubElement(root, "link", name=base.name)
    for joint, forward in chain:
        parent = cast(RigidBody, joint.body0 if forward else joint.body1)
        child = cast(RigidBody, joint.body1 if forward else joint.body0)
        frame0 = joint.frame0 if forward else joint.frame1
        frame1 = joint.frame1 if forward else joint.frame0
        virtual = f"__trajectory_frame_{joint.name}"
        ET.SubElement(root, "link", name=virtual)
        ET.SubElement(root, "link", name=child.name)
        kind = "fixed"
        if joint.dofs:
            dof = joint.dofs[0]
            axis = JointAxis(dof.axis)
            kind = "revolute" if axis.is_rotational else "prismatic"
            if axis.is_rotational and dof.limits is None:
                kind = "continuous"
        element = ET.SubElement(root, "joint", name=joint.name, type=kind)
        ET.SubElement(element, "parent", link=parent.name)
        ET.SubElement(element, "child", link=virtual)
        _origin(element, _frame_matrix(frame0))
        if joint.dofs:
            dof = joint.dofs[0]
            axis = JointAxis(dof.axis)
            direction = np.eye(3)[axis.component] * (1 if forward else -1)
            ET.SubElement(element, "axis", xyz=" ".join(map(str, direction)))
            if dof.limits is not None:
                ET.SubElement(
                    element,
                    "limit",
                    lower=str(dof.limits[0]),
                    upper=str(dof.limits[1]),
                    effort="1",
                    velocity="1",
                )
        fixed = ET.SubElement(root, "joint", name=f"__trajectory_body_{joint.name}", type="fixed")
        ET.SubElement(fixed, "parent", link=virtual)
        ET.SubElement(fixed, "child", link=child.name)
        _origin(fixed, np.linalg.inv(_frame_matrix(frame1)))
    ET.SubElement(root, "link", name="__trajectory_tip")
    mount = ET.SubElement(root, "joint", name="__trajectory_tip_mount", type="fixed")
    ET.SubElement(mount, "parent", link=cast(RigidBody, tip.body).name)
    ET.SubElement(mount, "child", link="__trajectory_tip")
    _origin(mount, _frame_matrix(tip.frame))
    path = Path(path)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[trajectory] Exported initial kinematics: {path}", file=sys.stderr, flush=True)
    print(
        f"[trajectory] Exported TCP: body={cast(RigidBody, tip.body).name}, "
        f"xyz_m={tip.xyz}, rpy_rad={tip.rpy}",
        file=sys.stderr,
        flush=True,
    )
    return {"urdf": path.name, "base_link": base.name, "tip_link": "__trajectory_tip"}


class _AssemblyKinematics:
    def __init__(
        self,
        model: RigidBodyAssembly,
        base: RigidBodyRef,
        tip: BodyFrame,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        self.base, self.chain = _chain(model, base, tip)
        self.resolved = model.resolve()
        self.tip = tip
        movable = [joint for joint, _ in self.chain if joint.dofs]
        self.joint_names = [joint.name for joint in movable]
        self.dof_ids = [joint.dof_id(joint.dofs[0]) for joint in movable]
        self.lower, self.upper = lower.copy(), upper.copy()
        if len(lower) != len(movable):
            raise ValueError("Rebuilt assembly changed the number of trajectory joints")
        for i, joint in enumerate(movable):
            if joint.dofs[0].limits is not None:
                lo, hi = joint.dofs[0].limits
                self.lower[i] = max(self.lower[i], lo)
                self.upper[i] = min(self.upper[i], hi)

    def fk(self, design: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, rotations, nodes = [], [], []
        for row in q:
            state = self.resolved.forward_kinematics(dict(zip(self.dof_ids, row, strict=True)))
            base_inverse = np.linalg.inv(state.matrix(self.base))
            tool = (
                base_inverse
                @ state.matrix(cast(RigidBody, self.tip.body))
                @ _frame_matrix(self.tip.frame)
            )
            positions.append(tool[:3, 3])
            rotations.append(tool[:3, :3])
            points = [np.zeros(3)]
            for joint, forward in self.chain:
                body = cast(RigidBody, joint.body0 if forward else joint.body1)
                frame = joint.frame0 if forward else joint.frame1
                points.append((base_inverse @ state.matrix(body) @ _frame_matrix(frame))[:3, 3])
                child = cast(RigidBody, joint.body1 if forward else joint.body0)
                if joint.is_prismatic:
                    child_frame = joint.frame1 if forward else joint.frame0
                    points.append(
                        (base_inverse @ state.matrix(child) @ _frame_matrix(child_frame))[:3, 3]
                    )
                points.append((base_inverse @ state.matrix(child))[:3, 3])
            points.append(tool[:3, 3])
            nodes.append(points)
        return np.asarray(positions), np.asarray(rotations), np.asarray(nodes)


def verify_trajectory(
    ctx: TestContext,
    task_path: Path | str,
    result_dir: Path | str,
    *,
    base: RigidBodyRef,
    tip: BodyFrame,
    preview_path: Path | str | None = None,
) -> dict[str, Any]:
    """Check saved motion on the rebuilt SDK model, against the original curve.

    Call after applying optimized dimensions. The compile worker automatically
    calls this for tasks registered by the synthesis tool. Additional physical
    collision checks can use the saved joint path with TestContext pose helpers.
    """
    from articraft.sdk._synthesis import InnerResult, load_problem, task_digest

    print("[trajectory] Verifying rebuilt SDK geometry and motion", file=sys.stderr, flush=True)
    task_path, result_dir = Path(task_path), Path(result_dir)
    report = json.loads((result_dir / "result.json").read_text())
    if task_digest(task_path) != report["task_digest"]:
        raise ValueError("Trajectory task or equation changed; rerun synthesize_trajectory")
    solver, spec = load_problem(task_path)
    if not ctx.check("trajectory synthesis accepted", report["functional_success"]):
        return {"passes": False}
    with (result_dir / "joint_path.csv").open(newline="") as file:
        reader = csv.reader(file)
        if next(reader) != ["waypoint", *report["joint_order"]]:
            raise ValueError("Saved joint path column order disagrees with synthesis result")
        rows = np.asarray(list(reader), dtype=float)
    if rows.shape != (solver.n, solver.dof + 1) or not np.isfinite(rows).all():
        raise ValueError("Saved joint path has invalid shape or values")
    if not np.array_equal(rows[:, 0], np.arange(solver.n)):
        raise ValueError("Saved joint path waypoint order is invalid")
    q = rows[:, 1:]
    design = np.asarray([report["design"][p["name"]] for p in spec["design_parameters"]])
    actual = _AssemblyKinematics(ctx.model, base, tip, solver.model.lower, solver.model.upper)
    tcp = {
        "body": cast(RigidBody, tip.body).name,
        "xyz_m": list(tip.xyz),
        "rpy_rad": list(tip.rpy),
    }
    tcp_matches = True
    if "tcp" in spec:
        expected_tcp = _read_tcp(spec, result_dir / "optimized_kinematics.urdf")
        tcp.update(description=expected_tcp["description"], axes=expected_tcp["axes"])
        expected_frame = np.eye(4)
        expected_frame[:3, 3] = expected_tcp["xyz_m"]
        expected_frame[:3, :3] = Rotation.from_euler("xyz", expected_tcp["rpy_rad"]).as_matrix()
        tcp_matches = expected_tcp["body"] == tcp["body"] and bool(
            np.allclose(expected_frame, _frame_matrix(tip.frame), atol=1e-7, rtol=0)
        )
        ctx.check(
            "trajectory TCP matches optimized frame",
            tcp_matches,
            f"Expected {expected_tcp}; rebuilt {tcp}. Apply the optimized TCP to the geometry. "
            "Rerun synthesis when deliberately changing the working point or axes.",
        )
    print(f"[trajectory] Rebuilt functional TCP: {tcp}", file=sys.stderr, flush=True)
    if (
        actual.joint_names != report["joint_order"]
        or actual.joint_names != solver.model.joint_names
    ):
        raise ValueError("Rebuilt assembly changed the trajectory joint order or names")
    endpoints_pass = True
    for label, value, endpoint in (
        ("start", solver.task.q_start, q[0]),
        ("end", solver.task.q_end, q[-1]),
    ):
        if value is not None:
            matches = bool(np.allclose(value, endpoint, atol=1e-9, rtol=0))
            endpoints_pass = endpoints_pass and matches
            ctx.check(f"trajectory fixed {label}", matches)
    # Use the SDK's own FK, not the optimized URDF, for all acceptance checks.
    solver.model = actual
    solver.q_range = actual.upper - actual.lower
    best = InnerResult(design, q, 0, 0, True, True, "Rebuilt SDK assembly")
    validation = solver.validate_dense(best, report["subdivisions"])
    validation["endpoints_pass"] = endpoints_pass
    validation["tcp_frame_pass"] = tcp_matches
    validation["passes"] = validation["passes"] and endpoints_pass and tcp_matches
    ctx.record_metric("trajectory maximum error", validation["max_error_m"], unit="m")
    ctx.check("trajectory rebuilt tracking", validation["position_pass"], str(validation))
    ctx.check("trajectory rebuilt constraints", validation["constraints_pass"], str(validation))
    if preview_path is not None:
        dense_q, targets, target_rotations = solver.dense_path(best, report["subdivisions"])
        achieved, rotations, _ = actual.fk(design, dense_q)
        preview = {
            "schema_version": 1,
            "base_body": actual.base.name,
            "tcp": tcp,
            "target_positions_m": targets.tolist(),
            "achieved_positions_m": achieved.tolist(),
            "target_quaternions_xyzw": (
                None
                if target_rotations is None
                else Rotation.from_matrix(target_rotations).as_quat().tolist()
            ),
            "achieved_quaternions_xyzw": Rotation.from_matrix(rotations).as_quat().tolist(),
            "joint_order": actual.joint_names,
            "joint_path": dense_q.tolist(),
            "validation": validation,
        }
        Path(preview_path).write_text(json.dumps(preview, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[trajectory] Rebuilt SDK validation: {validation}", file=sys.stderr, flush=True)
    return validation
