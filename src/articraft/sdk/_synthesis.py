#!/usr/bin/env python3
"""General nested dimensional synthesis for sampled kinematic trajectories.

NumPy + SciPy; no dynamics. A model implements batched FK and joint limits.
The included URDF adapter supports arbitrary fixed/revolute/prismatic serial
chains. It parameterizes joint-origin translations and rotations, not meshes.
Adapted from Project_extra_tools/synthesis.py. See the SDK trajectory reference
for task inputs and limitations. Geometry is rebuilt by the authoring agent.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.transform import Rotation, Slerp

Array: TypeAlias = NDArray[np.float64]


class KinematicModel(Protocol):
    joint_names: list[str]
    lower: Array
    upper: Array

    def fk(self, design: Array, q: Array) -> tuple[Array, Array, Array]:
        """Return positions (N,3), rotations (N,3,3), chain nodes (N,K,3).

        q is (N,DOF). All positions use metres; angles use radians.
        Nodes are optional: return (N,0,3) if no capsule obstacles are used.
        """
        ...


def vector(text: str | None, default: list[float]) -> Array:
    return np.asarray(default if text is None else list(map(float, text.split())), float)


@dataclass
class URDFJoint:
    name: str
    kind: str
    xyz: Array
    rpy: Array
    axis: Array
    coordinate: int | None


class SerialURDF:
    """Kinematic chain from base_link to tip_link, in the base-link frame.

    Design parameter bindings are affine increments about the original URDF:
        xyz_j(d) = xyz_j(original) + sum b_xyz * (d[k] - d_initial[k])
        rpy_j(d) = rpy_j(original) + sum b_rpy * (d[k] - d_initial[k]).
    A parameter can bind several joints; several parameters can bind one joint.
    Mimic joints and unsupported joint types are explicitly rejected.
    """

    def __init__(
        self,
        path: str | Path,
        base_link: str,
        tip_link: str,
        parameters: list[dict],
        joint_bounds: dict | None = None,
    ) -> None:
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        links = {x.attrib["name"] for x in root.findall("link")}
        if base_link not in links or tip_link not in links:
            raise ValueError("base_link and tip_link must exist in the URDF")
        parent_of = {}
        for element in root.findall("joint"):
            child_element = element.find("child")
            if child_element is None:
                raise ValueError("URDF joint is missing its child link")
            child = child_element.attrib["link"]
            if child in parent_of:
                raise ValueError("URDF must be a tree; multiple parent joints found")
            parent_of[child] = element
        chain, visited = [], set()
        link = tip_link
        while link != base_link:
            if link in visited or link not in parent_of:
                raise ValueError("No acyclic chain from base_link to tip_link")
            visited.add(link)
            element = parent_of[link]
            chain.append(element)
            link = element.find("parent").attrib["link"]
        chain.reverse()
        if not chain:
            raise ValueError("Choose different base and tip links")
        self.joints, self.joint_names = [], []
        lo, hi = [], []
        joint_bounds = joint_bounds or {}
        for element in chain:
            name, kind = element.attrib["name"], element.attrib["type"]
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"Unsupported joint type: {name}: {kind}")
            if element.find("mimic") is not None:
                raise ValueError(f"Mimic joint {name}: supply a custom model instead")
            origin, axis = element.find("origin"), element.find("axis")
            xyz = vector(None if origin is None else origin.get("xyz"), [0, 0, 0])
            rpy = vector(None if origin is None else origin.get("rpy"), [0, 0, 0])
            axis_v = vector(None if axis is None else axis.get("xyz"), [1, 0, 0])
            coordinate = None
            if kind != "fixed":
                if np.linalg.norm(axis_v) == 0:
                    raise ValueError(f"Zero joint axis: {name}")
                axis_v /= np.linalg.norm(axis_v)
                coordinate = len(lo)
                self.joint_names.append(name)
                limit = element.find("limit")
                if name in joint_bounds:
                    lower, upper = joint_bounds[name]
                elif kind == "continuous":
                    raise ValueError(f"Supply finite unwrapped joint_bounds for {name}")
                elif limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
                    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                else:
                    raise ValueError(f"Missing joint bounds: {name}")
                lo.append(lower)
                hi.append(upper)
            self.joints.append(URDFJoint(name, kind, xyz, rpy, axis_v, coordinate))
        self.lower, self.upper = np.array(lo, float), np.array(hi, float)
        if not len(lo) or np.any(self.lower >= self.upper):
            raise ValueError("Need movable joints with finite lower < upper bounds")
        if not np.all(np.isfinite([self.lower, self.upper])):
            raise ValueError("Joint limits must be finite")
        self.reference = np.array([p["initial"] for p in parameters], float)
        lookup = {j.name: i for i, j in enumerate(self.joints)}
        self.xyz_basis = np.zeros((len(chain), len(parameters), 3))
        self.rpy_basis = np.zeros_like(self.xyz_basis)
        for k, p in enumerate(parameters):
            if not p.get("bindings"):
                raise ValueError(f"Design parameter {p['name']} has no bindings")
            for b in p["bindings"]:
                j = lookup[b["joint"]]
                self.xyz_basis[j, k] += np.asarray(b.get("xyz", [0, 0, 0]), float)
                self.rpy_basis[j, k] += np.asarray(b.get("rpy", [0, 0, 0]), float)

    def fk(self, design: Array, q: Array) -> tuple[Array, Array, Array]:
        q = np.atleast_2d(q)
        n = len(q)
        rotations = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        positions = np.zeros((n, 3))
        nodes = [positions.copy()]
        delta = np.asarray(design) - self.reference
        for i, j in enumerate(self.joints):
            xyz = j.xyz + delta @ self.xyz_basis[i]
            rpy = j.rpy + delta @ self.rpy_basis[i]
            positions += np.einsum("nij,j->ni", rotations, xyz)
            rotations = rotations @ Rotation.from_euler("xyz", rpy).as_matrix()
            nodes.append(positions.copy())
            if j.kind in {"revolute", "continuous"}:
                motion = Rotation.from_rotvec(q[:, j.coordinate, None] * j.axis).as_matrix()
                rotations = rotations @ motion
            elif j.kind == "prismatic":
                positions += np.einsum("nij,nj->ni", rotations, q[:, j.coordinate, None] * j.axis)
                nodes.append(positions.copy())
        return positions, rotations, np.stack(nodes, axis=1)

    def write_kinematic_urdf(self, design: Array, path: str | Path) -> None:
        """Update joint origins ONLY. Not a mesh/collision/inertial redesign."""
        delta = np.asarray(design) - self.reference
        by_name = {x.attrib["name"]: x for x in self.tree.getroot().findall("joint")}
        for i, j in enumerate(self.joints):
            element = by_name[j.name]
            origin = element.find("origin")
            if origin is None:
                origin = ET.SubElement(element, "origin")
            for key, value in [
                ("xyz", j.xyz + delta @ self.xyz_basis[i]),
                ("rpy", j.rpy + delta @ self.rpy_basis[i]),
            ]:
                origin.set(key, " ".join(f"{v:.12g}" for v in value))
        ET.indent(self.tree, space="  ")
        self.tree.write(path, encoding="utf-8", xml_declaration=True)


@dataclass
class Task:
    positions: Array
    q_start: Array | None
    q_end: Array | None
    max_step: Array
    position_tolerance: float = 1e-3
    length_scale: float = 1.0
    rotations: Array | None = None
    orientation_tolerance: float = np.deg2rad(5)
    # Each sphere: [x, y, z, radius]. Robot links are capsule approximations.
    obstacles: Array = field(default_factory=lambda: np.empty((0, 4)))
    link_radius: float = 0.0
    clearance: float = 0.0
    # Return dimensionless margins >= 0. Can implement richer collisions,
    # coupled design restrictions, or loop-closure tolerances here.
    extra_margins: Callable[[Array, Array], Array] | None = None
    target_positions: Callable[[Array], Array] | None = None


@dataclass
class Settings:
    inner_maxiter: int = 250
    outer_maxeval: int = 10
    inner_ftol: float = 1e-7
    feasibility_tolerance: float = 2e-6
    outer_tol: float = 1e-3
    multistarts: int = 1
    seed: int = 7


@dataclass
class InnerResult:
    design: Array
    q: Array
    max_error: float
    min_margin: float
    feasible: bool
    solver_success: bool
    message: str


class _TrackingThresholdReached(Exception):
    pass


class _GeometryBudgetReached(Exception):
    pass


class NestedSynthesis:
    """General solver; no URDF-specific or fixed-DOF logic in either loop.

    Internal: SLSQP, min z s.t. z >= normalized waypoint error norms.
    External: COBYLA over normalized design coordinates, rerunning internal
    optimization at every new trial. Infeasible inner solves return finite
    error + violations; they never silently count as successful designs.
    """

    def __init__(
        self,
        model: KinematicModel,
        task: Task,
        initial: Array,
        lower: Array,
        upper: Array,
        settings: Settings | None = None,
    ) -> None:
        self.model, self.task = model, task
        self.settings = settings or Settings()
        self.initial, self.lower, self.upper = (
            np.asarray(x, float) for x in (initial, lower, upper)
        )
        self.range = self.upper - self.lower
        if not len(initial) or np.any(self.range <= 0):
            raise ValueError("Need design parameters with lower < upper")
        if not np.all(np.isfinite([self.initial, self.lower, self.upper])):
            raise ValueError("Design values must be finite")
        if np.any(self.initial < self.lower) or np.any(self.initial > self.upper):
            raise ValueError("Initial design lies outside bounds")
        t = self.task
        t.positions = np.asarray(t.positions, float)
        t.q_start = None if t.q_start is None else np.asarray(t.q_start, float)
        t.q_end = None if t.q_end is None else np.asarray(t.q_end, float)
        t.max_step = np.asarray(t.max_step, float)
        self.n, self.dof = len(t.positions), len(model.lower)
        if t.positions.shape != (self.n, 3) or self.n < 2:
            raise ValueError("Need at least two 3D target positions")
        if not np.all(np.isfinite(t.positions)):
            raise ValueError("Target positions must be finite")
        coordinates = [x for x in (t.q_start, t.q_end, t.max_step) if x is not None]
        if any(x.shape != (self.dof,) for x in coordinates):
            raise ValueError("Endpoint configurations and max_step must have DOF entries")
        if not all(np.all(np.isfinite(x)) for x in coordinates):
            raise ValueError("Endpoint configurations and max_step must be finite")
        if np.any(t.max_step <= 0) or min(t.position_tolerance, t.length_scale) <= 0:
            raise ValueError("Step limits and position/length scales must be positive")
        self.q_range = model.upper - model.lower
        if np.any(self.q_range <= 0) or not np.all(np.isfinite(self.q_range)):
            raise ValueError("Model needs finite, nondegenerate joint bounds")
        for q in (t.q_start, t.q_end):
            if q is None:
                continue
            if np.any(q < model.lower) or np.any(q > model.upper):
                raise ValueError("Endpoint configuration violates joint bounds")
        if (
            t.q_start is not None
            and t.q_end is not None
            and np.any(np.abs(t.q_end - t.q_start) > (self.n - 1) * t.max_step + 1e-12)
        ):
            raise ValueError("Endpoints cannot be connected under supplied step bounds")
        if t.rotations is not None:
            t.rotations = np.asarray(t.rotations, float)
            if t.rotations.shape != (self.n, 3, 3):
                raise ValueError("Target rotations must be (N,3,3)")
            if not 0 < t.orientation_tolerance <= np.pi:
                raise ValueError("Orientation tolerance must lie in (0, pi]")
            if not np.allclose(
                t.rotations @ t.rotations.transpose(0, 2, 1), np.eye(3), atol=1e-7
            ) or not np.allclose(np.linalg.det(t.rotations), 1, atol=1e-7):
                raise ValueError("Target rotations must be proper rotation matrices")
        t.obstacles = np.asarray(t.obstacles, float).reshape(-1, 4)
        if np.any(t.obstacles[:, 3] < 0) or min(t.link_radius, t.clearance) < 0:
            raise ValueError("Radii and clearance must be nonnegative")
        if self.settings.multistarts < 1:
            raise ValueError("multistarts must be >= 1")
        self.cache: dict[tuple, InnerResult] = {}
        self.history: list[dict] = []
        self.verbose = False
        self.free_rows = np.ones(self.n, dtype=bool)
        if t.q_start is not None:
            self.free_rows[0] = False
        if t.q_end is not None:
            self.free_rows[-1] = False

    def decode_q(self, y: Array) -> Array:
        q = np.empty((self.n, self.dof))
        if self.task.q_start is not None:
            q[0] = self.task.q_start
        if self.task.q_end is not None:
            q[-1] = self.task.q_end
        q[self.free_rows] = self.model.lower + y.reshape(-1, self.dof) * self.q_range
        return q

    def hard_margins(
        self,
        design: Array,
        q: Array,
        fk: tuple[Array, Array, Array] | None = None,
        max_step: Array | None = None,
    ) -> Array:
        """Positive is feasible. Position tracking remains the objective."""
        t = self.task
        _, rotations, nodes = self.model.fk(design, q) if fk is None else fk
        step = t.max_step if max_step is None else max_step
        dq = np.diff(q, axis=0)
        margins = [
            (step - dq).ravel() / np.tile(self.q_range, len(dq)),
            (step + dq).ravel() / np.tile(self.q_range, len(dq)),
            ((q - self.model.lower) / self.q_range).ravel(),
            ((self.model.upper - q) / self.q_range).ravel(),
        ]
        if t.rotations is not None:
            trace = np.einsum("nij,nij->n", t.rotations, rotations)
            cosine = (trace - 1) / 2
            c = np.cos(t.orientation_tolerance)
            margins.append((cosine - c) / max(1 - c, 1e-10))
        if len(t.obstacles):
            if nodes.shape[1] < 2:
                raise ValueError("Capsule obstacles need model chain nodes")
            a, v = nodes[:, :-1], np.diff(nodes, axis=1)
            vv = np.einsum("nkj,nkj->nk", v, v)
            for sphere in t.obstacles:
                w = sphere[:3] - a
                fraction = np.clip(np.einsum("nkj,nkj->nk", w, v) / np.maximum(vv, 1e-30), 0, 1)
                closest = a + fraction[:, :, None] * v
                distance = np.linalg.norm(closest - sphere[:3], axis=2)
                margins.append(
                    ((distance - sphere[3] - t.link_radius - t.clearance) / t.length_scale).ravel()
                )
        if t.extra_margins is not None:
            margins.append(np.asarray(t.extra_margins(design, q), float).ravel())
        result = np.concatenate(margins)
        if not np.all(np.isfinite(result)):
            raise ValueError("Constraint callback returned nonfinite values")
        return result

    def solve_internal(self, design: Array) -> InnerResult:
        """Fixed deterministic seeds at every geometry; no stateful warm-start bias."""
        cfg, t = self.settings, self.task
        middle = (self.model.lower + self.model.upper) / 2
        start = middle if t.q_start is None else t.q_start
        end = middle if t.q_end is None else t.q_end
        q_seed = np.linspace(start, end, self.n)
        y_seed = ((q_seed[self.free_rows] - self.model.lower) / self.q_range).ravel()
        if not len(y_seed):
            return self._result(design, q_seed, True, "Endpoints only; no free coordinates")
        rng = np.random.default_rng(cfg.seed)
        seeds = [y_seed]
        seeds.extend(
            np.clip(y_seed + rng.normal(0, 0.12, len(y_seed)), 0, 1)
            for _ in range(cfg.multistarts - 1)
        )
        candidates = []
        for seed in seeds:
            # One FK call per distinct y, shared by objective constraints.
            saved: dict = {}

            def evaluate(y: Array, saved: dict = saved) -> tuple[Array, Array]:
                key = np.asarray(y).tobytes()
                if saved.get("key") != key:
                    q = self.decode_q(y)
                    fk = self.model.fk(design, q)
                    errors2 = np.sum(((fk[0] - t.positions) / t.length_scale) ** 2, axis=1)
                    saved.update(key=key, errors2=errors2, margins=self.hard_margins(design, q, fk))
                return saved["errors2"], saved["margins"]

            def tracking_constraint(x: Array) -> Array:
                errors2, margins = evaluate(x[:-1])
                # A 1e-10 normalized-length smoothing floor avoids the norm's
                # derivative singularity at zero without a squared objective's
                # loss of resolution close to a high-precision solution.
                return np.r_[x[-1] - np.sqrt(errors2 + 1e-20), margins]

            def optimize_tracking(y: Array) -> OptimizeResult:
                e2, _ = evaluate(y)
                x0 = np.r_[y, np.sqrt(max(e2) + 1e-20) + 1e-8]
                objective_jac = np.r_[np.zeros(len(y)), 1.0]
                return minimize(
                    lambda x: x[-1],
                    x0,
                    jac=lambda x: objective_jac,
                    method="SLSQP",
                    bounds=[(0, 1)] * len(y) + [(0, None)],
                    constraints=[{"type": "ineq", "fun": tracking_constraint}],
                    options={"maxiter": cfg.inner_maxiter, "ftol": cfg.inner_ftol},
                )

            optimized = optimize_tracking(seed)
            q = self.decode_q(optimized.x[:-1])
            first = self._result(design, q, optimized.success, optimized.message)
            candidates.append(first)
            if not first.feasible:
                # Restore constraints with a common slack rho. This returns a
                # graded violation even when no feasible path was found.
                y0 = optimized.x[:-1]
                _, margins = evaluate(y0)
                x0 = np.r_[y0, max(0.0, -min(margins)) + 1e-7]
                restoration_jac = np.r_[np.zeros(len(y0)), 1.0]
                restored = minimize(
                    lambda x: x[-1],
                    x0,
                    jac=lambda x, jac=restoration_jac: jac,
                    method="SLSQP",
                    bounds=[(0, 1)] * len(y0) + [(0, None)],
                    constraints=[{"type": "ineq", "fun": lambda x: evaluate(x[:-1])[1] + x[-1]}],
                    options={"maxiter": cfg.inner_maxiter, "ftol": cfg.inner_ftol},
                )
                candidates.append(
                    self._result(
                        design,
                        self.decode_q(restored.x[:-1]),
                        False,
                        "Feasibility restoration: " + restored.message,
                    )
                )
                if min(evaluate(restored.x[:-1])[1]) >= -cfg.feasibility_tolerance:
                    retried = optimize_tracking(restored.x[:-1])
                    candidates.append(
                        self._result(
                            design, self.decode_q(retried.x[:-1]), retried.success, retried.message
                        )
                    )
        feasible = [r for r in candidates if r.feasible]
        if feasible:
            return min(feasible, key=lambda r: r.max_error)
        return min(candidates, key=lambda r: (-r.min_margin, r.max_error))

    def _result(self, design: Array, q: Array, success: bool, message: str) -> InnerResult:
        fk = self.model.fk(design, q)
        error = float(np.max(np.linalg.norm(fk[0] - self.task.positions, axis=1)))
        margin = float(np.min(self.hard_margins(design, q, fk)))
        return InnerResult(
            np.array(design).copy(),
            q.copy(),
            error,
            margin,
            margin >= -self.settings.feasibility_tolerance,
            bool(success),
            str(message),
        )

    def evaluate_geometry(self, u: Array) -> InnerResult:
        # COBYLA may evaluate outside bounds: clip for safe model evaluation,
        # while explicit outer bounds still reject such proposals.
        u = np.clip(np.asarray(u, float), 0, 1)
        key = tuple(u)  # exact cache; rounding would corrupt outer search steps
        if key not in self.cache:
            if len(self.cache) >= 10:
                raise _GeometryBudgetReached
            result = self.solve_internal(self.lower + self.range * u)
            self.cache[key] = result
            self.history.append(
                {
                    "evaluation": len(self.history) + 1,
                    "design": result.design.tolist(),
                    "max_error_m": result.max_error,
                    "min_margin": result.min_margin,
                    "feasible": result.feasible,
                    "inner_solver_success": result.solver_success,
                }
            )
            if self.verbose:
                print(
                    f"[trajectory] Geometry evaluation {len(self.history)}: "
                    f"max_error={result.max_error:.6g} m, feasible={result.feasible}",
                    file=sys.stderr,
                    flush=True,
                )
        return self.cache[key]

    def solve(
        self, verbose: bool = True, *, subdivisions: int = 10
    ) -> tuple[InnerResult, OptimizeResult]:
        self.verbose = verbose
        threshold = self.task.position_tolerance / 10
        u0 = (self.initial - self.lower) / self.range
        initial = self.evaluate_geometry(u0)
        best = initial
        if verbose:
            print(
                f"[trajectory] Initial maximum error: {initial.max_error:.6g} m",
                file=sys.stderr,
                flush=True,
            )

        def objective(u: Array) -> float:
            r = self.evaluate_geometry(u)
            if r.feasible and r.max_error <= threshold:
                dense = self.validate_dense(r, subdivisions)
                if dense["passes"] and dense["max_error_m"] <= threshold:
                    raise _TrackingThresholdReached(r)
            return r.max_error / self.task.length_scale + 100 * max(0.0, -r.min_margin)

        def constraint(u: Array) -> Array:
            r = self.evaluate_geometry(u)
            return np.r_[u, 1 - u, r.min_margin + self.settings.feasibility_tolerance]

        try:
            outer = minimize(
                objective,
                u0,
                method="COBYLA",
                constraints=[{"type": "ineq", "fun": constraint}],
                options={
                    # COBYLA requires n+2; evaluate_geometry enforces the hard cap.
                    "maxiter": max(len(u0) + 2, min(self.settings.outer_maxeval, 10)),
                    "rhobeg": 0.15,
                    "tol": self.settings.outer_tol,
                    "catol": self.settings.feasibility_tolerance,
                },
            )
            self.evaluate_geometry(outer.x)
        except _TrackingThresholdReached as stop:
            best = stop.args[0]
            outer = OptimizeResult(
                x=(best.design - self.lower) / self.range,
                success=True,
                message=f"Early stop: waypoint and dense errors <= {threshold:g} m (10% of position limit)",
                early_stopped=True,
            )
            if verbose:
                print(f"[trajectory] {outer.message}", file=sys.stderr, flush=True)
        except _GeometryBudgetReached:
            outer = OptimizeResult(success=False, message="Geometry evaluation limit (10) reached")
        if not outer.get("early_stopped", False):
            # Return best actually checked feasible candidate, not blindly outer.x.
            feasible = [r for r in self.cache.values() if r.feasible]
            candidates = feasible or list(self.cache.values())
            best = min(
                candidates,
                key=(lambda r: r.max_error)
                if feasible
                else (lambda r: (-r.min_margin, r.max_error)),
            )
            outer.x = (best.design - self.lower) / self.range
        if verbose:
            print(
                f"[trajectory] Best maximum error: {best.max_error:.6g} m; "
                f"feasible={best.feasible}; geometry evaluations={len(self.cache)}",
                file=sys.stderr,
                flush=True,
            )
        return best, outer

    def dense_path(
        self, result: InnerResult, subdivisions: int = 10
    ) -> tuple[Array, Array, Array | None]:
        """Sample joint motion and the original target on the same dense grid."""
        if subdivisions < 1:
            raise ValueError("subdivisions must be >= 1")
        s = np.arange(self.n)
        dense_s = np.linspace(0, self.n - 1, (self.n - 1) * subdivisions + 1)
        q = np.stack([np.interp(dense_s, s, result.q[:, j]) for j in range(self.dof)], axis=1)
        positions = np.stack(
            [np.interp(dense_s, s, self.task.positions[:, j]) for j in range(3)], axis=1
        )
        t = self.task
        if t.target_positions is not None:
            positions = checked_positions(t.target_positions(dense_s / (self.n - 1)), len(q))
        target_rotations = (
            None
            if t.rotations is None
            else Slerp(s, Rotation.from_matrix(t.rotations))(dense_s).as_matrix()
        )
        return q, positions, target_rotations

    def validate_dense(self, result: InnerResult, subdivisions: int = 10) -> dict[str, Any]:
        """Check interpolated q against the original function or target polyline.

        This finite check is evidence, not a mathematical continuous-path proof.
        Without target_positions, the curve between waypoints is unknown.
        Custom task.extra_margins is evaluated on the dense path as supplied.
        """
        q, positions, target_rotations = self.dense_path(result, subdivisions)
        t = self.task
        # Build a separate validation task so the optimization task is unchanged.
        dense_task = Task(
            positions,
            t.q_start,
            t.q_end,
            t.max_step / subdivisions,
            t.position_tolerance,
            t.length_scale,
            target_rotations,
            t.orientation_tolerance,
            t.obstacles,
            t.link_radius,
            t.clearance,
            t.extra_margins,
        )
        validator = NestedSynthesis(
            self.model, dense_task, self.initial, self.lower, self.upper, self.settings
        )
        fk = self.model.fk(result.design, q)
        error = float(np.max(np.linalg.norm(fk[0] - positions, axis=1)))
        margin = float(np.min(validator.hard_margins(result.design, q, fk)))
        return {
            "samples": len(q),
            "target_interpolation": (
                "original trajectory function, SLERP rotations"
                if t.target_positions
                else "linear positions, SLERP rotations"
            ),
            "joint_interpolation": "linear, in unwrapped coordinates",
            "max_error_m": error,
            "min_margin": margin,
            "constraints_pass": margin >= -self.settings.feasibility_tolerance,
            "position_pass": error <= t.position_tolerance,
            "passes": bool(
                error <= t.position_tolerance and margin >= -self.settings.feasibility_tolerance
            ),
        }


def checked_positions(value: Array, samples: int) -> Array:
    positions = np.asarray(value, float)
    if positions.shape != (samples, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("Trajectory function must return finite positions with shape (N, 3)")
    return positions


def load_target(spec: dict, directory: Path) -> Callable[[Array], Array] | None:
    """Load a local Python curve; its input is the original equation parameter."""
    if "trajectory" not in spec:
        return None
    curve = spec["trajectory"]
    module = (directory / curve["module"]).resolve()
    module.relative_to(directory.resolve())
    start, end = np.asarray(curve["parameter_range"], float)
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Trajectory parameter_range must be finite and increasing")
    samples = curve.setdefault("samples", 21)
    if not isinstance(samples, int) or not 2 <= samples <= 200:
        raise ValueError("Trajectory samples must be an integer between 2 and 200")
    function = runpy.run_path(str(module))[curve.get("function", "positions")]

    def target(s: Array) -> Array:
        return checked_positions(function(start + np.asarray(s) * (end - start)), len(s))

    return target


def task_digest(path: Path) -> str:
    """Identify the task, initial kinematics, and curve source for saved results."""
    digest = hashlib.sha256(path.read_bytes())
    spec = json.loads(path.read_text())
    urdf = (path.parent / spec["urdf"]).resolve()
    digest.update(urdf.read_bytes())
    if "trajectory" in spec:
        module = (path.parent / spec["trajectory"]["module"]).resolve()
        module.relative_to(path.parent.resolve())
        digest.update(module.read_bytes())
    return digest.hexdigest()


def load_problem(path: str | Path) -> tuple[NestedSynthesis, dict]:
    path = Path(path)
    spec = json.loads(path.read_text())
    if "trajectory" in spec and "positions_m" in spec.get("waypoints", {}):
        raise ValueError("Supply either a trajectory function or waypoint positions, not both")
    p = spec["design_parameters"]
    model = SerialURDF(
        path.parent / spec["urdf"], spec["base_link"], spec["tip_link"], p, spec.get("joint_bounds")
    )
    waypoints = spec.get("waypoints", {})
    target = load_target(spec, path.parent)
    positions = (
        checked_positions(
            target(np.linspace(0, 1, spec["trajectory"]["samples"])), spec["trajectory"]["samples"]
        )
        if target is not None
        else np.asarray(waypoints["positions_m"], float)
    )
    bbox_points = target(np.linspace(0, 1, 201)) if target is not None else positions
    diagonal = float(np.linalg.norm(np.ptp(bbox_points, axis=0)))
    tolerance = spec.get(
        "position_tolerance_m", spec.get("position_tolerance_ratio", 0.03) * diagonal
    )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "Position tolerance must be positive and finite. For zero-span paths, set position_tolerance_m."
        )
    rotations = None
    if "orientation_tolerance_rad" in spec and "quaternions_xyzw" not in waypoints:
        raise ValueError(
            "orientation_tolerance_rad requires a target in waypoints.quaternions_xyzw. "
            "For fixed zero roll/pitch/yaw, supply [0, 0, 0, 1]. "
            "Keep the requested orientation constraint when correcting task input."
        )
    if "quaternions_xyzw" in waypoints:
        quaternions = np.asarray(waypoints["quaternions_xyzw"], float)
        if quaternions.shape in ((4,), (1, 4)):
            quaternions = np.broadcast_to(quaternions, (len(positions), 4)).copy()
        elif quaternions.shape != (len(positions), 4):
            raise ValueError(
                "waypoints.quaternions_xyzw: expected "
                f"{len(positions)} quaternion rows (one per position sample), "
                f"got shape {quaternions.shape}. For a fixed orientation, supply "
                "one quaternion [x, y, z, w], for example [0, 0, 0, 1]."
            )
        rotations = Rotation.from_quat(quaternions).as_matrix()
    task = Task(
        positions,
        spec.get("q_start"),
        spec.get("q_end"),
        np.array(spec.get("max_joint_step", model.upper - model.lower), float),
        position_tolerance=tolerance,
        length_scale=spec.get("length_scale_m", 1.0),
        rotations=rotations,
        orientation_tolerance=spec.get("orientation_tolerance_rad", np.deg2rad(5)),
        obstacles=np.array(spec.get("sphere_obstacles", []), float).reshape(-1, 4),
        link_radius=spec.get("capsule_radius_m", 0.0),
        clearance=spec.get("clearance_m", 0.0),
        target_positions=target,
    )
    settings_values = dict(spec.get("settings", {}))
    settings_values.pop("early_stop_error_m", None)  # Ignore obsolete stopping settings.
    settings = Settings(**settings_values)
    solver = NestedSynthesis(
        model,
        task,
        np.asarray([x["initial"] for x in p], float),
        np.asarray([x["lower"] for x in p], float),
        np.asarray([x["upper"] for x in p], float),
        settings,
    )
    return solver, spec


def save_outputs(
    solver: NestedSynthesis,
    best: InnerResult,
    outer: OptimizeResult,
    spec: dict,
    output: str | Path,
    subdivisions: int = 10,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = solver.evaluate_geometry((solver.initial - solver.lower) / solver.range)
    dense = solver.validate_dense(best, subdivisions)
    report = {
        "waypoint_count": solver.n,
        "position_tolerance_m": solver.task.position_tolerance,
        "design": dict(
            zip([p["name"] for p in spec["design_parameters"]], best.design.tolist(), strict=True)
        ),
        "joint_order": solver.model.joint_names,
        "joint_configurations": {
            name: {
                "start": float(best.q[0, i]),
                "end": float(best.q[-1, i]),
                "minimum": float(best.q[:, i].min()),
                "maximum": float(best.q[:, i].max()),
            }
            for i, name in enumerate(solver.model.joint_names)
        },
        "initial_max_error_m": baseline.max_error,
        "waypoint_max_error_m": best.max_error,
        "waypoint_constraints_pass": best.feasible,
        "waypoint_position_pass": best.max_error <= solver.task.position_tolerance,
        "inner_solver_success": best.solver_success,
        "inner_message": best.message,
        "outer_solver_success": bool(outer.success),
        "outer_message": str(outer.message),
        "early_stopped": bool(outer.get("early_stopped", False)),
        "geometry_evaluations": len(solver.cache),
        "dense_validation": dense,
        "functional_success": bool(
            best.feasible and best.max_error <= solver.task.position_tolerance and dense["passes"]
        ),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "history.json").write_text(
        json.dumps(solver.history, indent=2, allow_nan=False) + "\n"
    )
    header = ",".join(["waypoint", *solver.model.joint_names])
    np.savetxt(
        output / "joint_path.csv",
        np.c_[np.arange(solver.n), best.q],
        delimiter=",",
        header=header,
        comments="",
    )
    if isinstance(solver.model, SerialURDF):
        solver.model.write_kinematic_urdf(best.design, output / "optimized_kinematics.urdf")
    return report
