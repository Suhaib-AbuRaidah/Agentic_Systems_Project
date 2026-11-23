#!/usr/bin/env python3
"""Topology-independent planar rigid-body mechanism dimensional synthesis.

AGENT TOOL DESCRIPTION
======================
Name: synthesize_planar_mechanism
Purpose: Given assembled 2-D rigid bodies, local joint frames, revolute / prismatic /
  fixed connections, and sampled point/pose trajectories, optimize adjustable body-local
  attachment coordinates and other declared design geometry. No mechanism names,
  hard-coded four-bar or slider-crank equations, or explicit loop definitions are used.
  Cycles are closed automatically by joint constraint equations.

Scope / non-goals:
  * Planar, rigid, holonomic kinematics only. No dynamics, torque optimization, link
    thickness, collision checking, stress, or automatic mechanism topology synthesis.
  * The topology and trajectory reference points must be provided by the agent.
  * Local nonlinear optimizer (CasADi/IPOPT). No global-optimality guarantee.
  * Joint closure and prescribed drivers are constrained at samples. Target poses
    are fitted; intermediate states are checked using
    numeric continuation, but checks do not prove global collision-free operation.
  * If only a geometric path is supplied, point timing is not inferred globally;
    input samples are ordered and a virtual joint drive is inferred for local
    between-sample continuation validation.
  * Design optimization is over joint-frame x/y (+ selected axis orientations) and
    optionally tool point x/y. CAD solids are *not* modified by this script.

Install: pip install numpy scipy casadi
Usage:
  python planar_linkage_synth.py --schema
  python planar_linkage_synth.py --example fourbar > mechanism.json
  python planar_linkage_synth.py --input mechanism.json --output result.json
  python planar_linkage_synth.py --demo fourbar --output result.json
  python planar_linkage_synth.py --demo slidercrank --output result.json
  python planar_linkage_synth.py --demo fivebar --output result.json
  python planar_linkage_synth.py --self-test

Input JSON outline:
  {
    "units": "mm",                  # metadata only; all lengths must be consistent
    "bodies": [
      {"id":"ground", "ground":true, "pose":[0,0,0],
       "frames":{"pivot":{"pose":[0,0,0]}}, "points":{}},
      {"id":"link", "pose":[0,0,0],
       "frames":{"prox":{"pose":[0,0,0], "free":[]},
                 "dist":{"pose":[50,0,0], "free":["x","y"],
                         "bounds":{"x":[25,80],"y":[-20,20]}}},
       "points":{"tip":{"xy":[25,10], "free":["x","y"]}}}
    ],
    "joints": [
       {"id":"J1","type":"revolute","a":["ground","pivot"],
        "b":["link","prox"], "limits":[-3.14,3.14]}
    ],
    "targets": [
      {"body":"link", "point":"tip", "positions":[[10,20], ...],
       "orientations":[0.1, ...], "position_weight":1, "orientation_weight":1}
    ],
    "drivers": [{"joint":"J1", "values":[0,0.1,...]}],
    "geometry_constraints": [{"type":"distance","body":"link",
       "a":["frame","prox"], "b":["frame","dist"], "min":10,"max":120}],
    "design": {"mode":"auto", "max_shift_fraction":0.4,
                "max_axis_rotation":0.52},
    "options": {"max_iter":700, "regularization":1e-6,
                "smoothness":1e-6, "verify_substeps":2,
                "closure_tolerance":0.001, "driver_tolerance":0.001,
                "trajectory_tolerance":1.0}
  }

`free` flags: ["x","y","theta"], [] (lock), or omitted (automatic choice).
Automatic choice: Ground frames and first joint frame on each moving body are
  locked (sets a local-coordinate gauge). Additional moving-body joint frames can
  vary in x/y; prismatic frame axis theta can vary. Target point local x/y can
  vary. Explicit `free` always overrides defaults, including for ground pivots.
  If auto-free bounds are not supplied, finite bounds are generated around the
  initial value based on initial mechanism size. Explicit bounds are recommended.
  Multiple joint frames on the same body are rigidly related at each sampled
  instant; their local coordinates are design variables shared across all samples.
  No change in topology occurs.

`pose` is [x, y, angle in radians], for a body in world coordinates or a frame
  in its body-local coordinates. `points` values are body-local [x,y] coordinates.
A prismatic joint aligns the A and B frame x axes, and permits sliding along the
  A-frame x direction. Its scalar coordinate is dot(x_axis_A, origin_B-origin_A).
A revolute joint joins frame origins and permits arbitrary relative rotation;
  its scalar coordinate is angle_B - angle_A (in radians, unwrapped).
A fixed joint enforces coincident frames in position and orientation.

If a driver has `values`, they must match the number of trajectory samples and
  are exact equality constraints. Optional drivers without values may specify
  `monotonic`: "increasing" or "decreasing" (constrains sampled joint progress).
Use enough trajectory samples, particularly near singularities / sharp turns.
Angles should be consistent/unwrapped over the motion sequence.

Output JSON:
  status: "success" or "infeasible_or_unverified" or "solver_failed"
  mechanism: optimized body-local geometry AND assembled world poses at first
             trajectory sample; topology intact
  design_changes: list of {path, initial, optimized, lower, upper}
  motion: per-sample body world poses, joint coordinates, and generated target points
  validation: closure / driver / target errors, between-sample continuation checks,
              mobility estimate, singularity diagnostics, warnings
  optimization: IPOPT status, iterations, objective, elapsed time, etc.
  The calling agent MUST check `status == "success"`, as an NLP may return an
  approximate/infeasible point or different assembly branch. Returned points are
  only an optimized *kinematic design*, not a validated physical CAD assembly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

try:
    import casadi as ca
except ImportError:
    ca = None

AXES = {"x": 0, "y": 1, "theta": 2}


class SpecificationError(ValueError):
    """Raised for invalid agent-supplied mechanism specifications."""


def frame_spec(value: Any, is_point: bool = False) -> dict:
    field = "xy" if is_point else "pose"
    if isinstance(value, (list, tuple)):
        return {field: list(value)}
    if not isinstance(value, dict) or field not in value:
        raise SpecificationError(f"Frame / point must contain '{field}': {value}")
    return copy.deepcopy(value)


def as_float_list(value: Any, n: int, description: str) -> list[float]:
    try:
        seq = [float(v) for v in value]
    except (ValueError, TypeError):
        raise SpecificationError(f"{description}: expected {n} numeric entries") from None
    if len(seq) != n or not all(math.isfinite(v) for v in seq):
        raise SpecificationError(f"{description}: expected {n} finite numbers")
    return seq


def rot_numpy(t):
    c, s = math.cos(float(t)), math.sin(float(t))
    return np.array([[c, -s], [s, c]], dtype=float)


def angular_delta(x):
    return math.atan2(math.sin(x), math.cos(x))


def mean_target_scale(spec: dict) -> float:
    vec = []
    for body in spec["bodies"]:
        p = body.get("pose", [0.0, 0.0, 0.0])
        vec.extend(map(abs, p[:2]))
        for fr in body.get("frames", {}).values():
            vec.extend(map(abs, frame_spec(fr)["pose"][:2]))
        for po in body.get("points", {}).values():
            vec.extend(map(abs, frame_spec(po, True)["xy"]))
    for tgt in spec.get("targets", []):
        if "positions" in tgt:
            positions = np.asarray(tgt["positions"], dtype=float)
            if len(positions) > 1:
                vec.append(float(np.linalg.norm(np.ptp(positions, axis=0))))
    return max(1e-6, max(vec, default=0.0))


@dataclass
class Variable:
    kind: str  # 'frame' or 'point'
    body: str
    feature: str
    axis: str
    initial: float
    lower: float
    upper: float

    @property
    def path(self):
        return f"bodies.{self.body}.{self.kind}s.{self.feature}.{self.axis}"


class PlanarModel:
    """A joint graph, NOT a collection of named mechanism-type solvers."""

    def __init__(self, specification: dict):
        if not isinstance(specification, dict):
            raise SpecificationError("Root specification must be a JSON object")
        self.original = copy.deepcopy(specification)
        self.spec = copy.deepcopy(specification)
        self.design_options = self.spec.get("design", {})
        self.options = self.spec.get("options", {})
        self.bodies: dict[str, dict] = {}
        self.joints: dict[str, dict] = {}
        self.variables: list[Variable] = []
        self.target_records = []
        self.driver_records = []
        self.geometry_constraints = []

        for b in self.spec.get("bodies", []):
            name = str(b.get("id", ""))
            if not name or name in self.bodies:
                raise SpecificationError("Every body needs a unique nonempty id")
            b["pose"] = as_float_list(b.get("pose", [0, 0, 0]), 3, f"body {name}.pose")
            b["frames"] = {n: frame_spec(v) for n, v in b.get("frames", {}).items()}
            b["points"] = {n: frame_spec(v, True) for n, v in b.get("points", {}).items()}
            for fname, f in b["frames"].items():
                f["pose"] = as_float_list(f["pose"], 3, f"{name}.{fname}.pose")
            for pname, p in b["points"].items():
                p["xy"] = as_float_list(p["xy"], 2, f"{name}.{pname}.xy")
            self.bodies[name] = b
        ground = [name for name, b in self.bodies.items() if b.get("ground")]
        if len(ground) != 1 or len(self.bodies) < 2:
            raise SpecificationError("Need >=2 bodies and exactly one ground body")
        self.ground = ground[0]
        self.moving = [b for b in self.bodies if b != self.ground]
        for j in self.spec.get("joints", []):
            jid = str(j.get("id", ""))
            if not jid or jid in self.joints:
                raise SpecificationError("Every joint needs a unique id")
            typ = j.get("type")
            if typ not in {"revolute", "prismatic", "fixed"}:
                raise SpecificationError(f"Unsupported joint type: {typ}")
            for side in ("a", "b"):
                ref = j.get(side)
                if not isinstance(ref, (list, tuple)) or len(ref) != 2:
                    raise SpecificationError(f"Joint {jid}.{side}: expected [body,frame]")
                body, fr = ref
                if body not in self.bodies or fr not in self.bodies[body]["frames"]:
                    raise SpecificationError(f"Joint {jid}.{side}: unknown body/frame {ref}")
            if j["a"][0] == j["b"][0]:
                raise SpecificationError(f"Joint {jid}: cannot connect body to itself")
            if "limits" in j:
                lim = as_float_list(j["limits"], 2, f"{jid}.limits")
                if lim[0] > lim[1]:
                    raise SpecificationError(f"{jid}: invalid limits")
            self.joints[jid] = j
        if not self.joints:
            raise SpecificationError("At least one joint is required")
        self._check_graph_connected()

        self.target_records = copy.deepcopy(self.spec.get("targets", []))
        sample_count = None
        for tgt in self.target_records:
            bid = tgt.get("body")
            if bid not in self.bodies:
                raise SpecificationError(f"Target: unknown body {bid}")
            if "positions" not in tgt and "orientations" not in tgt:
                raise SpecificationError("Target needs positions and/or orientations")
            if "frame" in tgt and tgt["frame"] not in self.bodies[bid]["frames"]:
                raise SpecificationError(f"Unknown target frame {bid}.{tgt['frame']}")
            if "positions" in tgt:
                point, fr = tgt.get("point"), tgt.get("frame")
                if (point is None) == (fr is None):
                    raise SpecificationError(
                        "Target positions require exactly one of point or frame"
                    )
                if point is not None and point not in self.bodies[bid]["points"]:
                    raise SpecificationError(f"Unknown target point {bid}.{point}")
                if fr is not None and fr not in self.bodies[bid]["frames"]:
                    raise SpecificationError(f"Unknown target frame {bid}.{fr}")
                tgt["positions"] = [
                    as_float_list(x, 2, "target position") for x in tgt["positions"]
                ]
                length = len(tgt["positions"])
            else:
                length = len(tgt["orientations"])
            if "orientations" in tgt:
                tgt["orientations"] = [float(v) for v in tgt["orientations"]]
                if len(tgt["orientations"]) != length:
                    raise SpecificationError("Target position and orientation sample counts differ")
                if not all(math.isfinite(v) for v in tgt["orientations"]):
                    raise SpecificationError("Non-finite orientation")
            if sample_count is None:
                sample_count = length
            if length != sample_count:
                raise SpecificationError("All target trajectories must have same sample count")
            for weight in ("position_weight", "orientation_weight"):
                if float(tgt.get(weight, 1)) < 0:
                    raise SpecificationError("Trajectory weights cannot be negative")
        if sample_count is None:
            sample_count = next(
                (
                    len(driver["values"])
                    for driver in self.spec.get("drivers", [])
                    if "values" in driver
                ),
                0,
            )
        if sample_count < 2:
            raise SpecificationError("Need at least 2 target samples or prescribed driver values")
        self.K = sample_count
        self.driver_records = copy.deepcopy(self.spec.get("drivers", []))
        driven_ids = set()
        for dr in self.driver_records:
            if dr.get("joint") not in self.joints:
                raise SpecificationError(f"Driver refers to unknown joint {dr.get('joint')}")
            if self.joints[dr["joint"]]["type"] == "fixed":
                raise SpecificationError("A fixed joint cannot be a motion driver")
            if dr["joint"] in driven_ids:
                raise SpecificationError("Only one driver declaration per joint")
            driven_ids.add(dr["joint"])
            if "values" in dr:
                dr["values"] = [float(v) for v in dr["values"]]
                if len(dr["values"]) != self.K or not all(math.isfinite(x) for x in dr["values"]):
                    raise SpecificationError(
                        "Driver values must be one finite value per target sample"
                    )
            elif dr.get("monotonic") not in ("increasing", "decreasing"):
                raise SpecificationError("Driver without values must specify monotonic direction")
        self.geometry_constraints = copy.deepcopy(self.spec.get("geometry_constraints", []))
        for gc in self.geometry_constraints:
            if gc.get("type") != "distance":
                raise SpecificationError("Only geometry constraint type=distance is supported")
            bid = gc.get("body")
            if bid not in self.bodies:
                raise SpecificationError(f"Unknown geometry constraint body: {bid}")
            for side in ("a", "b"):
                ref = gc.get(side)
                if (
                    not isinstance(ref, (list, tuple))
                    or len(ref) != 2
                    or ref[0] not in ("frame", "point")
                ):
                    raise SpecificationError(
                        "Geometry constraint endpoint must be [frame|point,name]"
                    )
                registry = "frames" if ref[0] == "frame" else "points"
                if ref[1] not in self.bodies[bid][registry]:
                    raise SpecificationError(f"Unknown geometry endpoint: {bid}.{ref}")
            if "min" not in gc and "max" not in gc:
                raise SpecificationError("Distance constraint requires min and/or max")
            lo = float(gc.get("min", 0))
            hi = float(gc.get("max", float("inf")))
            if lo < 0 or hi < lo or not math.isfinite(lo):
                raise SpecificationError("Invalid distance bounds")
        self.length_scale = mean_target_scale(self.spec)
        self._make_variables()
        self.moving_index = {bid: i for i, bid in enumerate(self.moving)}
        self.nq = 3 * len(self.moving)

    def _check_graph_connected(self):
        edges = {b: [] for b in self.bodies}
        for j in self.joints.values():
            a, b = j["a"][0], j["b"][0]
            edges[a].append(b)
            edges[b].append(a)
        seen, Q = {self.ground}, deque([self.ground])
        while Q:
            for neighbor in edges[Q.popleft()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    Q.append(neighbor)
        if len(seen) != len(self.bodies):
            raise SpecificationError("Joint graph must connect all bodies to ground")

    def _make_variables(self):
        mode = self.design_options.get("mode", "auto")
        if mode not in ("auto", "explicit"):
            raise SpecificationError("design.mode must be auto or explicit")
        fraction = float(self.design_options.get("max_shift_fraction", 0.4))
        axis_angle = float(self.design_options.get("max_axis_rotation", 0.52))
        if fraction <= 0 or axis_angle <= 0:
            raise SpecificationError("Auto design default bounds must be positive")
        attached = {bid: [] for bid in self.bodies}
        prism_frames = set()
        for j in self.joints.values():
            for side in ("a", "b"):
                bid, fr = j[side]
                if fr not in attached[bid]:
                    attached[bid].append(fr)
                if j["type"] == "prismatic":
                    prism_frames.add((bid, fr))
        targeted_points = {
            (t["body"], t.get("point")) for t in self.target_records if "positions" in t
        }
        for bid, body in self.bodies.items():
            for kind in ("frame", "point"):
                feature_dict = body["frames"] if kind == "frame" else body["points"]
                key = "pose" if kind == "frame" else "xy"
                axes = ("x", "y", "theta") if kind == "frame" else ("x", "y")
                for feature, vals in feature_dict.items():
                    if "free" in vals:
                        free = set(vals["free"])
                        if not free.issubset(axes):
                            raise SpecificationError(
                                f"Unknown free coordinates for {bid}.{feature}"
                            )
                    elif mode == "explicit" or bid == self.ground:
                        free = set()
                    elif kind == "frame":
                        # Gauge choice: lock the first body-local joint frame unless explicitly free.
                        first = attached[bid][0] if attached[bid] else None
                        free = (
                            set()
                            if feature == first or feature not in attached[bid]
                            else {"x", "y"}
                        )
                        if (bid, feature) in prism_frames and feature != first:
                            free.add("theta")
                    else:
                        free = {"x", "y"} if (bid, feature) in targeted_points else set()
                    for axis in axes:
                        if axis not in free:
                            continue
                        p = float(vals[key][AXES[axis]])
                        bound = vals.get("bounds", {}).get(axis)
                        if bound is None:
                            delta = axis_angle if axis == "theta" else self.length_scale * fraction
                            lo, hi = p - delta, p + delta
                        else:
                            lo, hi = as_float_list(bound, 2, f"{bid}.{feature}.{axis}.bounds")
                        if not lo < hi or not lo <= p <= hi:
                            raise SpecificationError(f"Invalid bounds for {bid}.{feature}.{axis}")
                        self.variables.append(Variable(kind, bid, feature, axis, p, lo, hi))

    def updated_spec(self, values):
        out = copy.deepcopy(self.spec)
        byid = {b["id"]: b for b in out["bodies"]}
        for var, v in zip(self.variables, values, strict=False):
            container = byid[var.body]["frames" if var.kind == "frame" else "points"]
            container[var.feature]["pose" if var.kind == "frame" else "xy"][AXES[var.axis]] = float(
                v
            )
        return out

    def design_mapping(self, values):
        d = {
            (v.kind, v.body, v.feature, v.axis): float(x)
            for v, x in zip(self.variables, values, strict=False)
        }
        return d

    def position_and_angle(self, body, feature, kind, q_body, design_values, symbolic=False):
        b = self.bodies[body]
        payload = b["frames"][feature] if kind == "frame" else b["points"][feature]
        original = payload["pose"] if kind == "frame" else payload["xy"]
        dims = [
            design_values.get((kind, body, feature, axis), original[AXES[axis]])
            for axis in ("x", "y", "theta")[: len(original)]
        ]
        theta = q_body[2]
        if symbolic:
            c, s = ca.cos(theta), ca.sin(theta)
            p = ca.vertcat(
                q_body[0] + c * dims[0] - s * dims[1], q_body[1] + s * dims[0] + c * dims[1]
            )
        else:
            p = np.asarray(q_body[:2], dtype=float) + rot_numpy(theta) @ np.asarray(
                dims[:2], dtype=float
            )
        return p, theta + (dims[2] if kind == "frame" else 0)

    def q_for(self, Q, k, bid, symbolic=False):
        if bid == self.ground:
            return (
                ca.DM(self.bodies[bid]["pose"])
                if symbolic
                else np.asarray(self.bodies[bid]["pose"], dtype=float)
            )
        ix = 3 * self.moving_index[bid]
        return Q[k][ix : ix + 3]

    def endpoints(self, joint, Q, k, design, symbolic=False):
        a, b = joint["a"], joint["b"]
        pa, aa = self.position_and_angle(
            a[0], a[1], "frame", self.q_for(Q, k, a[0], symbolic), design, symbolic
        )
        pb, ab = self.position_and_angle(
            b[0], b[1], "frame", self.q_for(Q, k, b[0], symbolic), design, symbolic
        )
        return pa, aa, pb, ab

    def joint_coordinate(self, joint, Q, k, design, symbolic=False):
        pa, aa, pb, ab = self.endpoints(joint, Q, k, design, symbolic)
        if joint["type"] == "revolute":
            return ab - aa
        if joint["type"] == "prismatic":
            return (
                ca.dot(ca.vertcat(ca.cos(aa), ca.sin(aa)), pb - pa)
                if symbolic
                else float(np.dot(np.array([math.cos(aa), math.sin(aa)]), pb - pa))
            )
        raise SpecificationError("Fixed joint has no motion coordinate")

    def numeric_constraints(
        self, q, dvalues, drivers_values=None, with_targets=False, k=0, include_drivers=True
    ):
        des = self.design_mapping(dvalues)
        Q = [np.asarray(q, dtype=float)]
        constraints = []
        for joint in self.joints.values():
            pa, aa, pb, ab = self.endpoints(joint, Q, 0, des)
            if joint["type"] in ("revolute", "fixed"):
                constraints.extend((pa - pb) / self.length_scale)
            if joint["type"] == "fixed":
                constraints.append(angular_delta(aa - ab))
            if joint["type"] == "prismatic":
                normal = np.array([-math.sin(aa), math.cos(aa)])
                constraints.extend(
                    [float(normal @ (pb - pa)) / self.length_scale, angular_delta(aa - ab)]
                )
        for idx, driver in enumerate(
            [dr for dr in self.driver_records if "values" in dr] if include_drivers else []
        ):
            if "values" in driver:
                val = driver["values"][k] if drivers_values is None else drivers_values[idx]
                actual = self.joint_coordinate(self.joints[driver["joint"]], Q, 0, des)
                if self.joints[driver["joint"]]["type"] == "revolute":
                    constraints.append(angular_delta(actual - val))
                else:
                    constraints.append((actual - val) / self.length_scale)
        if with_targets:
            for target in self.target_records:
                if "positions" in target:
                    feature = target.get("point", target.get("frame"))
                    kind = "point" if "point" in target else "frame"
                    p, _ = self.position_and_angle(
                        target["body"], feature, kind, self.q_for(Q, 0, target["body"]), des
                    )
                    constraints.extend(
                        0.2 * (p - np.asarray(target["positions"][k])) / self.length_scale
                    )
                if "orientations" in target:
                    theta = self.q_for(Q, 0, target["body"])[2]
                    if "frame" in target:
                        _, theta = self.position_and_angle(
                            target["body"],
                            target["frame"],
                            "frame",
                            self.q_for(Q, 0, target["body"]),
                            des,
                        )
                    constraints.append(0.2 * angular_delta(theta - target["orientations"][k]))
        return np.asarray(constraints, dtype=float)

    def initial_q(self):
        """Warm-start configurations using continuation on the initial joint graph."""
        initial = np.concatenate([self.bodies[b]["pose"] for b in self.moving]).astype(float)
        design = np.asarray([v.initial for v in self.variables], dtype=float)
        current = initial.copy()
        out = []
        # A driver without values is not a prescribed coordinate, so use targets as a weak IK guide.
        driven = any("values" in d for d in self.driver_records)
        for k in range(self.K):
            q0 = current.copy()
            if k > 0 and driven:
                last_dr = [dr["values"][k - 1] for dr in self.driver_records if "values" in dr]
                next_dr = [dr["values"][k] for dr in self.driver_records if "values" in dr]
                steps = 1
                for a, b, dr in zip(
                    last_dr,
                    next_dr,
                    [dr for dr in self.driver_records if "values" in dr],
                    strict=False,
                ):
                    scale = (
                        0.28
                        if self.joints[dr["joint"]]["type"] == "revolute"
                        else self.length_scale * 0.15
                    )
                    steps = max(steps, min(20, math.ceil(abs(b - a) / scale)))
                for step in range(1, steps + 1):
                    frac = step / steps
                    vals = [a + (b - a) * frac for a, b in zip(last_dr, next_dr, strict=False)]

                    def f(q, vals=vals, q0=q0):
                        r = self.numeric_constraints(q, design, drivers_values=vals)
                        return np.r_[
                            r,
                            1e-4
                            * (q - q0)
                            / np.tile(
                                [self.length_scale, self.length_scale, 1.0], len(self.moving)
                            ),
                        ]

                    result = least_squares(
                        f, current, max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9
                    )
                    current = result.x

            def f(q, k=k, current=current):
                r = self.numeric_constraints(q, design, k=k, with_targets=True)
                return np.r_[
                    r,
                    1e-5
                    * (q - current)
                    / np.tile([self.length_scale, self.length_scale, 1.0], len(self.moving)),
                ]

            current = least_squares(f, current, max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9).x
            # Equivalent angles differing by 2π describe the same body pose;
            # unwrap consistently so the NLP does not inherit spurious revolutions.
            reference = out[-1] if out else initial
            for idx in range(2, self.nq, 3):
                current[idx] = reference[idx] + angular_delta(current[idx] - reference[idx])
            out.append(current.copy())
        return np.asarray(out)

    def build_nlp(self, initial_Q, override_options=None):
        if ca is None:
            raise RuntimeError("CasADi is required: pip install casadi")
        opts = {**self.options, **(override_options or {})}
        opti = ca.Opti()
        K = self.K
        L = self.length_scale
        d = opti.variable(len(self.variables)) if self.variables else ca.DM.zeros(0, 1)
        design = {(v.kind, v.body, v.feature, v.axis): d[i] for i, v in enumerate(self.variables)}
        Q = [opti.variable(self.nq) for _ in range(K)]
        d0 = np.asarray([v.initial for v in self.variables], dtype=float)
        if self.variables:
            for i, v in enumerate(self.variables):
                opti.subject_to(opti.bounded(v.lower, d[i], v.upper))
            opti.set_initial(d, d0)
        for k in range(K):
            opti.set_initial(Q[k], initial_Q[k].tolist())
        # Do not allow independent samples to jump between equivalent 2π
        # rotations or disconnected configurations. Agent can adjust limits if
        # the path is sparsely sampled or contains unusually large steps.
        angle_step = float(opts.get("max_step_angle_radians", 1.2))
        position_step = float(opts.get("max_step_translation_fraction", 1.0)) * L
        for k in range(1, K):
            for i in range(len(self.moving)):
                j = 3 * i
                opti.subject_to(
                    opti.bounded(-angle_step, Q[k][j + 2] - Q[k - 1][j + 2], angle_step)
                )
                for ax in (0, 1):
                    opti.subject_to(
                        opti.bounded(-position_step, Q[k][j + ax] - Q[k - 1][j + ax], position_step)
                    )
        loss = 0
        # Shared body-local geometry limits, independent of time or topology.
        for gc in self.geometry_constraints:
            bid = gc["body"]
            a = gc["a"]
            b = gc["b"]
            va = self.bodies[bid]["frames" if a[0] == "frame" else "points"][a[1]]
            vb = self.bodies[bid]["frames" if b[0] == "frame" else "points"][b[1]]

            def geom_xy(ref, feature, bid=bid):
                key = "pose" if ref[0] == "frame" else "xy"
                return ca.vertcat(
                    *[
                        design.get((ref[0], bid, ref[1], axis), feature[key][i])
                        for i, axis in enumerate(("x", "y"))
                    ]
                )

            delta = (geom_xy(a, va) - geom_xy(b, vb)) / L
            dist2 = ca.dot(delta, delta)
            if "min" in gc:
                opti.subject_to(dist2 >= (float(gc["min"]) / L) ** 2)
            if "max" in gc:
                opti.subject_to(dist2 <= (float(gc["max"]) / L) ** 2)
        for k in range(K):
            for j in self.joints.values():
                pa, aa, pb, ab = self.endpoints(j, Q, k, design, symbolic=True)
                if j["type"] in ("revolute", "fixed"):
                    opti.subject_to((pa - pb) / L == 0)
                if j["type"] == "fixed":
                    opti.subject_to(aa - ab == 0)
                if j["type"] == "prismatic":
                    normal = ca.vertcat(-ca.sin(aa), ca.cos(aa))
                    opti.subject_to(ca.dot(normal, pb - pa) / L == 0)
                    opti.subject_to(aa - ab == 0)
                if "limits" in j and j["type"] != "fixed":
                    coord = self.joint_coordinate(j, Q, k, design, symbolic=True)
                    cscale = L if j["type"] == "prismatic" else 1.0
                    opti.subject_to(
                        opti.bounded(
                            j["limits"][0] / cscale, coord / cscale, j["limits"][1] / cscale
                        )
                    )
            for dr in self.driver_records:
                j = self.joints[dr["joint"]]
                coord = self.joint_coordinate(j, Q, k, design, symbolic=True)
                cscale = L if j["type"] == "prismatic" else 1.0
                if "values" in dr:
                    opti.subject_to(coord / cscale == dr["values"][k] / cscale)
                elif k > 0:
                    prev = self.joint_coordinate(j, Q, k - 1, design, symbolic=True)
                    if dr["monotonic"] == "increasing":
                        opti.subject_to((coord - prev) / cscale >= -1e-9)
                    else:
                        opti.subject_to((coord - prev) / cscale <= 1e-9)
            for tgt in self.target_records:
                bid = tgt["body"]
                if "positions" in tgt:
                    kind = "point" if "point" in tgt else "frame"
                    feature = tgt.get("point", tgt.get("frame"))
                    p, _ = self.position_and_angle(
                        bid, feature, kind, self.q_for(Q, k, bid, True), design, True
                    )
                    err = (p - ca.DM(tgt["positions"][k])) / L
                    loss += float(tgt.get("position_weight", 1)) * ca.dot(err, err) / K
                if "orientations" in tgt:
                    theta = self.q_for(Q, k, bid, True)[2]
                    if "frame" in tgt:
                        _, theta = self.position_and_angle(
                            bid, tgt["frame"], "frame", self.q_for(Q, k, bid, True), design, True
                        )
                    err = theta - float(tgt["orientations"][k])
                    loss += float(tgt.get("orientation_weight", 1)) * err**2 / K
        # Penalize excessive deviation from starting *geometry*, not sample motion.
        regularization = float(opts.get("regularization", 1e-6))
        for i, v in enumerate(self.variables):
            scale = max((v.upper - v.lower) * 0.5, 1e-8)
            loss += regularization * ((d[i] - v.initial) / scale) ** 2 / max(1, len(self.variables))
        # A small second-difference regularizer improves continuation for path-only tasks.
        smooth = float(opts.get("smoothness", 1e-6))
        if smooth > 0 and K >= 3:
            s = ca.DM(np.tile([L, L, 1.0], len(self.moving)))
            for k in range(1, K - 1):
                dd = (Q[k + 1] - 2 * Q[k] + Q[k - 1]) / s
                loss += smooth * ca.dot(dd, dd) / (K - 2)
        opti.minimize(loss)
        ipopt_opts = {
            "print_time": False,
            "ipopt": {
                "print_level": int(opts.get("print_level", 0)),
                "max_iter": int(opts.get("max_iter", 700)),
                "tol": float(opts.get("nlp_tolerance", 1e-8)),
                "acceptable_tol": float(opts.get("acceptable_tol", 1e-6)),
                "sb": "yes",
                "mu_strategy": "adaptive",
            },
        }
        opti.solver("ipopt", ipopt_opts)
        return opti, d, Q

    def constraint_metrics(self, q, dvalues, driving=None):
        """Unscaled physical constraint errors: lengths and angles kept separate."""
        design = self.design_mapping(dvalues)
        Q = [np.asarray(q, dtype=float)]
        lin = []
        ang = []
        drive_lin = []
        drive_ang = []
        limit_lin = []
        limit_ang = []
        for j in self.joints.values():
            pa, aa, pb, ab = self.endpoints(j, Q, 0, design)
            if j["type"] in ("revolute", "fixed"):
                lin.append(float(np.linalg.norm(pa - pb)))
            if j["type"] == "prismatic":
                lin.append(abs(float(np.dot([-math.sin(aa), math.cos(aa)], pb - pa))))
            if j["type"] in ("fixed", "prismatic"):
                ang.append(abs(angular_delta(aa - ab)))
            if "limits" in j and j["type"] != "fixed":
                coordinate = float(self.joint_coordinate(j, Q, 0, design))
                lo, hi = j["limits"]
                violated = max(0.0, lo - coordinate, coordinate - hi)
                (limit_ang if j["type"] == "revolute" else limit_lin).append(violated)
        if driving:
            for jid, desired in driving.items():
                j = self.joints[jid]
                actual = float(self.joint_coordinate(j, Q, 0, design))
                if j["type"] == "revolute":
                    drive_ang.append(abs(angular_delta(actual - desired)))
                else:
                    drive_lin.append(abs(actual - desired))
        return {
            "closure_length": max(lin, default=0.0),
            "closure_angle": max(ang, default=0.0),
            "driver_length": max(drive_lin, default=0.0),
            "driver_angle": max(drive_ang, default=0.0),
            "joint_limit_length": max(limit_lin, default=0.0),
            "joint_limit_angle": max(limit_ang, default=0.0),
        }

    def _analyze(self, opt_spec, dvalues, Qvalues, solver_info=None):
        design = self.design_mapping(dvalues)
        Q = [np.asarray(x, dtype=float) for x in Qvalues]
        target_position = []
        target_angle = []
        generated_targets = []
        joint_states = []
        measures = []
        kinematic_mobility = []
        jacobian_conditions = []
        for k in range(self.K):
            per_joint = {}
            for jid, j in self.joints.items():
                if j["type"] != "fixed":
                    per_joint[jid] = float(self.joint_coordinate(j, Q, k, design))
                else:
                    per_joint[jid] = 0.0
            joint_states.append(per_joint)
            requested = {
                dr["joint"]: dr["values"][k] for dr in self.driver_records if "values" in dr
            }
            measures.append(self.constraint_metrics(Q[k], dvalues, requested))
            # Numerical rank of the *undriven* constraint Jacobian; catches
            # special-geometry/redundancy missed by the generic Grübler count.
            q = Q[k]
            r0 = self.numeric_constraints(q, dvalues, include_drivers=False)
            jac = np.zeros((len(r0), len(q)))
            for j in range(len(q)):
                step = 1e-6 * max(1.0, abs(q[j]))
                x = q.copy()
                x[j] += step
                y = q.copy()
                y[j] -= step
                jac[:, j] = (
                    self.numeric_constraints(x, dvalues, include_drivers=False)
                    - self.numeric_constraints(y, dvalues, include_drivers=False)
                ) / (2 * step)
            sv = np.linalg.svd(jac, compute_uv=False)
            eps = max(sv[0] * 1e-7, 1e-10) if len(sv) else 1e-10
            rank = int(np.sum(sv > eps))
            kinematic_mobility.append(self.nq - rank)
            positive = sv[sv > eps]
            jacobian_conditions.append(float(positive[0] / positive[-1]) if len(positive) else 1.0)
            for tgt in self.target_records:
                result = {"body": tgt["body"]}
                if "positions" in tgt:
                    kind = "point" if "point" in tgt else "frame"
                    feature = tgt.get("point", tgt.get("frame"))
                    p, _ = self.position_and_angle(
                        tgt["body"], feature, kind, self.q_for(Q, k, tgt["body"]), design
                    )
                    result.update({kind: feature, "position": list(map(float, p))})
                    target_position.append(
                        float(np.linalg.norm(p - np.asarray(tgt["positions"][k])))
                    )
                if "orientations" in tgt:
                    theta = float(self.q_for(Q, k, tgt["body"])[2])
                    if "frame" in tgt:
                        _, theta = self.position_and_angle(
                            tgt["body"],
                            tgt["frame"],
                            "frame",
                            self.q_for(Q, k, tgt["body"]),
                            design,
                        )
                        theta = float(theta)
                    result["orientation"] = theta
                    target_angle.append(abs(angular_delta(theta - tgt["orientations"][k])))
                generated_targets.append({"sample": k, **result})

        def rms(xs: list[float]) -> float:
            return float(math.sqrt(np.mean(np.square(xs)))) if xs else 0.0

        def maxv(key: str) -> float:
            return max((m[key] for m in measures), default=0.0)

        mobility = 3 * len(self.moving) - sum(
            (3 if j["type"] == "fixed" else 2) for j in self.joints.values()
        )
        # Validate actual optimized body-local distances, not merely IPOPT status.
        geom_violation = 0.0
        for gc in self.geometry_constraints:
            points = []
            for ref in (gc["a"], gc["b"]):
                kind, name = ref
                feature = self.bodies[gc["body"]]["frames" if kind == "frame" else "points"][name]
                key = "pose" if kind == "frame" else "xy"
                p = np.array(
                    [
                        design.get((kind, gc["body"], name, axis), feature[key][i])
                        for i, axis in enumerate(("x", "y"))
                    ],
                    dtype=float,
                )
                points.append(p)
            dist = float(np.linalg.norm(points[0] - points[1]))
            geom_violation = max(
                geom_violation,
                float(gc.get("min", 0)) - dist,
                dist - float(gc.get("max", float("inf"))),
            )
        validation = {
            "geometry_constraint_max_violation_length_units": max(0.0, geom_violation),
            "closure_max_length_units": maxv("closure_length"),
            "closure_max_angle_radians": maxv("closure_angle"),
            "driver_max_length_units": maxv("driver_length"),
            "driver_max_angle_radians": maxv("driver_angle"),
            "joint_limit_max_translation_violation_length_units": maxv("joint_limit_length"),
            "joint_limit_max_angle_violation_radians": maxv("joint_limit_angle"),
            "target_position_rmse_length_units": rms(target_position),
            "target_position_max_length_units": max(target_position, default=0.0),
            "target_orientation_rmse_radians": rms(target_angle),
            "target_orientation_max_radians": max(target_angle, default=0.0),
            "generic_mobility_count": mobility,
            "sampled_mobility_rank_min": int(min(kinematic_mobility)),
            "sampled_mobility_rank_max": int(max(kinematic_mobility)),
            "jacobian_condition_number_max": float(max(jacobian_conditions)),
            "near_singular_samples": [
                i
                for i, (n, cond) in enumerate(
                    zip(kinematic_mobility, jacobian_conditions, strict=False)
                )
                if n > min(kinematic_mobility) or cond > 1e8
            ],
            "continuation_max_closure_length_units": None,
            "continuation_max_pose_jump_normalized": None,
            "continuation_pass": None,
            "warnings": [],
        }
        if mobility != 1:
            validation["warnings"].append(
                f"Generic planar Grübler count is {mobility} (not 1). Multiple driven degrees of freedom may be intentional; numeric rank is also reported."
            )
        if validation["near_singular_samples"]:
            validation["warnings"].append(
                "Potential singularity in sampled configurations; inspect Jacobian diagnostics"
            )
        return validation, generated_targets, joint_states

    def verify_continuation(
        self, dvalues, Qvalues, verification_substeps, samples: list[dict] | None = None
    ):
        """Check closure and continuity at interpolated driver values or inferred DOF."""
        defaults = {
            "continuation_pass": None,
            "continuation_max_closure_length_units": None,
            "continuation_max_closure_angle_radians": None,
            "continuation_max_pose_jump_normalized": None,
        }
        if verification_substeps <= 0:
            return {**defaults, "continuation_note": "Verification explicitly disabled"}
        design = self.design_mapping(dvalues)
        prescribed = [dr for dr in self.driver_records if "values" in dr]
        inferred = False
        if not prescribed:
            # For path-only input, infer a motion parameter from observed joint
            # changes. The chosen parameter is just a continuation coordinate,
            # never an additional mechanism-design assumption.
            candidates = [dr["joint"] for dr in self.driver_records if "values" not in dr]
            if not candidates:
                candidates = [jid for jid, j in self.joints.items() if j["type"] != "fixed"]
            if not candidates:
                return {**defaults, "continuation_note": "No movable joint available"}
            sequences = {
                jid: [
                    float(
                        self.joint_coordinate(
                            self.joints[jid], [np.asarray(q, dtype=float)], 0, design
                        )
                    )
                    for q in Qvalues
                ]
                for jid in candidates
            }
            ranked = sorted(
                candidates, key=lambda jid: sum(abs(np.diff(sequences[jid]))), reverse=True
            )
            selected = ranked[0]
            if sum(abs(np.diff(sequences[selected]))) < 1e-7:
                return {
                    **defaults,
                    "continuation_note": "No measurable motion from which to infer a driver",
                }
            prescribed = [{"joint": selected, "values": sequences[selected]}]
            inferred = True
        L = self.length_scale
        scaling = np.tile([L, L, 1.0], len(self.moving))
        dvalues = np.asarray(dvalues, dtype=float)
        max_lin = 0.0
        max_ang = 0.0
        max_drive_lin = 0.0
        max_drive_ang = 0.0
        max_jump = 0.0
        max_limit_lin = 0.0
        max_limit_ang = 0.0
        failures = []
        for k in range(self.K - 1):
            start = np.asarray(Qvalues[k], dtype=float)
            end = np.asarray(Qvalues[k + 1], dtype=float)
            previous = start.copy()
            for h in range(1, verification_substeps + 2):
                frac = h / (verification_substeps + 1)
                vals = [
                    dr["values"][k] * (1 - frac) + dr["values"][k + 1] * frac for dr in prescribed
                ]
                targets = {dr["joint"]: v for dr, v in zip(prescribed, vals, strict=False)}
                guess = previous * (1 - frac) + end * frac

                def f(x, vals=vals, guess=guess):
                    # Only joint graph residuals plus the selected interpolated drive(s).
                    r = self.numeric_constraints(x, dvalues, include_drivers=False)
                    des = self.design_mapping(dvalues)
                    qlist = [np.asarray(x, dtype=float)]
                    extras = []
                    for dr, val in zip(prescribed, vals, strict=False):
                        j = self.joints[dr["joint"]]
                        actual = self.joint_coordinate(j, qlist, 0, des)
                        extras.append(
                            angular_delta(actual - val)
                            if j["type"] == "revolute"
                            else (actual - val) / L
                        )
                    return np.r_[r, extras, 1e-7 * (x - guess) / scaling]

                sol = least_squares(f, guess, max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                met = self.constraint_metrics(sol.x, dvalues, targets)
                max_lin = max(max_lin, met["closure_length"])
                max_ang = max(max_ang, met["closure_angle"])
                max_drive_lin = max(max_drive_lin, met["driver_length"])
                max_drive_ang = max(max_drive_ang, met["driver_angle"])
                max_limit_lin = max(max_limit_lin, met["joint_limit_length"])
                max_limit_ang = max(max_limit_ang, met["joint_limit_angle"])
                jump = float(np.max(np.abs((sol.x - previous) / scaling)))
                max_jump = max(max_jump, jump)
                if met["closure_length"] > float(self.options.get("closure_tolerance", 0.001)):
                    failures.append(
                        {"interval": k, "substep": h, "closure_length": met["closure_length"]}
                    )
                if h <= verification_substeps and samples is not None:
                    samples.append(
                        {
                            "interval": k,
                            "fraction": frac,
                            "bodies": self.body_poses(sol.x),
                            "joint_coordinates": {
                                name: float(self.joint_coordinate(joint, [sol.x], 0, design))
                                for name, joint in self.joints.items()
                                if joint["type"] != "fixed"
                            },
                        }
                    )
                previous = sol.x.copy()
        length_tol = float(self.options.get("closure_tolerance", 0.001))
        angle_tol = float(self.options.get("angular_closure_tolerance", 0.001))
        drive_length_tol = float(self.options.get("driver_tolerance", 0.001))
        drive_angle_tol = float(self.options.get("angular_driver_tolerance", 0.001))
        passed = (
            max_lin <= length_tol
            and max_ang <= angle_tol
            and max_drive_lin <= drive_length_tol
            and max_drive_ang <= drive_angle_tol
            and max_limit_lin <= drive_length_tol
            and max_limit_ang <= drive_angle_tol
            and max_jump <= float(self.options.get("continuation_jump_threshold", 0.85))
        )
        return {
            "continuation_pass": bool(passed),
            "continuation_max_closure_length_units": max_lin,
            "continuation_max_closure_angle_radians": max_ang,
            "continuation_max_driver_length_units": max_drive_lin,
            "continuation_max_driver_angle_radians": max_drive_ang,
            "continuation_max_joint_limit_translation_violation_length_units": max_limit_lin,
            "continuation_max_joint_limit_angle_violation_radians": max_limit_ang,
            "continuation_max_pose_jump_normalized": max_jump,
            "continuation_driver_source": "inferred_from_solved_motion"
            if inferred
            else "prescribed",
            "continuation_driver_joint_ids": [dr["joint"] for dr in prescribed],
            "continuation_note": "Interpolated joint-coordinate continuation; local numeric check, not collision/formal branch proof",
            "continuation_failures": failures[:12],
        }

    def body_poses(self, configuration: np.ndarray) -> dict[str, list[float]]:
        return {body: self.q_for([configuration], 0, body).tolist() for body in self.bodies}

    def solve(self, options_override=None):
        if ca is None:
            raise RuntimeError("Install dependencies: pip install numpy scipy casadi")
        t0 = time.perf_counter()
        initial_Q = self.initial_q()
        opti, d, Q = self.build_nlp(initial_Q, options_override)
        successful = False
        error = None
        objective = None
        status = "unknown"
        iterations = 0
        try:
            sol = opti.solve()
            successful = True
        except RuntimeError as ex:
            error = str(ex).splitlines()[0][:400]
            sol = opti.debug
        try:
            values = (
                np.asarray(sol.value(d), dtype=float).reshape(-1)
                if self.variables
                else np.asarray([])
            )
            poses = [np.asarray(sol.value(v), dtype=float).reshape(-1) for v in Q]
            objective = float(sol.value(opti.f))
        except Exception as ex:
            return {
                "status": "solver_failed",
                "error": error or str(ex),
                "optimization": {"elapsed_seconds": time.perf_counter() - t0},
            }
        try:
            stats = opti.stats()
            status = stats.get("return_status", "unknown")
            iterations = int(stats.get("iter_count", 0))
        except Exception:
            pass
        output_mechanism = self.updated_spec(values)
        # The input `pose` values are merely a warm start. Return an assembled
        # geometry: first-sample body poses must actually close all joints.
        for body in output_mechanism["bodies"]:
            bid = body["id"]
            if bid != self.ground:
                i = 3 * self.moving_index[bid]
                body["pose"] = [float(v) for v in poses[0][i : i + 3]]
        # Length summary facilitates downstream CAD generation without
        # imposing a topology-specific concept of 'crank' or 'connecting rod'.
        dimensions = []
        for b in output_mechanism["bodies"]:
            features = []
            for fid, fr in b.get("frames", {}).items():
                features.append(("frame", fid, np.array(fr["pose"][:2], dtype=float)))
            for pid, pt in b.get("points", {}).items():
                features.append(("point", pid, np.array(pt["xy"][:2], dtype=float)))
            for (ka, na, pa), (kb, nb, pb) in combinations(features, 2):
                dimensions.append(
                    {
                        "body": b["id"],
                        "a": [ka, na],
                        "b": [kb, nb],
                        "distance": float(np.linalg.norm(pa - pb)),
                    }
                )
        validation, trajectory, states = self._analyze(output_mechanism, values, poses)
        intermediate = []
        validation.update(
            self.verify_continuation(
                values, poses, int(self.options.get("verify_substeps", 2)), intermediate
            )
        )
        closure_ok = validation["closure_max_length_units"] <= float(
            self.options.get("closure_tolerance", 0.001)
        ) and validation["closure_max_angle_radians"] <= float(
            self.options.get("angular_closure_tolerance", 0.001)
        )
        geometry_ok = validation["geometry_constraint_max_violation_length_units"] <= float(
            self.options.get("closure_tolerance", 0.001)
        )
        driver_ok = (
            validation["driver_max_length_units"]
            <= float(self.options.get("driver_tolerance", 0.001))
            and validation["driver_max_angle_radians"]
            <= float(self.options.get("angular_driver_tolerance", 0.001))
            and validation["joint_limit_max_translation_violation_length_units"]
            <= float(self.options.get("driver_tolerance", 0.001))
            and validation["joint_limit_max_angle_violation_radians"]
            <= float(self.options.get("angular_driver_tolerance", 0.001))
        )
        target_ok = validation["target_position_max_length_units"] <= float(
            self.options.get("trajectory_tolerance", 1.0)
        ) and validation["target_orientation_max_radians"] <= float(
            self.options.get("orientation_tolerance", 0.05)
        )
        contin_ok = validation["continuation_pass"] is True
        verdict = (
            "success"
            if successful and closure_ok and driver_ok and geometry_ok and target_ok and contin_ok
            else "infeasible_or_unverified"
        )
        if not successful:
            validation["warnings"].append(f"IPOPT did not report success: {error}")
        if not geometry_ok:
            validation["warnings"].append("At least one geometric design constraint was violated")
        if not target_ok:
            validation["warnings"].append("Specified path/pose tolerances not achieved")
        if not contin_ok:
            validation["warnings"].append("Between-sample continuation check failed")
        if validation["continuation_pass"] is None:
            validation["warnings"].append(
                "Continuity not verified: no usable driver or check disabled"
            )
        if validation.get("continuation_driver_source") == "inferred_from_solved_motion":
            validation["warnings"].append(
                "Path-only verification inferred an input coordinate; no prescribed timing or speed was certified"
            )
        if not self.variables:
            validation["warnings"].append(
                "No free design variables; only mechanism configurations were fitted"
            )
        changes = [
            {
                "path": var.path,
                "initial": var.initial,
                "optimized": float(v),
                "lower": var.lower,
                "upper": var.upper,
            }
            for var, v in zip(self.variables, values, strict=False)
        ]
        poses_out = []
        for k, pose in enumerate(poses):
            bodies = {self.ground: self.bodies[self.ground]["pose"]}
            for b in self.moving:
                ix = 3 * self.moving_index[b]
                bodies[b] = [float(x) for x in pose[ix : ix + 3]]
            poses_out.append({"sample": k, "bodies": bodies})
        return {
            "status": verdict,
            "mechanism": output_mechanism,
            "design_changes": changes,
            "body_local_dimensions": dimensions,
            "motion": {
                "body_poses": poses_out,
                "intermediate": intermediate,
                "joint_coordinates": states,
                "generated_targets": trajectory,
            },
            "validation": validation,
            "optimization": {
                "backend": "CasADi / IPOPT, exact joint constraints",
                "solver_success": successful,
                "return_status": status,
                "iterations": iterations,
                "objective": objective,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "number_of_design_parameters": len(self.variables),
                "number_of_configuration_variables": self.K * self.nq,
                "number_of_samples": self.K,
            },
            **({"solver_error": error} if error else {}),
        }


# The following two examples create target paths using elementary geometry.
# The CORE SOLVER ABOVE knows neither mechanism name nor specialized equations.
def fourbar_example(samples=12):
    A = np.array([0.0, 0.0])
    D = np.array([120.0, 0.0])
    a, b, c = 35.0, 105.0, 90.0
    path = []
    angles = np.linspace(0.40, 2.65, samples)
    for th in angles:
        B = A + a * np.array([math.cos(th), math.sin(th)])
        v = D - B
        dist = np.linalg.norm(v)
        u = v / dist
        x = (b * b - c * c + dist * dist) / (2 * dist)
        y = math.sqrt(max(0.0, b * b - x * x))
        C = B + x * u + y * np.array([-u[1], u[0]])
        phi = math.atan2(C[1] - B[1], C[0] - B[0])
        path.append((B + rot_numpy(phi) @ np.array([48.0, 23.0])).tolist())
    return {
        "units": "mm",
        "bodies": [
            {
                "id": "ground",
                "ground": True,
                "pose": [0, 0, 0],
                "frames": {"A": [0, 0, 0], "D": [120, 0, 0]},
            },
            {"id": "crank", "pose": [0, 0, 0], "frames": {"A": [0, 0, 0], "B": [31, 0, 0]}},
            {
                "id": "coupler",
                "pose": [31, 0, 0],
                "frames": {"B": [0, 0, 0], "C": [99, 0, 0]},
                "points": {"trace": {"xy": [42, 21], "bounds": {"x": [25, 65], "y": [8, 35]}}},
            },
            {"id": "rocker", "pose": [120 - 90, 0, 0], "frames": {"D": [0, 0, 0], "C": [94, 0, 0]}},
        ],
        "joints": [
            {"id": "A", "type": "revolute", "a": ["ground", "A"], "b": ["crank", "A"]},
            {"id": "B", "type": "revolute", "a": ["crank", "B"], "b": ["coupler", "B"]},
            {"id": "C", "type": "revolute", "a": ["coupler", "C"], "b": ["rocker", "C"]},
            {"id": "D", "type": "revolute", "a": ["ground", "D"], "b": ["rocker", "D"]},
        ],
        "targets": [{"body": "coupler", "point": "trace", "positions": path}],
        "drivers": [{"joint": "A", "values": angles.tolist()}],
        "design": {"mode": "auto", "max_shift_fraction": 0.3},
        "options": {
            "max_iter": 600,
            "trajectory_tolerance": 0.5,
            "closure_tolerance": 0.003,
            "verify_substeps": 2,
            "regularization": 1e-8,
            "smoothness": 1e-7,
        },
    }


def slidercrank_example(samples=10):
    r, rod_length, e = 26.0, 87.0, 9.0
    angles = np.linspace(-0.6, 1.5, samples)
    # Frame on ground axis at y=e, with slider reference at its own origin.
    path = []
    for th in angles:
        B = np.array([r * math.cos(th), r * math.sin(th)])
        x = B[0] + math.sqrt(rod_length * rod_length - (e - B[1]) ** 2)
        path.append([x, e])
    return {
        "units": "mm",
        "bodies": [
            {
                "id": "ground",
                "ground": True,
                "pose": [0, 0, 0],
                "frames": {"pivot": [0, 0, 0], "rail": [0, e, 0]},
            },
            {"id": "crank", "pose": [0, 0, 0], "frames": {"pivot": [0, 0, 0], "tip": [23, 0, 0]}},
            {"id": "rod", "pose": [23, 0, 0], "frames": {"prox": [0, 0, 0], "dist": [81, 0, 0]}},
            {
                "id": "slider",
                "pose": [104, e, 0],
                "frames": {"rail": [0, 0, 0], "pin": [0, 0, 0]},
                "points": {"output": {"xy": [0, 0], "free": []}},
            },
        ],
        "joints": [
            {"id": "base", "type": "revolute", "a": ["ground", "pivot"], "b": ["crank", "pivot"]},
            {"id": "elbow", "type": "revolute", "a": ["crank", "tip"], "b": ["rod", "prox"]},
            {"id": "pin", "type": "revolute", "a": ["rod", "dist"], "b": ["slider", "pin"]},
            {"id": "rail", "type": "prismatic", "a": ["ground", "rail"], "b": ["slider", "rail"]},
        ],
        "targets": [{"body": "slider", "point": "output", "positions": path}],
        "drivers": [{"joint": "base", "values": angles.tolist()}],
        "design": {"mode": "auto", "max_shift_fraction": 0.25},
        "options": {
            "max_iter": 600,
            "trajectory_tolerance": 0.4,
            "closure_tolerance": 0.003,
            "verify_substeps": 2,
            "regularization": 1e-8,
            "smoothness": 1e-7,
        },
    }


def fivebar_example(samples=9):
    """Two-DOF five-revolute parallel linkage: no changes to solver internals."""
    A = np.array([0.0, 0.0])
    E = np.array([140.0, 0.0])
    a1 = np.linspace(0.55, 1.20, samples)
    a2 = np.linspace(2.00, 2.40, samples)
    path = []
    for th1, th2 in zip(a1, a2, strict=False):
        B = A + 35 * np.array([math.cos(th1), math.sin(th1)])
        D = E + 30 * np.array([math.cos(th2), math.sin(th2)])
        v = D - B
        distance = float(np.linalg.norm(v))
        u = v / distance
        x = (100**2 - 110**2 + distance**2) / (2 * distance)
        y = math.sqrt(max(0.0, 100**2 - x * x))
        C = B + x * u + y * np.array([-u[1], u[0]])
        path.append(C.tolist())
    return {
        "units": "mm",
        "bodies": [
            {"id": "ground", "ground": True, "frames": {"A": [0, 0, 0], "E": [140, 0, 0]}},
            {"id": "left_crank", "pose": [0, 0, 0], "frames": {"A": [0, 0, 0], "B": [33, 0, 0]}},
            {"id": "left_rod", "pose": [33, 0, 0], "frames": {"B": [0, 0, 0], "C": [95, 0, 0]}},
            {"id": "right_rod", "pose": [115, 0, 0], "frames": {"C": [104, 0, 0], "D": [0, 0, 0]}},
            {"id": "right_crank", "pose": [140, 0, 0], "frames": {"E": [0, 0, 0], "D": [31, 0, 0]}},
        ],
        "joints": [
            {"id": "A", "type": "revolute", "a": ["ground", "A"], "b": ["left_crank", "A"]},
            {"id": "B", "type": "revolute", "a": ["left_crank", "B"], "b": ["left_rod", "B"]},
            {"id": "C", "type": "revolute", "a": ["left_rod", "C"], "b": ["right_rod", "C"]},
            {"id": "D", "type": "revolute", "a": ["right_rod", "D"], "b": ["right_crank", "D"]},
            {"id": "E", "type": "revolute", "a": ["ground", "E"], "b": ["right_crank", "E"]},
        ],
        # The trace is the coupler joint itself: target a named frame rather
        # than duplicating its coordinates in an unrelated body-local point.
        "targets": [{"body": "left_rod", "frame": "C", "positions": path}],
        "drivers": [{"joint": "A", "values": a1.tolist()}, {"joint": "E", "values": a2.tolist()}],
        "geometry_constraints": [
            {
                "type": "distance",
                "body": "left_crank",
                "a": ["frame", "A"],
                "b": ["frame", "B"],
                "min": 15,
                "max": 65,
            },
            {
                "type": "distance",
                "body": "left_rod",
                "a": ["frame", "B"],
                "b": ["frame", "C"],
                "min": 70,
                "max": 135,
            },
            {
                "type": "distance",
                "body": "right_rod",
                "a": ["frame", "C"],
                "b": ["frame", "D"],
                "min": 75,
                "max": 140,
            },
        ],
        "design": {"mode": "auto", "max_shift_fraction": 0.4},
        "options": {
            "max_iter": 800,
            "closure_tolerance": 0.005,
            "verify_substeps": 2,
            "trajectory_tolerance": 0.5,
            "regularization": 1e-8,
            "max_step_angle_radians": 0.6,
            "smoothness": 1e-7,
        },
    }


def schema():
    """Agent-readable concise JSON Schema plus semantic documentation."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Planar rigid-body dimensional synthesis input",
        "description": "Mechanism-neutral graph of bodies and joint frames with sampled motion targets. Radians only. See --help / module docstring for constraints, optimization policy, and output interpretation.",
        "type": "object",
        "required": ["bodies", "joints", "targets"],
        "properties": {
            "units": {
                "type": "string",
                "description": "Informational length units; use consistent lengths across input.",
            },
            "bodies": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "required": ["id", "frames"],
                    "properties": {
                        "id": {"type": "string"},
                        "ground": {"type": "boolean"},
                        "pose": {"$ref": "#/$defs/pose"},
                        "frames": {
                            "type": "object",
                            "additionalProperties": {
                                "oneOf": [{"$ref": "#/$defs/pose"}, {"$ref": "#/$defs/frame"}]
                            },
                        },
                        "points": {
                            "type": "object",
                            "additionalProperties": {
                                "oneOf": [{"$ref": "#/$defs/point2"}, {"$ref": "#/$defs/point"}]
                            },
                        },
                    },
                },
            },
            "joints": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "type", "a", "b"],
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"enum": ["revolute", "prismatic", "fixed"]},
                        "a": {"$ref": "#/$defs/ref"},
                        "b": {"$ref": "#/$defs/ref"},
                        "limits": {
                            "type": "array",
                            "prefixItems": [{"type": "number"}, {"type": "number"}],
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
            },
            "targets": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["body"],
                    "properties": {
                        "body": {"type": "string"},
                        "point": {"type": "string"},
                        "frame": {"type": "string"},
                        "positions": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"$ref": "#/$defs/point2"},
                        },
                        "orientations": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"type": "number"},
                        },
                        "position_weight": {"type": "number", "minimum": 0},
                        "orientation_weight": {"type": "number", "minimum": 0},
                    },
                    "anyOf": [
                        {"required": ["positions", "point"]},
                        {"required": ["positions", "frame"]},
                        {"required": ["orientations"]},
                    ],
                },
            },
            "geometry_constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "body", "a", "b"],
                    "properties": {
                        "type": {"const": "distance"},
                        "body": {"type": "string"},
                        "a": {"$ref": "#/$defs/feature_ref"},
                        "b": {"$ref": "#/$defs/feature_ref"},
                        "min": {"type": "number", "minimum": 0},
                        "max": {"type": "number", "minimum": 0},
                    },
                    "anyOf": [{"required": ["min"]}, {"required": ["max"]}],
                },
            },
            "drivers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["joint"],
                    "properties": {
                        "joint": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "number"}},
                        "monotonic": {"enum": ["increasing", "decreasing"]},
                    },
                },
            },
            "design": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["auto", "explicit"]},
                    "max_shift_fraction": {"type": "number", "exclusiveMinimum": 0},
                    "max_axis_rotation": {"type": "number", "exclusiveMinimum": 0},
                },
            },
            "options": {
                "type": "object",
                "properties": {
                    "max_iter": {"type": "integer", "minimum": 1},
                    "regularization": {"type": "number", "minimum": 0},
                    "smoothness": {"type": "number", "minimum": 0},
                    "verify_substeps": {"type": "integer", "minimum": 0},
                    "closure_tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "driver_tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "trajectory_tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "orientation_tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "continuation_jump_threshold": {"type": "number", "exclusiveMinimum": 0},
                    "nlp_tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "max_step_angle_radians": {"type": "number", "exclusiveMinimum": 0},
                    "max_step_translation_fraction": {"type": "number", "exclusiveMinimum": 0},
                    "print_level": {"type": "integer", "minimum": 0},
                },
            },
        },
        "$defs": {
            "pose": {
                "type": "array",
                "prefixItems": [{"type": "number"}, {"type": "number"}, {"type": "number"}],
                "minItems": 3,
                "maxItems": 3,
            },
            "point2": {
                "type": "array",
                "prefixItems": [{"type": "number"}, {"type": "number"}],
                "minItems": 2,
                "maxItems": 2,
            },
            "feature_ref": {
                "type": "array",
                "prefixItems": [{"enum": ["frame", "point"]}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            },
            "ref": {
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            },
            "bounds": {
                "type": "object",
                "properties": {
                    axis: {
                        "type": "array",
                        "prefixItems": [{"type": "number"}, {"type": "number"}],
                        "minItems": 2,
                        "maxItems": 2,
                    }
                    for axis in ("x", "y", "theta")
                },
            },
            "frame": {
                "type": "object",
                "required": ["pose"],
                "properties": {
                    "pose": {"$ref": "#/$defs/pose"},
                    "free": {
                        "type": "array",
                        "items": {"enum": ["x", "y", "theta"]},
                        "uniqueItems": True,
                    },
                    "bounds": {"$ref": "#/$defs/bounds"},
                },
            },
            "point": {
                "type": "object",
                "required": ["xy"],
                "properties": {
                    "xy": {"$ref": "#/$defs/point2"},
                    "free": {"type": "array", "items": {"enum": ["x", "y"]}, "uniqueItems": True},
                    "bounds": {"$ref": "#/$defs/bounds"},
                },
            },
        },
        "x-output-contract": {
            "status": ["success", "infeasible_or_unverified", "solver_failed"],
            "required_success_checks": [
                "validation.closure_max_length_units",
                "validation.target_position_max_length_units",
                "validation.continuation_pass",
            ],
            "artifacts": [
                "mechanism",
                "design_changes",
                "body_local_dimensions",
                "motion",
                "validation",
                "optimization",
            ],
        },
    }


def synthesize_planar_mechanism(specification: dict) -> dict:
    """Main agent-callable synthesis tool.

    Args:
        specification: Mechanism-neutral JSON-compatible dictionary following
            schema() / --schema and the module-level AGENT TOOL DESCRIPTION.

    Returns:
        JSON-compatible dictionary with status, optimized mechanism (its
        `pose` fields are assembled at sample 0), body_local_dimensions,
        design_changes, motion, validation, and optimization diagnostics.
        Callers must check status == 'success' AND review validation warnings.

    Raises:
        SpecificationError: Invalid graph, missing frames, malformed targets.
        RuntimeError: Missing CasADi or other unrecoverable backend issue.

    This is a nonconvex, local, planar *kinematic* synthesis method. It does not
    select topology, model solid body collisions, or prove global optimality.
    """
    return PlanarModel(specification).solve()


def run_self_test():
    failures = []
    for name, fun in [
        ("fourbar", fourbar_example),
        ("slidercrank", slidercrank_example),
        ("fivebar", fivebar_example),
    ]:
        try:
            spec = fun(8)
            result = PlanarModel(spec).solve()
            checks = {
                "status": result["status"],
                "fit_max": result["validation"]["target_position_max_length_units"],
                "closure_max": result["validation"]["closure_max_length_units"],
                "continuation_pass": result["validation"]["continuation_pass"],
                "design_variables": result["optimization"]["number_of_design_parameters"],
                "ipopt": result["optimization"]["return_status"],
            }
            print(name, json.dumps(checks))
            if result["status"] != "success":
                failures.append(name)
        except Exception as exc:
            print(name, repr(exc), file=sys.stderr)
            failures.append(name)
    return len(failures) == 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Path to mechanism JSON; use - for stdin")
    g.add_argument("--schema", action="store_true", help="Print machine-readable JSON Schema")
    g.add_argument(
        "--example",
        choices=["fourbar", "slidercrank", "fivebar"],
        help="Print a complete input example; does not solve",
    )
    g.add_argument(
        "--demo",
        choices=["fourbar", "slidercrank", "fivebar"],
        help="Run built-in mechanism-neutral synthesis benchmark",
    )
    g.add_argument(
        "--self-test",
        action="store_true",
        help="Run integrated smoke tests on two distinct joint graphs",
    )
    parser.add_argument("--output", default="-", help="JSON output path; default stdout")
    args = parser.parse_args(argv)
    if args.schema:
        result = schema()
    elif args.example:
        result = {
            "fourbar": fourbar_example,
            "slidercrank": slidercrank_example,
            "fivebar": fivebar_example,
        }[args.example]()
    elif args.self_test:
        return 0 if run_self_test() else 1
    else:
        if args.demo:
            raw = {
                "fourbar": fourbar_example,
                "slidercrank": slidercrank_example,
                "fivebar": fivebar_example,
            }[args.demo]()
        else:
            with (
                sys.stdin if args.input == "-" else Path(args.input).open(encoding="utf8") as stream
            ):
                raw = json.load(stream)
        try:
            result = synthesize_planar_mechanism(raw)
        except (SpecificationError, ValueError) as exc:
            print(f"Invalid mechanism specification: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Solver error: {exc}", file=sys.stderr)
            return 3
    content = json.dumps(result, indent=2, allow_nan=False)
    if args.output == "-":
        print(content)
    else:
        with Path(args.output).open("w", encoding="utf8") as f:
            f.write(content + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    if isinstance(result, dict) and "status" in result and result["status"] != "success":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
