from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from build123d import Box, Pos
from harness import ModelQuery, WarmEnvironment, calls, run, run_scenario, text, tool_call
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation

from articraft.agent.tools import ToolContext, get
from articraft.agent.workspace import LocalWorkspace
from articraft.sdk import (
    BodyFrame,
    JointDOF,
    JointFrame,
    RigidBody,
    RigidBodyAssembly,
    TestContext,
    export_kinematic_urdf,
    verify_trajectory,
)
from articraft.sdk._synthesis import (
    InnerResult,
    SerialURDF,
    load_problem,
    save_outputs,
    task_digest,
)
from articraft.sdk.trajectory import _AssemblyKinematics, _read_tcp
from articraft.viewer import load_viewer_run


def slider(height: float = 0.1) -> tuple[RigidBodyAssembly, RigidBody, BodyFrame]:
    model = RigidBodyAssembly("slider")
    base = model.rigid_body("base")
    base.add(Box(0.02, 0.02, 0.02), name="body")
    tool = model.rigid_body("tool")
    tool.add(Box(0.01, 0.01, height), name="body")
    model.joint("slide", base.at(), tool.at(), dofs=[JointDOF("transX", limits=(-0.5, 0.5))])
    model.articulation("motion", root=base)
    return model, base, tool.at((0, 0, height))


def write_task(directory: Path, *, samples: int = 3) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model, base, tip = slider()
    fields = export_kinematic_urdf(model, directory / "initial.urdf", base=base, tip=tip)
    (directory / "trajectory.py").write_text(
        "import numpy as np\n"
        "def positions(t):\n"
        "    return np.column_stack((0.1 + 0.1*t, np.zeros_like(t), np.full_like(t, 0.2)))\n"
    )
    spec = {
        **fields,
        "tcp": {
            "description": "Working tip at the current tool height",
            "axes": {
                "x": "Travel direction",
                "y": "Lateral direction",
                "z": "Tool shaft direction",
            },
        },
        "trajectory": {"module": "trajectory.py", "parameter_range": [0, 1], "samples": samples},
        "design_parameters": [
            {
                "name": "height",
                "initial": 0.1,
                "lower": 0.05,
                "upper": 0.05 + 1 / 3,
                "bindings": [{"joint": "__trajectory_tip_mount", "xyz": [0, 0, 1]}],
            }
        ],
        "max_joint_step": [0.1],
        "position_tolerance_m": 0.001,
        "settings": {"inner_maxiter": 80},
    }
    path = directory / "task.json"
    path.write_text(json.dumps(spec))
    return path


def solve_task(task: Path) -> tuple[Path, dict]:
    solver, spec = load_problem(task)
    best, outer = solver.solve(verbose=False, subdivisions=4)
    output = task.parent / "solved"
    report = save_outputs(solver, best, outer, spec, output, subdivisions=4)
    report.update(task_digest=task_digest(task), subdivisions=4)
    (output / "result.json").write_text(json.dumps(report))
    return output, report


@pytest.mark.parametrize("reverse", [False, True])
def test_urdf_export_matches_sdk_for_rotated_child_frames_and_mixed_joints(tmp_path, reverse):
    model = RigidBodyAssembly("spatial")
    bodies = [model.rigid_body(name) for name in ("base", "arm", "tool")]
    for body in bodies:
        body.add(Box(0.01, 0.01, 0.01), name="body")
    base, arm, tool = bodies
    first = base.at(JointFrame((0.1, 0.2, 0.3), (0.2, -0.3, 0.1)))
    second = arm.at(JointFrame((0.03, -0.02, 0.05), (-0.1, 0.3, 0.4)))
    model.joint(
        "hinge",
        second if reverse else first,
        first if reverse else second,
        dofs=[JointDOF("rotY", limits=(-1, 1))],
    )
    model.joint(
        "slide",
        arm.at(JointFrame((0.1, 0, 0.3), (0.2, 0.1, 0))),
        tool.at(JointFrame((0.02, 0.03, 0), (0, 0.1, -0.2))),
        dofs=[JointDOF("transZ", limits=(-0.1, 0.3))],
    )
    model.articulation("motion", root=base)
    tip = tool.at(JointFrame((0.01, 0.02, 0.1), (0.1, 0.2, 0.3)))
    fields = export_kinematic_urdf(model, tmp_path / "initial.urdf", base=base, tip=tip)
    urdf = SerialURDF(tmp_path / fields["urdf"], fields["base_link"], fields["tip_link"], [])
    q = np.array([[0, 0], [0.7, 0.2], [-0.4, -0.05]])
    positions, rotations, _ = urdf.fk(np.array([]), q)
    actual = _AssemblyKinematics(model, base, tip, urdf.lower, urdf.upper)
    assert np.allclose(actual.fk(np.array([]), q)[2], urdf.fk(np.array([]), q)[2], atol=1e-10)
    for row, position, rotation in zip(q, positions, rotations, strict=True):
        state = model.resolve().forward_kinematics({"hinge.rotY": row[0], "slide.transZ": row[1]})
        target = state.frame_in(tip, base)
        assert np.allclose(position, target.xyz, atol=1e-10)
        assert np.allclose(rotation, Rotation.from_euler("xyz", target.rpy).as_matrix(), atol=1e-10)


@pytest.mark.parametrize("start,end", [(None, None), ([0.1], None), (None, [0.2]), ([0.1], [0.2])])
def test_optional_endpoints_are_free_and_supplied_endpoints_remain_exact(tmp_path, start, end):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec.update(q_start=start, q_end=end)
    task.write_text(json.dumps(spec))
    solver, _ = load_problem(task)
    result = solver.solve_internal(np.array([0.2]))
    assert result.feasible
    assert result.max_error < 1e-5
    if start is not None:
        assert np.array_equal(result.q[0], start)
    if end is not None:
        assert np.array_equal(result.q[-1], end)


def test_dense_validation_evaluates_original_equation_between_waypoints(tmp_path):
    task = write_task(tmp_path, samples=2)
    (tmp_path / "trajectory.py").write_text(
        "import numpy as np\ndef positions(t):\n"
        "    return np.column_stack((0.1+0.1*t, 0.1*np.sin(np.pi*t), np.full_like(t, 0.2)))\n"
    )
    solver, _ = load_problem(task)
    result = InnerResult(np.array([0.2]), np.array([[0.1], [0.2]]), 0, 0, True, True, "test")
    dense = solver.validate_dense(result, subdivisions=4)
    assert not dense["passes"]
    assert dense["max_error_m"] == pytest.approx(0.1)
    assert "original trajectory function" in dense["target_interpolation"]


@pytest.mark.parametrize("absolute", [None, 0.002])
def test_default_waypoints_and_bbox_relative_tolerance(tmp_path, absolute):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec["trajectory"].pop("samples")
    spec["trajectory"]["parameter_range"] = [0, np.pi / 2]
    spec.pop("position_tolerance_m")
    if absolute is not None:
        spec["position_tolerance_m"] = absolute
    task.write_text(json.dumps(spec))
    (tmp_path / "trajectory.py").write_text(
        "import numpy as np\ndef positions(t):\n"
        "    return np.column_stack((0.2+0.1*np.cos(t), 0.1*np.sin(t), 0.12+0.08*np.sin(2*t)**2))\n"
    )
    solver, _ = load_problem(task)
    assert solver.n == 21
    expected = absolute if absolute is not None else 0.03 * np.linalg.norm([0.1, 0.1, 0.08])
    assert solver.task.position_tolerance == pytest.approx(expected)


@pytest.mark.parametrize(
    "height,bow,absolute,stop_initially",
    [
        (0.1002, 0, None, True),
        (0.1005, 0, None, False),
        (0.1, 0.001, None, False),
        (0.1005, 0, 0.01, True),
        (0.1375, 0, None, False),
    ],
)
def test_early_stop_requires_one_tenth_of_object_limit(
    tmp_path, height, bow, absolute, stop_initially
):
    task = write_task(tmp_path, samples=2)
    spec = json.loads(task.read_text())
    spec.pop("position_tolerance_m")
    if absolute is not None:
        spec["position_tolerance_m"] = absolute
    spec["settings"]["outer_maxeval"] = 6
    spec["design_parameters"][0]["upper"] = 0.3
    task.write_text(json.dumps(spec))
    (tmp_path / "trajectory.py").write_text(
        "import numpy as np\ndef positions(t):\n"
        f"    return np.column_stack((0.1+0.1*t, {bow}*np.sin(np.pi*t), np.full_like(t, {height})))\n"
    )
    solver, _ = load_problem(task)
    best, outer = solver.solve(verbose=False, subdivisions=4)
    assert (len(solver.history) == 1) is stop_initially
    if height == 0.1375:
        assert outer.early_stopped
    if outer.get("early_stopped", False):
        assert best.max_error <= solver.task.position_tolerance / 10
        dense = solver.validate_dense(best, subdivisions=4)
        assert dense["passes"] and dense["max_error_m"] <= solver.task.position_tolerance / 10


@pytest.mark.parametrize("budget,dimensions", [(None, 1), (100, 1), (100, 12)])
def test_geometry_search_stops_after_ten_evaluations(tmp_path, budget, dimensions):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec["settings"]["outer_tol"] = 1e-12
    if budget is not None:
        spec["settings"]["outer_maxeval"] = budget
    parameter = spec["design_parameters"][0]
    spec["design_parameters"] = [dict(parameter, name=f"height_{i}") for i in range(dimensions)]
    task.write_text(json.dumps(spec))
    (tmp_path / "trajectory.py").write_text(
        "import numpy as np\ndef positions(t):\n"
        "    return np.column_stack((0.1+0.1*t, np.full_like(t, 0.1), np.full_like(t, 0.2)))\n"
    )
    solver, _ = load_problem(task)
    best, outer = solver.solve(verbose=False)
    assert len(solver.history) <= 10
    if dimensions > 10:
        assert len(solver.history) == 10
        assert outer.message == "Geometry evaluation limit (10) reached"
    else:
        assert outer.nfev == 10
    assert not outer.get("early_stopped", False)
    assert best.max_error == min(r.max_error for r in solver.cache.values() if r.feasible)


@pytest.mark.parametrize("offset,passes", [(0.0029, True), (0.0031, False)])
def test_relative_limit_applies_to_waypoint_and_dense_maximum_error(tmp_path, offset, passes):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec.pop("position_tolerance_m")
    task.write_text(json.dumps(spec))
    solver, spec = load_problem(task)
    q = np.linspace(0.1, 0.2, solver.n)[:, None] + offset
    best = solver._result(np.array([0.2]), q, True, "Known motion")
    report = save_outputs(
        solver, best, OptimizeResult(success=True, message="test"), spec, tmp_path / "solved"
    )
    assert report["position_tolerance_m"] == pytest.approx(0.003)
    assert report["waypoint_position_pass"] is passes
    assert report["dense_validation"]["position_pass"] is passes
    assert report["functional_success"] is passes


def test_optimized_feedback_requires_matching_rebuilt_tip_and_detects_stale_target(tmp_path):
    task = write_task(tmp_path)
    output, report = solve_task(task)
    assert report["functional_success"]
    assert report["design"]["height"] == pytest.approx(0.2, abs=0.001)
    model, base, tip = slider(report["design"]["height"])
    ctx = TestContext(model)
    preview = tmp_path / "preview.json"
    assert verify_trajectory(ctx, task, output, base=base, tip=tip, preview_path=preview)["passes"]
    saved = json.loads(preview.read_text())
    assert saved["base_body"] == "base"
    assert saved["tcp"]["body"] == "tool"
    assert saved["tcp"]["xyz_m"] == pytest.approx([0, 0, report["design"]["height"]])
    assert saved["tcp"]["axes"]["z"] == "Tool shaft direction"
    assert len(saved["target_positions_m"]) == 9
    assert np.allclose(np.array(saved["target_positions_m"])[:, 2], 0.2)
    assert np.allclose(np.array(saved["achieved_positions_m"])[:, 2], report["design"]["height"])
    assert ctx.report().passed
    wrong_model, wrong_base, wrong_tip = slider()
    ctx = TestContext(wrong_model)
    assert not verify_trajectory(ctx, task, output, base=wrong_base, tip=wrong_tip)["passes"]
    assert not ctx.report().passed
    (tmp_path / "trajectory.py").write_text("# edited\n" + (tmp_path / "trajectory.py").read_text())
    with pytest.raises(ValueError, match="changed"):
        verify_trajectory(ctx, task, output, base=wrong_base, tip=wrong_tip)


def test_tool_runs_isolated_solver_returns_compact_feedback_and_logs_stages(tmp_path, capsys):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("synthesis")
    workspace = run_dir / "workspace"
    write_task(workspace)
    ctx = ToolContext(env, run_dir, workspace)
    feedback = run(
        get("synthesize_trajectory").run(ctx, {"task_path": "task.json", "subdivisions": 4})
    )
    assert feedback["status"] == "success"
    assert feedback["tcp"]["body"] == "tool"
    assert feedback["tcp"]["xyz_m"] == pytest.approx([0, 0, feedback["design"]["height"]])
    assert feedback["tcp"]["description"] == "Working tip at the current tool height"
    assert "joint_order" in feedback and "joint_path" in feedback
    assert feedback["joint_configurations"]["slide"]["start"] == pytest.approx(0.1, abs=0.001)
    assert feedback["joint_configurations"]["slide"]["end"] == pytest.approx(0.2, abs=0.001)
    assert "q" not in feedback
    assert (workspace / feedback["joint_path"]).is_file()
    assert (run_dir / "trajectory_request.json").is_file()
    logs = capsys.readouterr().err
    for stage in (
        "Tool invoked",
        "Functional TCP before optimization",
        "Optimized functional TCP",
        "Optimizing",
        "Geometry evaluation",
        "Checking dense",
        "feedback to LLM",
    ):
        assert stage in logs


@pytest.mark.parametrize("cancel", [False, True])
def test_tool_timeout_and_cancellation_leave_no_live_worker(tmp_path, cancel):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("slow")
    task = write_task(run_dir / "workspace")
    (task.parent / "trajectory.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    ctx = ToolContext(env, run_dir, run_dir / "workspace")

    async def invoke():
        pending = asyncio.ensure_future(
            get("synthesize_trajectory").run(
                ctx,
                {"task_path": "task.json", "timeout_seconds": 8 if not cancel else 15},
            )
        )
        if cancel:
            async with asyncio.timeout(10):
                while not (task.parent / "trajectory.pid").is_file():
                    await asyncio.sleep(0.02)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert (await pending)["timed_out"]

    run(invoke())
    request = json.loads((run_dir / "trajectory_request.json").read_text())
    assert not (ctx.workspace / request["output"] / "result.json").exists()
    pid = int((task.parent / "trajectory.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_tool_reports_unsatisfied_task_separately_from_worker_failure(tmp_path):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("inaccurate")
    task = write_task(run_dir / "workspace")
    spec = json.loads(task.read_text())
    spec["design_parameters"][0]["upper"] = 0.15
    task.write_text(json.dumps(spec))
    ctx = ToolContext(env, run_dir, run_dir / "workspace")
    result = run(get("synthesize_trajectory").run(ctx, {"task_path": "task.json"}))
    assert result["status"] == "needs_revision"
    assert result["dense_validation"]["max_error_m"] >= 0.049
    spec["design_parameters"][0]["lower"] = 0.2
    task.write_text(json.dumps(spec))
    result = run(get("synthesize_trajectory").run(ctx, {"task_path": "task.json"}))
    assert result["status"] == "error"
    assert "lower < upper" in result["details"]


def test_timeout_invalidates_a_result_written_before_worker_exit(tmp_path, monkeypatch):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("partial")
    write_task(run_dir / "workspace")
    original_spawn = asyncio.create_subprocess_exec

    async def stalled_worker(*command, **kwargs):
        return await original_spawn(
            sys.executable,
            "-c",
            "import sys,time; from pathlib import Path; "
            "Path(sys.argv[1]).joinpath('result.json').write_text('{}'); time.sleep(30)",
            command[4],
            **kwargs,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_worker)
    ctx = ToolContext(env, run_dir, run_dir / "workspace")
    result = run(
        get("synthesize_trajectory").run(ctx, {"task_path": "task.json", "timeout_seconds": 1})
    )
    assert result["timed_out"]
    request = json.loads((run_dir / "trajectory_request.json").read_text())
    assert not (ctx.workspace / request["output"] / "result.json").exists()


@pytest.mark.parametrize("quaternions", [[0, 0, 0, 1], [[0, 0, 0, 1]], [[0, 0, 0, 1]] * 3])
def test_rebuilt_orientation_is_checked_with_original_position_curve(tmp_path, quaternions):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec["waypoints"] = {"quaternions_xyzw": quaternions}
    spec["orientation_tolerance_rad"] = 0.01
    task.write_text(json.dumps(spec))
    output, report = solve_task(task)
    assert report["functional_success"]
    model, base, tip = slider(report["design"]["height"])
    ctx = TestContext(model)
    rotated = tip.body.at(JointFrame(tip.xyz, (0, 0, 0.1)))
    preview = tmp_path / "pose-preview.json"
    result = verify_trajectory(ctx, task, output, base=base, tip=rotated, preview_path=preview)
    saved = json.loads(preview.read_text())
    assert np.allclose(saved["target_quaternions_xyzw"], [0, 0, 0, 1])
    assert not np.allclose(saved["achieved_quaternions_xyzw"], [0, 0, 0, 1])
    assert result["position_pass"]
    assert not result["constraints_pass"]


@pytest.mark.parametrize("single_row", [False, True])
def test_fixed_nonzero_orientation_is_repeated_for_every_target_sample(tmp_path, single_row):
    task = write_task(tmp_path, samples=7)
    spec = json.loads(task.read_text())
    quaternion = Rotation.from_euler("xyz", [0.2, -0.3, 0.7]).as_quat().tolist()
    spec["waypoints"] = {"quaternions_xyzw": [quaternion] if single_row else quaternion}
    task.write_text(json.dumps(spec))
    solver, _ = load_problem(task)
    assert solver.task.rotations is not None
    assert solver.task.rotations.shape == (7, 3, 3)
    assert np.allclose(solver.task.rotations, Rotation.from_quat(quaternion).as_matrix())
    best = InnerResult(np.array([0.2]), np.linspace(0.1, 0.2, 7)[:, None], 0, 0, True, True, "test")
    _, _, rotations = solver.dense_path(best, subdivisions=4)
    assert rotations is not None
    assert rotations.shape == (25, 3, 3)
    assert np.allclose(rotations, solver.task.rotations[0])


@pytest.mark.parametrize("count", [32, 34])
def test_orientation_sample_count_error_reports_counts_and_fixed_form(tmp_path, count):
    task = write_task(tmp_path, samples=33)
    spec = json.loads(task.read_text())
    spec["waypoints"] = {"quaternions_xyzw": [[0, 0, 0, 1]] * count}
    task.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=rf"expected 33 quaternion rows.*got shape \({count}, 4\)"):
        load_problem(task)


def test_tool_returns_orientation_count_error_with_fixed_orientation_guidance(tmp_path):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("orientation-count")
    task = write_task(run_dir / "workspace", samples=33)
    spec = json.loads(task.read_text())
    spec["waypoints"] = {"quaternions_xyzw": [[0, 0, 0, 1]] * 32}
    task.write_text(json.dumps(spec))
    ctx = ToolContext(env, run_dir, run_dir / "workspace")
    feedback = run(get("synthesize_trajectory").run(ctx, {"task_path": "task.json"}))
    assert feedback["status"] == "error"
    assert "expected 33 quaternion rows" in feedback["details"]
    assert "got shape (32, 4)" in feedback["details"]
    assert "[0, 0, 0, 1]" in feedback["details"]


def test_orientation_tolerance_cannot_be_silently_ignored_without_target(tmp_path):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec["orientation_tolerance_rad"] = float(np.deg2rad(2))
    task.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="orientation_tolerance_rad requires a target"):
        load_problem(task)


def test_fixed_endpoint_failure_is_reflected_in_verification_return_value(tmp_path):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    spec.update(q_start=[0.1], q_end=[0.2], position_tolerance_m=0.1)
    task.write_text(json.dumps(spec))
    output, report = solve_task(task)
    path = np.loadtxt(output / "joint_path.csv", delimiter=",", skiprows=1)
    path[0, 1] += 0.01
    np.savetxt(output / "joint_path.csv", path, delimiter=",", header="waypoint,slide", comments="")
    model, base, tip = slider(report["design"]["height"])
    ctx = TestContext(model)
    validation = verify_trajectory(ctx, task, output, base=base, tip=tip)
    assert validation["position_pass"]
    assert not validation["endpoints_pass"]
    assert not validation["passes"]
    assert not ctx.report().passed


def test_unsupported_closed_loops_and_multi_axis_joints_are_rejected(tmp_path):
    model, base, tip = slider()
    model.joint("closure", tip.body.at(), base.at())
    model.articulations.clear()
    model.articulation("motion", root=base, joints=["slide"])
    with pytest.raises(ValueError, match="not loops"):
        export_kinematic_urdf(model, tmp_path / "loop.urdf", base=base, tip=tip)
    model, base, tip = slider()
    model.joints.clear()
    model.articulations.clear()
    model.joint("ball", base.at(), tip.body.at(), dofs=[JointDOF("rotX"), JointDOF("rotY")])
    model.articulation("motion", root=base)
    with pytest.raises(ValueError, match="Multi-axis"):
        export_kinematic_urdf(model, tmp_path / "ball.urdf", base=base, tip=tip)


@pytest.mark.parametrize(
    "changes", [{"urdf": "../../outside.urdf"}, {"trajectory": {"module": "../../bad.py"}}]
)
def test_tool_rejects_input_path_escape(tmp_path, changes):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("paths")
    task = write_task(run_dir / "workspace")
    spec = json.loads(task.read_text())
    spec.update(changes)
    task.write_text(json.dumps(spec))
    ctx = ToolContext(env, run_dir, run_dir / "workspace")
    with pytest.raises(ValueError, match="inside"):
        run(get("synthesize_trajectory").run(ctx, {"task_path": "task.json"}))


MAIN = """from build123d import Box
from articraft.sdk import RigidBodyAssembly, JointDOF, TestContext
HEIGHT = {height!r}
def build_object_model():
    model = RigidBodyAssembly("slider")
    base = model.rigid_body("base")
    base.add(Box(0.02, 0.02, 0.02), name="body")
    tool = model.rigid_body("tool")
    tool.add(Box(0.01, 0.01, HEIGHT), name="body")
    model.joint("slide", base.at(), tool.at(), dofs=[JointDOF("transX", limits=(-0.5, 0.5))])
    model.articulation("motion", root=base)
    return model
object_model = build_object_model()
trajectory_base = object_model.get_rigid_body("base")
trajectory_tip = object_model.get_rigid_body("tool").at((0, 0, HEIGHT))
def run_tests():
    return TestContext(object_model).report()
"""


def test_agent_applies_numerical_feedback_and_compile_checks_current_geometry(tmp_path, capsys):
    def setup(query: ModelQuery):
        # Prepare the initial design using the same workspace files an agent authors.
        write_task(tmp_path / "trajectory" / "workspace")
        return calls(
            tool_call("synthesize_trajectory", {"task_path": "task.json", "subdivisions": 4})
        )

    def apply_feedback(query: ModelQuery):
        feedback = query.tool_outputs()[-1]["result"]
        assert feedback["status"] == "success"
        return calls(
            tool_call(
                "write",
                {"path": "main.py", "content": MAIN.format(height=feedback["design"]["height"])},
            )
        )

    artifacts = run_scenario(
        "Create a slider that traces the specified line equation",
        [setup, apply_feedback, calls(tool_call("compile")), text("Built the slider.")],
        env=WarmEnvironment(output_dir=tmp_path),
        run_id="trajectory",
        max_turns=4,
    )
    assert artifacts.record.status == "success"
    logs = capsys.readouterr().err
    assert "feedback added to conversation" in logs
    assert "Rebuilt SDK validation" in logs


def test_compile_blocks_unapplied_dimensions_and_failed_synthesis(tmp_path):
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("verification")
    workspace = run_dir / "workspace"
    task = write_task(workspace)
    output, report = solve_task(task)
    (run_dir / "trajectory_request.json").write_text(
        json.dumps({"task": "task.json", "output": "solved"})
    )
    (workspace / "main.py").write_text(MAIN.format(height=0.1))
    result = env.compile_path(run_dir)
    assert result["status"] == "error"
    assert "trajectory rebuilt tracking" in str(result["test_report"])
    (workspace / "main.py").write_text(MAIN.format(height=report["design"]["height"]))
    assert env.compile_path(run_dir)["status"] == "success"
    report["functional_success"] = False
    (output / "result.json").write_text(json.dumps(report))
    result = env.compile_path(run_dir)
    assert result["status"] == "error"
    assert "trajectory synthesis accepted" in str(result["test_report"])


def test_compile_saves_trajectory_with_each_object_version_and_viewer_reads_only_data(tmp_path):
    env = WarmEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("saved-path")
    workspace = run_dir / "workspace"
    task = write_task(workspace)
    _, report = solve_task(task)
    request = run_dir / "trajectory_request.json"
    request.write_text(json.dumps({"task": "task.json", "output": "solved"}))
    (workspace / "main.py").write_text(MAIN.format(height=report["design"]["height"]))

    assert env.compile_path(run_dir)["status"] == "success"
    viewer = load_viewer_run(run_dir)
    version = viewer.versions[0]
    assert version["end_effector_trajectory_url"] == "/api/trajectories/0000"
    saved = json.loads(viewer.trajectories["0000"].read_text())
    assert saved["validation"]["passes"]
    assert saved["tcp"]["body"] == "tool"
    assert np.allclose(saved["target_positions_m"], saved["achieved_positions_m"], atol=0.001)
    assert not (run_dir / "result" / ".trajectory-preview.json").exists()

    # Viewing must not execute workspace Python or recompute an older target.
    (workspace / "trajectory.py").write_text("raise RuntimeError('viewer must not execute this')")
    assert load_viewer_run(run_dir).trajectories["0000"] == viewer.trajectories["0000"]

    request.unlink()
    assert env.compile_path(run_dir)["status"] == "success"
    versions = load_viewer_run(run_dir).versions
    assert "end_effector_trajectory_url" not in versions[0]
    assert versions[1]["end_effector_trajectory_url"] == "/api/trajectories/0000"


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"description": "Tip", "axes": {"x": "Forward", "y": "Across", "z": ""}}],
)
def test_tool_requires_functional_tcp_description_before_starting_solver(tmp_path, metadata):
    env = LocalWorkspace(output_dir=tmp_path)
    run_dir = env.create_run("missing-tcp")
    task = write_task(run_dir / "workspace")
    spec = json.loads(task.read_text())
    spec["tcp"] = metadata
    task.write_text(json.dumps(spec))
    ctx = ToolContext(env, run_dir, run_dir / "workspace")
    with pytest.raises(ValueError, match="functional working point"):
        run(get("synthesize_trajectory").run(ctx, {"task_path": "task.json"}))
    assert not (run_dir / "trajectory_request.json").exists()


def test_tcp_axes_cannot_silently_change_even_for_position_only_targets(tmp_path):
    task = write_task(tmp_path)
    output, report = solve_task(task)
    model, base, tip = slider(report["design"]["height"])
    changed_axes = tip.body.at(JointFrame(tip.xyz, (0, 0, np.pi / 2)))
    ctx = TestContext(model)
    validation = verify_trajectory(ctx, task, output, base=base, tip=changed_axes)
    assert validation["position_pass"]
    assert validation["constraints_pass"]
    assert not validation["tcp_frame_pass"]
    assert not validation["passes"]
    assert not ctx.report().passed


def test_rotating_gripper_tracks_offset_functional_tcp_and_its_axes(tmp_path):
    model = RigidBodyAssembly("gripper")
    base = model.rigid_body("base")
    base.add(Box(0.04, 0.04, 0.01), name="mount")
    tool = model.rigid_body("gripper")
    length, center_z = 0.065, 0.0065
    for name, x in (("left_finger", -0.01), ("right_finger", 0.01)):
        tool.add(Pos(X=x, Y=length / 2, Z=center_z) * Box(0.005, length, 0.007), name=name)
    model.joint("wrist", base.at(), tool.at(), dofs=[JointDOF("rotZ", limits=(-2, 2))])
    model.articulation("motion", root=base)
    # Fingers extend along body +Y. Functional +X follows them, requiring a rotated TCP.
    tip = tool.at(JointFrame((0, length, center_z), (0, 0, np.pi / 2)))
    fields = export_kinematic_urdf(model, tmp_path / "initial.urdf", base=base, tip=tip)
    t = np.linspace(0, np.pi / 2, 3)
    targets = np.column_stack((-length * np.sin(t), length * np.cos(t), np.full_like(t, center_z)))
    (tmp_path / "trajectory.py").write_text(
        "import numpy as np\ndef positions(t):\n"
        f"    return np.column_stack((-{length}*np.sin(t), {length}*np.cos(t), "
        f"np.full_like(t, {center_z})))\n"
    )
    spec = {
        **fields,
        "tcp": {
            "description": "Midpoint between the distal finger tips",
            "axes": {"x": "Along fingers", "y": "Across jaw gap", "z": "Normal to palm"},
        },
        "trajectory": {"module": "trajectory.py", "parameter_range": [0, np.pi / 2], "samples": 3},
        "waypoints": {
            "quaternions_xyzw": Rotation.from_euler("z", (t + np.pi / 2)[:, None])
            .as_quat()
            .tolist(),
        },
        "design_parameters": [
            {
                "name": "finger_length",
                "initial": length,
                "lower": 0.06,
                "upper": 0.07,
                "bindings": [{"joint": "__trajectory_tip_mount", "xyz": [0, 1, 0]}],
            }
        ],
        "max_joint_step": [1.0],
        "position_tolerance_m": 0.002,
        "orientation_tolerance_rad": 0.001,
    }
    task = tmp_path / "task.json"
    task.write_text(json.dumps(spec))
    tcp = _read_tcp(spec, tmp_path / "initial.urdf")
    assert tcp["xyz_m"] == pytest.approx([0, length, center_z])
    assert tcp["rpy_rad"] == pytest.approx([0, 0, np.pi / 2])
    solver, _ = load_problem(task)
    output = tmp_path / "solved"
    best = InnerResult(np.array([length]), t[:, None], 0, 0, True, True, "Known wrist motion")
    report = save_outputs(
        solver,
        best,
        OptimizeResult(success=True, message="Known path"),
        spec,
        output,
        subdivisions=4,
    )
    assert report["functional_success"]
    report.update(task_digest=task_digest(task), subdivisions=4)
    (output / "result.json").write_text(json.dumps(report))
    ctx = TestContext(model)
    preview = tmp_path / "preview.json"
    assert verify_trajectory(ctx, task, output, base=base, tip=tip, preview_path=preview)["passes"]
    saved = json.loads(preview.read_text())
    assert saved["tcp"]["body"] == tcp["body"]
    assert saved["tcp"]["description"] == tcp["description"]
    assert saved["tcp"]["axes"] == tcp["axes"]
    assert saved["tcp"]["xyz_m"] == pytest.approx(tcp["xyz_m"])
    assert saved["tcp"]["rpy_rad"] == pytest.approx(tcp["rpy_rad"])
    assert np.allclose(np.array(saved["achieved_positions_m"])[::4], targets)
    assert np.allclose(saved["target_quaternions_xyzw"], saved["achieved_quaternions_xyzw"])

    wrist = _AssemblyKinematics(model, base, tool.at(), solver.model.lower, solver.model.upper)
    wrist_positions, _, _ = wrist.fk(np.array([length]), t[:, None])
    assert np.allclose(wrist_positions, 0)  # wrist stays still while the working tips move
    assert np.all(np.linalg.norm(targets - wrist_positions, axis=1) > 0.065)
    wrong = verify_trajectory(TestContext(model), task, output, base=base, tip=tool.at())
    assert not wrong["position_pass"]
    assert not wrong["tcp_frame_pass"]


def test_zero_offset_tcp_is_valid_when_explicitly_declared(tmp_path):
    task = write_task(tmp_path)
    spec = json.loads(task.read_text())
    model, base, tip = slider()
    export_kinematic_urdf(model, tmp_path / "initial.urdf", base=base, tip=tip.body.at())
    assert _read_tcp(spec, tmp_path / "initial.urdf")["xyz_m"] == [0, 0, 0]
