import math

from multi_agent_ai.agents import (
    BatteryHealthAgent,
    DeepRLEMSPolicy,
    DriverPedalModel,
    MultiAgentSystem,
    ProjectSupervisorAgent,
    build_benchmark_scenarios,
    build_realistic_drive_cycle,
    build_sample_state,
    compare_policies,
)


def test_build_realistic_drive_cycle_has_realistic_segments():
    cycle = build_realistic_drive_cycle()
    assert len(cycle) > 10
    assert all(segment["speed_kph"] >= 0 for segment in cycle)
    assert cycle[0]["speed_kph"] >= 0


def test_driver_pedal_model_maps_torque_request_to_motor_support():
    model = DriverPedalModel()
    demand = model.compute_torque_request(0.7, 60.0)
    assert demand["driver_pedal"] == 0.7
    assert demand["requested_total_torque_nm"] > 0
    assert demand["motor_support_fraction"] >= 0.0
    assert demand["motor_support_fraction"] <= 1.0


def test_battery_health_agent_limits_charge_and_discharge():
    agent = BatteryHealthAgent()
    result = agent.update(
        soc=55.0,
        temperature_c=30.0,
        charging_power_kw=35.0,
        discharging_power_kw=50.0,
        age_factor=0.2,
    )
    assert "max_charge_kw" in result
    assert "max_discharge_kw" in result
    assert result["max_charge_kw"] > 0
    assert result["max_discharge_kw"] > 0


def test_project_supervisor_agent_reports_project_status():
    agent = ProjectSupervisorAgent()
    report = agent.evaluate(
        code_readiness=0.88,
        github_readiness=0.9,
        writing_quality=0.86,
        literature_quality=0.82,
        simulation_quality=0.91,
    )
    assert report["status"] in {"ready", "watch", "needs_attention"}
    assert report["overall_score"] >= 0.0


def test_multi_agent_system_runs_with_realistic_template_state():
    system = MultiAgentSystem()
    state = build_sample_state()
    result = system.run(state)
    assert "final_decision" in result
    assert "agent_outputs" in result
    assert result["agent_outputs"]["performance"]["status"] in {"efficient", "stable", "degraded"}


def test_deep_rl_policy_and_benchmark_are_available():
    policy = DeepRLEMSPolicy()
    decision = policy.select_action(build_sample_state())
    assert decision["commanded_em_kw"] >= 0
    assert decision["commanded_ice_kw"] >= 0
    scenarios = build_benchmark_scenarios()
    assert len(scenarios) >= 3
    benchmark = compare_policies()
    assert "deep_rl_policy" in benchmark
    assert "rule_based_baseline" in benchmark
