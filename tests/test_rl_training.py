from pathlib import Path

import numpy as np

from multi_agent_ai.rl_training import (
    N_ACTIONS,
    N_STATES,
    EpisodeState,
    PlantConfig,
    TabularQPolicy,
    encode_observation,
    evaluate_policies,
    make_drive_cycle,
    simulate_episode,
    train_q_learning,
    transition,
)


def test_observation_encoder_and_action_table_have_valid_dimensions():
    index = encode_observation(60.0, 100.0, 55.0, 1.5, 28.0, False)
    assert 0 <= index < N_STATES
    assert N_ACTIONS == 8


def test_cycle_generation_is_reproducible_and_scenario_specific():
    first = make_drive_cycle("hilly_grade", np.random.default_rng(44), steps=24)
    second = make_drive_cycle("hilly_grade", np.random.default_rng(44), steps=24)
    assert first == second
    assert len(first) == 24
    assert max(point.grade_pct for point in first) > 3.0
    speed_deltas = [second.speed_kph - first.speed_kph for first, second in zip(first, first[1:])]
    assert max(speed_deltas) <= PlantConfig().maximum_cycle_acceleration_m_s2 * 3.6 + 1e-9
    assert min(speed_deltas) >= -PlantConfig().maximum_cycle_deceleration_m_s2 * 3.6 - 1e-9


def test_transition_enforces_battery_operating_window():
    state = EpisodeState(soc_pct=89.0, battery_temp_c=30.0, soh_pct=96.0, degradation_proxy=0.0)
    point = make_drive_cycle("city_stop_go", np.random.default_rng(3), steps=1)[0]
    transition(state, point, action=7, config=PlantConfig())
    assert 10.0 <= state.soc_pct <= 95.0
    assert state.battery_temp_c < 40.0
    assert state.speed_kph >= 0.0


def test_transition_accounts_for_battery_energy_in_reward():
    state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    point = make_drive_cycle("highway_cruise", np.random.default_rng(8), steps=1)[0]
    reward, metrics = transition(state, point, action=4, config=PlantConfig())
    assert metrics["battery_equivalent_fuel_l"] > 0.0
    assert reward < -100.0 * metrics["fuel_l"]


def test_transition_does_not_reset_randomized_initial_soh():
    point = make_drive_cycle("highway_cruise", np.random.default_rng(9), steps=2)[0]
    state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=88.0, degradation_proxy=0.0)
    transition(state, point, action=0, config=PlantConfig())
    assert state.soh_pct <= 88.0


def test_electric_gear_selection_changes_motor_energy_use():
    point = make_drive_cycle("highway_cruise", np.random.default_rng(88), steps=1)[0]
    low_gear_state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    high_gear_state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    _, low_gear = transition(low_gear_state, point, action=4, config=PlantConfig())
    _, high_gear = transition(high_gear_state, point, action=5, config=PlantConfig())
    assert low_gear["electric_gear"] != high_gear["electric_gear"]
    assert low_gear_state.soc_pct != high_gear_state.soc_pct
    assert low_gear["electric_path_efficiency"] != high_gear["electric_path_efficiency"]


def test_p4_longitudinal_dynamics_integrate_speed_and_separate_axles():
    cycle = make_drive_cycle("highway_cruise", np.random.default_rng(72), steps=8)
    state = EpisodeState(
        soc_pct=60.0,
        battery_temp_c=25.0,
        soh_pct=96.0,
        degradation_proxy=0.0,
        speed_kph=cycle[0].speed_kph,
    )
    _, metrics = transition(state, cycle[1], action=4, config=PlantConfig())
    assert metrics["road_load_force_n"] > 0.0
    assert metrics["engine_axle_force_n"] > 0.0
    assert metrics["electric_axle_force_n"] > 0.0
    assert metrics["engine_axle_force_n"] != metrics["electric_axle_force_n"]
    assert state.speed_kph == metrics["speed_kph"]
    assert state.distance_km > 0.0


def test_regeneration_reduces_friction_brake_work_and_charges_battery():
    cycle = make_drive_cycle("city_stop_go", np.random.default_rng(73), steps=30)
    point = next(item for item in cycle if item.brake_pedal > 0.0)
    no_regen_state = EpisodeState(
        soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0, speed_kph=point.speed_kph
    )
    regen_state = EpisodeState(
        soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0, speed_kph=point.speed_kph
    )
    _, no_regen = transition(no_regen_state, point, action=0, config=PlantConfig())
    _, with_regen = transition(regen_state, point, action=6, config=PlantConfig())
    assert with_regen["regen_power_kw"] > no_regen["regen_power_kw"]
    assert with_regen["friction_brake_kwh"] < no_regen["friction_brake_kwh"]
    assert regen_state.soc_pct > no_regen_state.soc_pct


def test_q_learning_trains_and_policy_roundtrips(tmp_path: Path):
    policy, log = train_q_learning(episodes=12, seed=31, steps_per_episode=24)
    assert len(log) == 12
    assert np.any(policy.q_values != 0.0)
    target = tmp_path / "q_policy.csv"
    policy.save(target)
    reloaded = TabularQPolicy.load(target)
    np.testing.assert_allclose(policy.q_values, reloaded.q_values, rtol=1e-5, atol=1e-6)


def test_evaluation_reports_charge_sustaining_metrics():
    policy, _ = train_q_learning(episodes=8, seed=17, steps_per_episode=18)
    metrics, trace = simulate_episode("mixed_route", 99, policy, steps=30, capture_trace=True)
    assert metrics["distance_km"] > 0.0
    assert metrics["fuel_l_per_100km"] >= 0.0
    assert 10.0 <= metrics["final_soc_pct"] <= 95.0
    assert len(trace) == 30
    assert "charge_sustaining_equivalent_l_per_100km" in metrics
    summary = evaluate_policies(policy, seeds=(1,), steps=20)
    assert len(summary) == 4 * 3
    assert {row["policy"] for row in summary} == {"engine_only_regen", "rule_based", "trained_q_learning"}
