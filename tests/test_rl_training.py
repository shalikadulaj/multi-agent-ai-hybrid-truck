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


def test_transition_enforces_battery_operating_window():
    state = EpisodeState(soc_pct=89.0, battery_temp_c=30.0, soh_pct=96.0, degradation_proxy=0.0)
    point = make_drive_cycle("city_stop_go", np.random.default_rng(3), steps=1)[0]
    transition(state, point, action=7, config=PlantConfig())
    assert 10.0 <= state.soc_pct <= 95.0
    assert state.battery_temp_c < 40.0


def test_transition_accounts_for_battery_energy_in_reward():
    state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    point = make_drive_cycle("highway_cruise", np.random.default_rng(8), steps=1)[0]
    reward, metrics = transition(state, point, action=4, config=PlantConfig())
    assert metrics["battery_equivalent_fuel_l"] > 0.0
    assert reward < -100.0 * metrics["fuel_l"]


def test_electric_gear_selection_changes_motor_energy_use():
    point = make_drive_cycle("highway_cruise", np.random.default_rng(88), steps=1)[0]
    low_gear_state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    high_gear_state = EpisodeState(soc_pct=60.0, battery_temp_c=25.0, soh_pct=96.0, degradation_proxy=0.0)
    _, low_gear = transition(low_gear_state, point, action=4, config=PlantConfig())
    _, high_gear = transition(high_gear_state, point, action=5, config=PlantConfig())
    assert low_gear["electric_gear"] != high_gear["electric_gear"]
    assert low_gear_state.soc_pct != high_gear_state.soc_pct


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
    assert {row["policy"] for row in summary} == {"no_assist", "rule_based", "trained_q_learning"}
