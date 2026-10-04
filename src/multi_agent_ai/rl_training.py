"""Reproducible tabular Q-learning EMS experiments for the hybrid-truck prototype.

The environment is a transparent, low-order longitudinal/powertrain model. It is
suitable for PC experiments and controller development, not vehicle certification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .agents import build_realistic_drive_cycle


OBSERVATION_BINS: tuple[tuple[float, ...], ...] = (
    (40.0, 65.0),  # SoC
    (65.0, 120.0, 175.0),  # traction/braking power request
    (45.0, 70.0),  # speed
    (1.0, 5.0),  # road grade
    (30.0,),  # battery temperature
    (0.5,),  # braking state
)
OBSERVATION_SHAPE = tuple(len(edges) + 1 for edges in OBSERVATION_BINS)
N_STATES = math.prod(OBSERVATION_SHAPE)
N_ACTIONS = 8
ASSIST_FRACTIONS = (0.0, 0.20, 0.40, 0.60)
REGEN_FRACTIONS = (0.0, 0.33, 0.67, 1.0)
SCENARIOS = ("city_stop_go", "mixed_route", "highway_cruise", "hilly_grade")


@dataclass(frozen=True)
class PlantConfig:
    """Transparent nominal parameters; calibrate against measured Sisu data."""

    timestep_s: float = 1.0
    battery_capacity_kwh: float = 120.0
    motor_max_power_kw: float = 120.0
    engine_max_power_kw: float = 300.0
    max_regen_power_kw: float = 110.0
    ambient_temp_c: float = 20.0
    initial_soc_pct: float = 60.0
    initial_soh_pct: float = 96.0
    reference_engine_efficiency: float = 0.38
    diesel_energy_kwh_per_l: float = 9.7


@dataclass
class EpisodeState:
    soc_pct: float
    battery_temp_c: float
    soh_pct: float
    degradation_proxy: float
    fuel_l: float = 0.0
    electric_discharge_kwh: float = 0.0
    recovered_energy_kwh: float = 0.0
    distance_km: float = 0.0
    cumulative_reward: float = 0.0


@dataclass(frozen=True)
class DrivePoint:
    speed_kph: float
    grade_pct: float
    demand_kw: float
    brake_pedal: float
    accelerator_pedal: float
    ambient_temp_c: float


def encode_observation(
    soc_pct: float,
    power_request_kw: float,
    speed_kph: float,
    grade_pct: float,
    battery_temp_c: float,
    braking: bool,
) -> int:
    """Discretize the measured state into a reproducible Q-table index."""
    values = (soc_pct, power_request_kw, speed_kph, grade_pct, battery_temp_c, float(braking))
    bins = tuple(int(np.digitize(value, edges)) for value, edges in zip(values, OBSERVATION_BINS))
    return int(np.ravel_multi_index(bins, OBSERVATION_SHAPE))


def _observation_power(point: DrivePoint) -> float:
    return point.demand_kw if point.brake_pedal <= 0.0 else point.speed_kph * point.brake_pedal


def _encode_point(state: EpisodeState, point: DrivePoint) -> int:
    return encode_observation(
        state.soc_pct,
        _observation_power(point),
        point.speed_kph,
        point.grade_pct,
        state.battery_temp_c,
        point.brake_pedal > 0.0,
    )


@lru_cache(maxsize=1)
def _base_mixed_cycle() -> list[dict[str, float | str]]:
    return build_realistic_drive_cycle()


def _mixed_point(index: int, phase: float, rng: np.random.Generator, offset: int) -> tuple[float, float, float, float, float]:
    source = _base_mixed_cycle()[(index + offset) % 120]
    return (
        float(source["speed_kph"]) + float(rng.normal(0.0, 2.0)),
        float(source["grade_pct"]) + float(rng.normal(0.0, 0.35)),
        float(source["accelerator_pedal"]),
        float(source["brake_pedal"]),
        float(source["predicted_demand_kw"]),
    )


def _city_point(index: int, phase: float, rng: np.random.Generator, offset: int = 0) -> tuple[float, float, float, float, float]:
    speed = 30.0 + 20.0 * math.sin(phase * 2.0) + float(rng.normal(0.0, 3.0))
    grade = 1.0 + 1.5 * math.sin(phase) + float(rng.normal(0.0, 0.4))
    brake = 0.65 if index % 24 in range(18, 22) else 0.0
    accel = float(np.clip(0.25 + 0.35 * abs(math.sin(phase * 1.7)), 0.0, 1.0))
    demand = 55.0 + 65.0 * accel + max(0.0, grade) * 3.0
    return speed, grade, accel, brake, demand


def _highway_point(index: int, phase: float, rng: np.random.Generator, offset: int = 0) -> tuple[float, float, float, float, float]:
    speed = 82.0 + 6.0 * math.sin(phase) + float(rng.normal(0.0, 1.5))
    grade = 0.5 + 1.2 * math.sin(phase * 1.4) + float(rng.normal(0.0, 0.25))
    brake = 0.25 if index % 50 in (47, 48) else 0.0
    accel = float(np.clip(0.35 + 0.08 * math.sin(phase * 2.0), 0.0, 1.0))
    demand = 105.0 + 22.0 * accel + max(0.0, grade) * 4.0
    return speed, grade, accel, brake, demand


def _hilly_point(index: int, phase: float, rng: np.random.Generator, offset: int = 0) -> tuple[float, float, float, float, float]:
    speed = 55.0 + 12.0 * math.sin(phase) + float(rng.normal(0.0, 2.0))
    grade = 4.0 + 4.0 * math.sin(phase * 1.5) + float(rng.normal(0.0, 0.6))
    brake = 0.45 if grade < 0.0 and index % 12 < 3 else 0.0
    accel = float(np.clip(0.42 + 0.20 * max(0.0, grade) / 8.0, 0.0, 1.0))
    demand = 85.0 + 35.0 * accel + max(0.0, grade) * 5.0
    return speed, grade, accel, brake, demand


def make_drive_cycle(scenario: str, rng: np.random.Generator, steps: int = 120) -> list[DrivePoint]:
    """Build a randomized, reproducible cycle for one of four duty-cycle families."""
    builders = {
        "city_stop_go": _city_point,
        "highway_cruise": _highway_point,
        "hilly_grade": _hilly_point,
    }
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    mixed_offset = int(rng.integers(0, 120))
    ambient_base = float(rng.uniform(5.0, 32.0))
    points: list[DrivePoint] = []
    for index in range(steps):
        phase = 2.0 * math.pi * index / max(steps, 1)
        builder = _mixed_point if scenario == "mixed_route" else builders[scenario]
        values = builder(index, phase, rng, mixed_offset)
        speed, grade, accel, brake, demand = values
        points.append(
            DrivePoint(
                speed_kph=float(np.clip(speed, 0.0, 100.0)),
                grade_pct=float(np.clip(grade, -8.0, 10.0)),
                demand_kw=float(np.clip(demand, 10.0, 240.0)),
                brake_pedal=float(np.clip(brake, 0.0, 1.0)),
                accelerator_pedal=float(np.clip(accel, 0.0, 1.0)),
                ambient_temp_c=float(np.clip(ambient_base + 2.0 * math.sin(phase / 4.0), -20.0, 40.0)),
            )
        )
    return points


def battery_power_limits(state: EpisodeState, config: PlantConfig) -> tuple[float, float]:
    """Return conservative discharge and charge limits from SoC and temperature."""
    soc_discharge_factor = float(np.clip((state.soc_pct - 20.0) / 20.0, 0.0, 1.0))
    soc_charge_factor = float(np.clip((90.0 - state.soc_pct) / 15.0, 0.0, 1.0))
    temperature_factor = float(np.clip(1.0 - max(0.0, state.battery_temp_c - 30.0) / 35.0, 0.35, 1.0))
    health_factor = float(np.clip(state.soh_pct / 100.0, 0.7, 1.0))
    return (
        config.motor_max_power_kw * soc_discharge_factor * temperature_factor * health_factor,
        config.max_regen_power_kw * soc_charge_factor * temperature_factor * health_factor,
    )


def transition(
    state: EpisodeState,
    point: DrivePoint,
    action: int,
    config: PlantConfig,
) -> tuple[float, dict[str, float]]:
    """Apply one bounded supervisory action to the low-order hybrid-truck plant."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"Action must be in [0, {N_ACTIONS - 1}], got {action}.")
    previous_soc_pct = state.soc_pct
    dt_h = config.timestep_s / 3600.0
    discharge_limit_kw, charge_limit_kw = battery_power_limits(state, config)
    motor_kw = 0.0
    regen_kw = 0.0
    engine_kw = 0.0
    unmet_kw = 0.0
    fuel_l = 0.0
    throughput_kwh = 0.0
    if point.brake_pedal > 0.0:
        regen_fraction = REGEN_FRACTIONS[action // 2]
        electric_gear = action % 2 + 1
        preferred_gear = 1 if point.speed_kph < 55.0 else 2
        motor_efficiency = 0.91 if electric_gear == preferred_gear else 0.87
        braking_power_kw = min(
            config.max_regen_power_kw,
            max(0.0, point.speed_kph * 0.75 * point.brake_pedal + max(0.0, -point.grade_pct) * 5.0),
        )
        regen_kw = min(braking_power_kw * regen_fraction, charge_limit_kw)
        throughput_kwh = regen_kw * dt_h * motor_efficiency
        state.soc_pct += throughput_kwh / config.battery_capacity_kwh * 100.0
        state.recovered_energy_kwh += throughput_kwh
    else:
        assist_fraction = ASSIST_FRACTIONS[action // 2]
        electric_gear = action % 2 + 1
        preferred_gear = 1 if point.speed_kph < 55.0 else 2
        motor_efficiency = 0.94 if electric_gear == preferred_gear else 0.89
        motor_kw = min(point.demand_kw * assist_fraction, discharge_limit_kw, config.motor_max_power_kw)
        engine_kw = min(max(0.0, point.demand_kw - motor_kw), config.engine_max_power_kw)
        unmet_kw = max(0.0, point.demand_kw - motor_kw - engine_kw)
        motor_battery_kw = motor_kw / motor_efficiency
        throughput_kwh = motor_battery_kw * dt_h
        state.soc_pct -= throughput_kwh / config.battery_capacity_kwh * 100.0
        state.electric_discharge_kwh += throughput_kwh

        if engine_kw > 0.0:
            # Nominal diesel engine efficiency map, not a measured Sisu engine map.
            engine_efficiency = 0.30 + 0.09 * math.exp(-((engine_kw - 120.0) / 75.0) ** 2)
            engine_efficiency -= 0.025 * max(0.0, engine_kw - 220.0) / 80.0
            engine_efficiency = float(np.clip(engine_efficiency, 0.25, 0.42))
            fuel_l = engine_kw * dt_h / (engine_efficiency * config.diesel_energy_kwh_per_l)
            state.fuel_l += fuel_l

    # First-order thermal and aging proxies for comparative control research.
    battery_loss_kw = abs(motor_kw) * (1.0 / motor_efficiency - 1.0) + regen_kw * (1.0 - motor_efficiency)
    state.battery_temp_c += dt_h * (0.025 * battery_loss_kw - 0.012 * (state.battery_temp_c - point.ambient_temp_c))
    stress = 1.0 + max(0.0, 35.0 - state.soc_pct) / 30.0 + max(0.0, state.battery_temp_c - 30.0) / 20.0
    state.degradation_proxy += throughput_kwh * stress * 1.0e-5
    state.soh_pct = max(70.0, config.initial_soh_pct - state.degradation_proxy * 100.0)
    state.soc_pct = float(np.clip(state.soc_pct, 10.0, 95.0))
    state.distance_km += point.speed_kph * config.timestep_s / 3600.0

    # Reward uses fuel-equivalent operating cost, cycle throughput, thermal stress,
    # unmet traction, and a terminal charge-sustaining penalty (added at episode end).
    fuel_cost = 100.0 * fuel_l
    battery_delta_kwh = (previous_soc_pct - state.soc_pct) / 100.0 * config.battery_capacity_kwh
    battery_equivalent_fuel_l = battery_delta_kwh / (
        config.reference_engine_efficiency * config.diesel_energy_kwh_per_l
    )
    battery_energy_cost = 100.0 * battery_equivalent_fuel_l
    cycle_cost = 0.65 * throughput_kwh * stress
    thermal_cost = 0.003 * max(0.0, state.battery_temp_c - 38.0) ** 2
    tracking_cost = 0.15 * unmet_kw
    reward = -(fuel_cost + battery_energy_cost + cycle_cost + thermal_cost + tracking_cost)
    metrics = {
        "engine_power_kw": engine_kw,
        "motor_power_kw": motor_kw,
        "regen_power_kw": regen_kw,
        "fuel_l": fuel_l,
        "battery_equivalent_fuel_l": battery_equivalent_fuel_l,
        "battery_throughput_kwh": throughput_kwh,
        "unmet_power_kw": unmet_kw,
        "electric_gear": float(electric_gear),
        "reward": reward,
    }
    state.cumulative_reward += reward
    return reward, metrics


class TabularQPolicy:
    """Learned discrete EMS policy backed by a portable state/action Q table."""

    def __init__(self, q_values: np.ndarray) -> None:
        if q_values.shape != (N_STATES, N_ACTIONS):
            raise ValueError(f"Expected Q table shape {(N_STATES, N_ACTIONS)}, got {q_values.shape}.")
        self.q_values = np.asarray(q_values, dtype=np.float32)

    def select_action(self, state: EpisodeState, point: DrivePoint, greedy: bool = True, rng: random.Random | None = None) -> int:
        observation = _encode_point(state, point)
        values = self.q_values[observation]
        if not greedy and rng is not None:
            return rng.randrange(N_ACTIONS)
        maxima = np.flatnonzero(values == values.max())
        return int(maxima[0])

    @classmethod
    def load(cls, path: Path) -> "TabularQPolicy":
        table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        if table.ndim == 1:
            table = table.reshape(1, -1)
        return cls(table[:, 1:])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "state_index," + ",".join(f"q_action_{i}" for i in range(N_ACTIONS))
        table = np.column_stack((np.arange(N_STATES), self.q_values))
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt=["%d"] + ["%.7g"] * N_ACTIONS)


def train_q_learning(
    episodes: int = 10000,
    seed: int = 2026,
    steps_per_episode: int = 120,
    alpha: float = 0.16,
    gamma: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.04,
) -> tuple[TabularQPolicy, list[dict[str, float | int | str]]]:
    """Train Q-learning with randomized cycles, initial SoC, and ambient conditions."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = np.random.default_rng(seed)
    action_rng = random.Random(seed)
    q_values = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    logs: list[dict[str, float | int | str]] = []
    config = PlantConfig()

    for episode in range(episodes):
        scenario = SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]
        cycle = make_drive_cycle(scenario, rng, steps_per_episode)
        initial_soc = float(rng.uniform(45.0, 72.0))
        state = EpisodeState(
            soc_pct=initial_soc,
            battery_temp_c=float(rng.uniform(12.0, 35.0)),
            soh_pct=float(rng.uniform(88.0, 100.0)),
            degradation_proxy=0.0,
        )
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-5.0 * episode / max(episodes - 1, 1))
        total_reward = 0.0

        for step, point in enumerate(cycle):
            observation = _encode_point(state, point)
            if action_rng.random() < epsilon:
                action = action_rng.randrange(N_ACTIONS)
            else:
                action = int(np.argmax(q_values[observation]))
            reward, _ = transition(state, point, action, config)
            total_reward += reward
            done = step == len(cycle) - 1
            if done:
                next_max = 0.0
            else:
                next_point = cycle[step + 1]
                next_observation = _encode_point(state, next_point)
                next_max = float(np.max(q_values[next_observation]))
            q_values[observation, action] += alpha * (reward + gamma * next_max - q_values[observation, action])

        logs.append({"episode": episode + 1, "scenario": scenario, "epsilon": epsilon, "return": total_reward})

    return TabularQPolicy(q_values), logs


def _fixed_policy_action(name: str, state: EpisodeState, point: DrivePoint) -> int:
    if point.brake_pedal > 0.0:
        return 7  # maximum safe regen request, electric axle gear 2
    if name == "no_assist":
        return 0
    # Safe rule-based comparator: assist more only when SoC is above the reserve.
    if state.soc_pct >= 62.0:
        level = 2
    elif state.soc_pct >= 55.0:
        level = 1
    else:
        level = 0
    gear = 0 if point.speed_kph < 55.0 else 1
    return level * 2 + gear


def simulate_episode(
    scenario: str,
    seed: int,
    policy: TabularQPolicy | str,
    steps: int = 120,
    initial_soc_pct: float = 60.0,
    capture_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Evaluate one deterministic-seed mission and optionally return its time series."""
    rng = np.random.default_rng(seed)
    cycle = make_drive_cycle(scenario, rng, steps)
    config = PlantConfig(initial_soc_pct=initial_soc_pct)
    state = EpisodeState(
        soc_pct=initial_soc_pct,
        battery_temp_c=float(rng.uniform(18.0, 28.0)),
        soh_pct=config.initial_soh_pct,
        degradation_proxy=0.0,
    )
    trace: list[dict[str, float]] = []
    for index, point in enumerate(cycle):
        if isinstance(policy, TabularQPolicy):
            action = policy.select_action(state, point)
        else:
            action = _fixed_policy_action(policy, state, point)
        reward, control = transition(state, point, action, config)
        if capture_trace:
            trace.append(
                {
                    "time_s": float(index * config.timestep_s),
                    "speed_kph": point.speed_kph,
                    "grade_pct": point.grade_pct,
                    "demand_kw": point.demand_kw,
                    "accelerator_pedal": point.accelerator_pedal,
                    "brake_pedal": point.brake_pedal,
                    "engine_power_kw": control["engine_power_kw"],
                    "motor_power_kw": control["motor_power_kw"],
                    "regen_power_kw": control["regen_power_kw"],
                    "electric_gear": control["electric_gear"],
                    "soc_pct": state.soc_pct,
                    "battery_temp_c": state.battery_temp_c,
                    "cumulative_fuel_l": state.fuel_l,
                    "step_reward": reward,
                }
            )

    score = state.cumulative_reward
    distance = max(state.distance_km, 1.0e-9)
    equivalent_fuel_l = state.fuel_l + (initial_soc_pct - state.soc_pct) / 100.0 * config.battery_capacity_kwh / (
        config.reference_engine_efficiency * config.diesel_energy_kwh_per_l
    )
    metrics: dict[str, Any] = {
        "scenario": scenario,
        "policy": "trained_q_learning" if isinstance(policy, TabularQPolicy) else policy,
        "seed": seed,
        "distance_km": state.distance_km,
        "fuel_l": state.fuel_l,
        "fuel_l_per_100km": state.fuel_l / distance * 100.0,
        "charge_sustaining_equivalent_fuel_l": equivalent_fuel_l,
        "charge_sustaining_equivalent_l_per_100km": equivalent_fuel_l / distance * 100.0,
        "initial_soc_pct": initial_soc_pct,
        "final_soc_pct": state.soc_pct,
        "soc_change_pct": state.soc_pct - initial_soc_pct,
        "battery_discharge_kwh": state.electric_discharge_kwh,
        "regen_recovered_kwh": state.recovered_energy_kwh,
        "battery_degradation_proxy": state.degradation_proxy,
        "final_battery_temp_c": state.battery_temp_c,
        "episode_return": score,
    }
    return metrics, trace


def evaluate_policies(
    policy: TabularQPolicy,
    seeds: tuple[int, ...] = tuple(range(4101, 4111)),
    steps: int = 120,
) -> list[dict[str, Any]]:
    """Compare the learned controller with a transparent no-assist and rule baseline."""
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in seeds:
            eval_seed = seed + scenario_index * 100
            for candidate in ("no_assist", "rule_based", policy):
                metrics, _ = simulate_episode(scenario, eval_seed, candidate, steps=steps)
                rows.append(metrics)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results_plot(training_log: list[dict[str, Any]], evaluation: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    episodes = np.asarray([int(row["episode"]) for row in training_log])
    returns = np.asarray([float(row["return"]) for row in training_log])
    smooth_window = max(1, min(500, len(returns) // 10))
    smooth = np.convolve(returns, np.ones(smooth_window) / smooth_window, mode="valid")
    axes[0].plot(episodes[: len(smooth)], smooth, color="#126782", linewidth=1.8)
    axes[0].set(title="Training progress (moving-average return)", xlabel="Episode", ylabel="Return")
    axes[0].grid(alpha=0.25)

    policies = ("no_assist", "rule_based", "trained_q_learning")
    colors = ("#8b9aa7", "#e09f3e", "#137c8b")
    scenarios = list(SCENARIOS)
    x = np.arange(len(scenarios))
    width = 0.24
    for offset, (name, color) in enumerate(zip(policies, colors)):
        means = [
            float(np.mean([row["charge_sustaining_equivalent_l_per_100km"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name]))
            for scenario in scenarios
        ]
        stds = [
            float(np.std([row["charge_sustaining_equivalent_l_per_100km"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name], ddof=1))
            for scenario in scenarios
        ]
        axes[1].bar(x + (offset - 1) * width, means, width, yerr=stds, capsize=3, label=name.replace("_", " "), color=color)
    axes[1].set(title="Held-out charge-sustaining equivalent", xlabel="Drive-cycle family", ylabel="Equivalent diesel (L/100 km)", xticks=x, xticklabels=[s.replace("_", "\n") for s in scenarios])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Hybrid-truck EMS: tabular Q-learning research simulation", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.parent.parent / "benchmark_dashboard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for policy in ("no_assist", "rule_based", "trained_q_learning"):
            subset = [row for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            summary.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "n_episodes": len(subset),
                    "mean_fuel_l_per_100km": float(np.mean([row["fuel_l_per_100km"] for row in subset])),
                    "std_fuel_l_per_100km": float(np.std([row["fuel_l_per_100km"] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                    "mean_charge_sustaining_equivalent_l_per_100km": float(np.mean([row["charge_sustaining_equivalent_l_per_100km"] for row in subset])),
                    "std_charge_sustaining_equivalent_l_per_100km": float(np.std([row["charge_sustaining_equivalent_l_per_100km"] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                    "mean_soc_change_pct": float(np.mean([row["soc_change_pct"] for row in subset])),
                    "mean_regen_kwh": float(np.mean([row["regen_recovered_kwh"] for row in subset])),
                    "mean_episode_return": float(np.mean([row["episode_return"] for row in subset])),
                }
            )
    return summary


def run_training(
    episodes: int = 10000,
    seed: int = 2026,
    output_dir: Path | None = None,
    steps_per_episode: int = 120,
) -> tuple[TabularQPolicy, list[dict[str, Any]], list[dict[str, Any]]]:
    """Train, save portable artifacts, and evaluate against transparent baselines."""
    root = output_dir or Path(__file__).resolve().parents[2] / "artifacts"
    policy, training_log = train_q_learning(episodes=episodes, seed=seed, steps_per_episode=steps_per_episode)
    evaluation = evaluate_policies(policy)
    summary = summarize_evaluation(evaluation)
    policy.save(root / "q_policy.csv")
    write_csv(root / "training_log.csv", training_log)
    write_csv(root / "evaluation_episodes.csv", evaluation)
    write_csv(root / "evaluation_summary.csv", summary)
    save_results_plot(training_log, evaluation, root / "rl_training_results.png")
    metadata = {
        "algorithm": "tabular Q-learning (epsilon-greedy)",
        "episodes": episodes,
        "seed": seed,
        "steps_per_episode": steps_per_episode,
        "observation_bins": OBSERVATION_BINS,
        "observation_shape": OBSERVATION_SHAPE,
        "action_count": N_ACTIONS,
        "propulsion_actions": {"assist_fraction": ASSIST_FRACTIONS, "electric_gear": [1, 2]},
        "braking_actions": {"regen_fraction": REGEN_FRACTIONS, "electric_gear": [1, 2]},
        "plant_config": asdict(PlantConfig()),
        "model_scope": "transparent low-order simulation; not calibrated to a specific Sisu truck or validated for real-time vehicle control",
    }
    (root / "q_policy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return policy, training_log, summary


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("Scenario            Policy               Fuel    CS-equiv   SoC Δ (%)   Regen")
    print("                                         L/100km    L/100km                  kWh")
    print("-" * 87)
    for row in rows:
        print(
            f"{row['scenario']:<20} {row['policy']:<20} "
            f"{row['mean_fuel_l_per_100km']:>8.2f} "
            f"{row['mean_charge_sustaining_equivalent_l_per_100km']:>9.2f} "
            f"{row['mean_soc_change_pct']:>10.3f} {row['mean_regen_kwh']:>8.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the hybrid truck tabular-Q EMS.")
    parser.add_argument("--episodes", type=int, default=10000, help="training episodes (default: 10000)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--evaluate-only", action="store_true", help="skip training and use the saved Q table")
    parser.add_argument("--model", type=Path, default=None, help="Q table CSV for --evaluate-only")
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed_route")
    parser.add_argument("--trajectory-csv", type=Path, default=None, help="write a trained-policy rollout CSV")
    args = parser.parse_args(argv)

    if args.evaluate_only:
        model_path = args.model or args.output_dir / "q_policy.csv"
        policy = TabularQPolicy.load(model_path)
        if args.trajectory_csv:
            metrics, trace = simulate_episode(args.scenario, args.seed, policy, args.steps, capture_trace=True)
            write_csv(args.trajectory_csv, trace)
            print(json.dumps(metrics, indent=2))
            print(f"Trajectory written to {args.trajectory_csv}")
        else:
            results = evaluate_policies(policy)
            summary = summarize_evaluation(results)
            write_csv(args.output_dir / "evaluation_episodes.csv", results)
            write_csv(args.output_dir / "evaluation_summary.csv", summary)
            _print_summary(summary)
        return

    _, training_log, summary = run_training(args.episodes, args.seed, args.output_dir, args.steps)
    _print_summary(summary)
    print(f"\nTraining episodes: {len(training_log)}")
    print(f"Artifacts saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
