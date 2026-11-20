"""Reproducible health-aware EMS experiments for the hybrid-truck prototype.

The environment is a transparent, low-order longitudinal/powertrain model. It is
suitable for PC experiments and controller development, not vehicle certification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .agents import build_realistic_drive_cycle


OBSERVATION_BINS: tuple[tuple[float, ...], ...] = (
    (40.0, 65.0),  # SoC
    (85.0, 92.0, 97.0, 99.0),  # SoH; expose battery aging state to RL
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
EVALUATION_POLICIES = (
    "engine_only_regen", "rule_based", "trained_q_learning",
    "health_aware_mpc", "health_priority_mpc",
)


@dataclass(frozen=True)
class PlantConfig:
    """Assumed P4 heavy-truck parameters; calibrate against measured Sisu data."""

    timestep_s: float = 1.0
    vehicle_mass_kg: float = 32_000.0
    rotating_mass_factor: float = 1.05
    wheel_radius_m: float = 0.50
    engine_axle_position: str = "front (assumed)"
    electric_axle_position: str = "rear (user-specified)"
    rolling_resistance_coefficient: float = 0.007
    drag_coefficient: float = 0.62
    frontal_area_m2: float = 9.0
    air_density_kg_m3: float = 1.20
    gravity_m_s2: float = 9.81
    engine_axle_efficiency: float = 0.94
    electric_axle_efficiency: float = 0.93
    battery_capacity_kwh: float = 120.0
    motor_max_power_kw: float = 160.0
    engine_max_power_kw: float = 360.0
    max_regen_power_kw: float = 120.0
    ambient_temp_c: float = 20.0
    initial_soc_pct: float = 60.0
    initial_soh_pct: float = 96.0
    reference_engine_efficiency: float = 0.38
    diesel_energy_kwh_per_l: float = 9.7
    maximum_acceleration_m_s2: float = 0.65
    maximum_service_deceleration_m_s2: float = 1.20
    maximum_cycle_acceleration_m_s2: float = 0.30
    maximum_cycle_deceleration_m_s2: float = 0.60
    speed_tracking_gain_per_s: float = 0.45
    low_speed_cutoff_m_s: float = 0.25
    # Illustrative cycle-fade coefficient only; replace with supplier/cell data.
    assumed_soh_loss_pct_per_efc: float = 0.01
    assumed_reference_c_rate: float = 0.5
    # Calendar-fade coefficient is an explicit prototype assumption. It must be
    # identified from cell/pack aging data before it is used for life prediction.
    assumed_calendar_soh_loss_pct_per_day: float = 0.005
    # Explicit shadow price; sensitivity analysis is required for publication.
    soh_fade_reward_weight: float = 25_000.0


@dataclass
class EpisodeState:
    soc_pct: float
    battery_temp_c: float
    soh_pct: float
    degradation_proxy: float
    speed_kph: float = 0.0
    fuel_l: float = 0.0
    electric_discharge_kwh: float = 0.0
    recovered_energy_kwh: float = 0.0
    battery_throughput_kwh: float = 0.0
    distance_km: float = 0.0
    cumulative_reward: float = 0.0
    peak_battery_temp_c: float = 0.0
    time_above_35_c_s: float = 0.0
    calendar_aging_proxy: float = 0.0


@dataclass(frozen=True)
class DrivePoint:
    speed_kph: float
    grade_pct: float
    demand_kw: float
    brake_pedal: float
    accelerator_pedal: float
    ambient_temp_c: float
    target_acceleration_m_s2: float = 0.0


def copy_episode_state(state: EpisodeState) -> EpisodeState:
    """Copy mutable plant state for isolated MPC rollout branches."""
    return EpisodeState(
        soc_pct=state.soc_pct,
        battery_temp_c=state.battery_temp_c,
        soh_pct=state.soh_pct,
        degradation_proxy=state.degradation_proxy,
        speed_kph=state.speed_kph,
        fuel_l=state.fuel_l,
        electric_discharge_kwh=state.electric_discharge_kwh,
        recovered_energy_kwh=state.recovered_energy_kwh,
        battery_throughput_kwh=state.battery_throughput_kwh,
        distance_km=state.distance_km,
        cumulative_reward=state.cumulative_reward,
        peak_battery_temp_c=state.peak_battery_temp_c,
        time_above_35_c_s=state.time_above_35_c_s,
        calendar_aging_proxy=state.calendar_aging_proxy,
    )


def encode_observation(
    soc_pct: float,
    soh_pct: float,
    power_request_kw: float,
    speed_kph: float,
    grade_pct: float,
    battery_temp_c: float,
    braking: bool,
) -> int:
    """Discretize the measured state into a reproducible Q-table index."""
    values = (soc_pct, soh_pct, power_request_kw, speed_kph, grade_pct, battery_temp_c, float(braking))
    bins = tuple(int(np.digitize(value, edges)) for value, edges in zip(values, OBSERVATION_BINS))
    return int(np.ravel_multi_index(bins, OBSERVATION_SHAPE))


def _observation_power(point: DrivePoint) -> float:
    return point.demand_kw if point.brake_pedal <= 0.0 else point.speed_kph * point.brake_pedal


def _encode_point(state: EpisodeState, point: DrivePoint) -> int:
    return encode_observation(
        state.soc_pct,
        state.soh_pct,
        _observation_power(point),
        state.speed_kph,
        point.grade_pct,
        state.battery_temp_c,
        point.brake_pedal > 0.0,
    )


@lru_cache(maxsize=1)
def _base_mixed_cycle() -> list[dict[str, float | str]]:
    return build_realistic_drive_cycle()


def _mixed_point(index: int, rng: np.random.Generator, offset: int) -> tuple[float, float, float, float, float]:
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


def road_load_force_n(speed_kph: float, grade_pct: float, config: PlantConfig) -> float:
    """Calculate forward rolling, aerodynamic, and grade loads at the wheels."""
    speed_m_s = max(0.0, speed_kph / 3.6)
    angle = math.atan(grade_pct / 100.0)
    normal_force = config.vehicle_mass_kg * config.gravity_m_s2 * math.cos(angle)
    rolling = config.rolling_resistance_coefficient * normal_force
    aerodynamic = 0.5 * config.air_density_kg_m3 * config.drag_coefficient * config.frontal_area_m2 * speed_m_s**2
    grade_force = config.vehicle_mass_kg * config.gravity_m_s2 * math.sin(angle)
    return rolling + aerodynamic + grade_force


def _smooth_reference_speeds(points: list[DrivePoint], config: PlantConfig) -> list[DrivePoint]:
    """Filter speed noise and limit reference acceleration to plausible values."""
    if len(points) < 2:
        return points
    result = [points[0]]
    max_up = config.maximum_cycle_acceleration_m_s2 * config.timestep_s * 3.6
    max_down = config.maximum_cycle_deceleration_m_s2 * config.timestep_s * 3.6
    for point in points[1:]:
        previous_speed = result[-1].speed_kph
        filtered_delta = 0.30 * (point.speed_kph - previous_speed)
        speed = previous_speed + float(np.clip(filtered_delta, -max_down, max_up))
        result.append(replace(point, speed_kph=float(np.clip(speed, 0.0, 100.0))))
    return result


def _derive_physical_cycle(points: list[DrivePoint], config: PlantConfig) -> list[DrivePoint]:
    """Derive acceleration, brake request, and wheel-power demand from route data."""
    cycles: list[DrivePoint] = []
    for index, point in enumerate(points):
        if index + 1 < len(points):
            target_acceleration = (points[index + 1].speed_kph - point.speed_kph) / 3.6 / config.timestep_s
        elif index > 0:
            target_acceleration = (point.speed_kph - points[index - 1].speed_kph) / 3.6 / config.timestep_s
        else:
            target_acceleration = 0.0
        target_acceleration = float(np.clip(target_acceleration, -config.maximum_cycle_deceleration_m_s2, config.maximum_cycle_acceleration_m_s2))
        if target_acceleration < -0.1:
            brake = max(point.brake_pedal, float(np.clip(-target_acceleration / config.maximum_service_deceleration_m_s2, 0.0, 1.0)))
            point = replace(point, brake_pedal=brake)
        if point.brake_pedal > 0.0:
            target_acceleration = min(target_acceleration, -max(0.25, point.brake_pedal * 1.2))
        elif target_acceleration > 0.0:
            target_acceleration = min(target_acceleration, point.accelerator_pedal * config.maximum_acceleration_m_s2)
        wheel_force = max(0.0, config.vehicle_mass_kg * config.rotating_mass_factor * target_acceleration + road_load_force_n(point.speed_kph, point.grade_pct, config))
        demand_kw = wheel_force * (point.speed_kph / 3.6) / 1000.0
        cycles.append(replace(point, demand_kw=float(demand_kw), target_acceleration_m_s2=target_acceleration))
    return cycles


def make_drive_cycle(scenario: str, rng: np.random.Generator, steps: int = 120) -> list[DrivePoint]:
    """Generate a smooth speed/grade route and calculate wheel demand from physics."""
    builders = {"city_stop_go": _city_point, "highway_cruise": _highway_point, "hilly_grade": _hilly_point}
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}.")
    config = PlantConfig()
    mixed_offset = int(rng.integers(0, 120))
    ambient_base = float(rng.uniform(5.0, 32.0))
    points: list[DrivePoint] = []
    for index in range(steps):
        phase = 2.0 * math.pi * index / max(steps, 1)
        values = _mixed_point(index, rng, mixed_offset) if scenario == "mixed_route" else builders[scenario](index, phase, rng)
        speed, grade, accelerator, brake, _ = values
        points.append(DrivePoint(
            speed_kph=float(np.clip(speed, 0.0, 100.0)),
            grade_pct=float(np.clip(grade, -8.0, 10.0)),
            demand_kw=0.0,
            brake_pedal=float(np.clip(brake, 0.0, 1.0)),
            accelerator_pedal=float(np.clip(accelerator, 0.0, 1.0)),
            ambient_temp_c=float(np.clip(ambient_base + 2.0 * math.sin(phase / 4.0), -20.0, 40.0)),
        ))
    return _derive_physical_cycle(_smooth_reference_speeds(points, config), config)


def battery_power_limits(state: EpisodeState, config: PlantConfig) -> tuple[float, float]:
    """Return SoC-, temperature-, and SoH-dependent battery discharge/charge caps."""
    discharge_soc_factor = float(np.clip((state.soc_pct - 20.0) / 20.0, 0.0, 1.0))
    charge_soc_factor = float(np.clip((90.0 - state.soc_pct) / 15.0, 0.0, 1.0))
    temperature_factor = float(np.clip(1.0 - max(0.0, state.battery_temp_c - 30.0) / 35.0, 0.35, 1.0))
    health_factor = float(np.clip(state.soh_pct / 100.0, 0.7, 1.0))
    return (
        config.motor_max_power_kw * discharge_soc_factor * temperature_factor * health_factor,
        config.max_regen_power_kw * charge_soc_factor * temperature_factor * health_factor,
    )


def transition(
    state: EpisodeState,
    point: DrivePoint,
    action: int,
    config: PlantConfig,
    health_cost_multiplier: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Advance P4 longitudinal dynamics with distinct engine and electric axles."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"Action must be in [0, {N_ACTIONS - 1}], got {action}.")
    dt_s = config.timestep_s
    previous_soc = state.soc_pct
    speed = max(0.0, state.speed_kph / 3.6)
    target_speed = point.speed_kph / 3.6
    mass = config.vehicle_mass_kg * config.rotating_mass_factor
    road_force = road_load_force_n(state.speed_kph, point.grade_pct, config)
    acceleration_request = point.target_acceleration_m_s2 + config.speed_tracking_gain_per_s * (target_speed - speed)
    if point.brake_pedal > 0.0:
        acceleration_request = min(acceleration_request, -max(0.25, point.brake_pedal * 1.2))
    else:
        acceleration_request = min(acceleration_request, point.accelerator_pedal * config.maximum_acceleration_m_s2)
    acceleration_request = float(np.clip(acceleration_request, -config.maximum_service_deceleration_m_s2, config.maximum_acceleration_m_s2))
    braking = point.brake_pedal > 0.0 or acceleration_request < -0.03
    discharge_limit, charge_limit = battery_power_limits(state, config)

    engine_force = motor_force = regen_force = friction_force = 0.0
    engine_shaft_kw = motor_wheel_kw = regen_battery_kw = 0.0
    battery_energy_kwh = friction_brake_kwh = unmet_force = fuel_l = 0.0
    electric_gear = action % 2 + 1
    preferred_electric_gear = 1 if state.speed_kph < 55.0 else 2
    electric_path_efficiency = config.electric_axle_efficiency * (
        0.98 if electric_gear == preferred_electric_gear else 0.94
    )

    if braking:
        regen_fraction = REGEN_FRACTIONS[action // 2]
        brake_force_request = max(0.0, -mass * acceleration_request - road_force)
        regen_power_cap = min(config.max_regen_power_kw, charge_limit / electric_path_efficiency)
        regen_force_cap = regen_power_cap * 1000.0 / max(speed, config.low_speed_cutoff_m_s)
        regen_force = min(brake_force_request * regen_fraction, regen_force_cap)
        friction_force = max(0.0, brake_force_request - regen_force)
        actual_brake_acceleration = -(regen_force + friction_force + road_force) / mass
        mean_speed = max(0.0, speed + 0.5 * actual_brake_acceleration * dt_s)
        regen_wheel_kw = regen_force * mean_speed / 1000.0
        regen_battery_kw = min(regen_wheel_kw * electric_path_efficiency, charge_limit)
        battery_energy_kwh = regen_battery_kw * dt_s / 3600.0
        state.soc_pct += battery_energy_kwh / config.battery_capacity_kwh * 100.0
        state.recovered_energy_kwh += battery_energy_kwh
    else:
        assist = ASSIST_FRACTIONS[action // 2]
        wheel_force_request = max(0.0, mass * acceleration_request + road_force)
        motor_force_cap = min(config.motor_max_power_kw * 1000.0, discharge_limit * electric_path_efficiency * 1000.0) / max(speed, config.low_speed_cutoff_m_s)
        motor_force = min(wheel_force_request * assist, motor_force_cap)
        engine_force_cap = config.engine_max_power_kw * config.engine_axle_efficiency * 1000.0 / max(speed, config.low_speed_cutoff_m_s)
        engine_force = min(max(0.0, wheel_force_request - motor_force), engine_force_cap)
        unmet_force = max(0.0, wheel_force_request - engine_force - motor_force)
        drive_acceleration = (engine_force + motor_force - road_force) / mass
        mean_speed = max(0.0, speed + 0.5 * drive_acceleration * dt_s)
        motor_wheel_kw = motor_force * mean_speed / 1000.0
        motor_battery_kw = motor_wheel_kw / electric_path_efficiency
        engine_wheel_kw = engine_force * mean_speed / 1000.0
        engine_shaft_kw = engine_wheel_kw / config.engine_axle_efficiency
        battery_energy_kwh = motor_battery_kw * dt_s / 3600.0
        state.soc_pct -= battery_energy_kwh / config.battery_capacity_kwh * 100.0
        state.electric_discharge_kwh += battery_energy_kwh
        if engine_shaft_kw > 0.0:
            engine_efficiency = 0.30 + 0.09 * math.exp(-((engine_shaft_kw - 160.0) / 100.0) ** 2)
            engine_efficiency -= 0.025 * max(0.0, engine_shaft_kw - 280.0) / 80.0
            engine_efficiency = float(np.clip(engine_efficiency, 0.25, 0.42))
            fuel_l = engine_shaft_kw * dt_s / (3600.0 * engine_efficiency * config.diesel_energy_kwh_per_l)
            state.fuel_l += fuel_l

    net_force = engine_force + motor_force - regen_force - friction_force - road_force
    actual_acceleration = net_force / mass
    next_speed = max(0.0, speed + actual_acceleration * dt_s)
    if next_speed <= 0.0 and dt_s > 0.0:
        actual_acceleration = (next_speed - speed) / dt_s
    state.speed_kph = next_speed * 3.6
    mean_speed = 0.5 * (speed + next_speed)
    state.distance_km += mean_speed * dt_s / 1000.0
    if braking:
        regen_wheel_kw = regen_force * mean_speed / 1000.0
        battery_loss_kw = max(0.0, regen_wheel_kw - regen_battery_kw)
    else:
        battery_loss_kw = max(0.0, motor_battery_kw - motor_wheel_kw)
    dt_h = dt_s / 3600.0
    state.battery_temp_c += dt_h * (0.025 * battery_loss_kw - 0.012 * (state.battery_temp_c - point.ambient_temp_c))
    state.peak_battery_temp_c = max(state.peak_battery_temp_c, state.battery_temp_c)
    if state.battery_temp_c > 35.0:
        state.time_above_35_c_s += dt_s
    c_rate = (battery_energy_kwh * 3600.0 / dt_s) / config.battery_capacity_kwh if dt_s > 0.0 else 0.0
    soc_stress = 1.0 + max(0.0, 35.0 - state.soc_pct) / 30.0 + max(0.0, state.soc_pct - 70.0) / 20.0
    temperature_stress = 2.0 ** (max(0.0, state.battery_temp_c - 25.0) / 10.0)
    rate_stress = 1.0 + max(0.0, c_rate - config.assumed_reference_c_rate)
    stress = soc_stress * temperature_stress * rate_stress
    state.battery_throughput_kwh += battery_energy_kwh
    degradation_increment = (battery_energy_kwh / (2.0 * config.battery_capacity_kwh)) * config.assumed_soh_loss_pct_per_efc * stress
    # A separate calendar term prevents the model from treating an idle,
    # high-temperature battery as aging-free. Both terms are proxies, not a
    # chemistry model or a remaining-useful-life estimate.
    soc_calendar_stress = 1.0 + max(0.0, state.soc_pct - 60.0) / 40.0
    calendar_increment = (
        config.assumed_calendar_soh_loss_pct_per_day
        * dt_s / 86_400.0
        * 2.0 ** (max(0.0, state.battery_temp_c - 25.0) / 10.0)
        * soc_calendar_stress
    )
    total_degradation_increment = degradation_increment + calendar_increment
    state.degradation_proxy += total_degradation_increment
    state.calendar_aging_proxy += calendar_increment
    state.soh_pct = max(70.0, state.soh_pct - total_degradation_increment)
    state.soc_pct = float(np.clip(state.soc_pct, 10.0, 95.0))

    battery_delta_kwh = (previous_soc - state.soc_pct) / 100.0 * config.battery_capacity_kwh
    battery_equivalent_l = battery_delta_kwh / (config.reference_engine_efficiency * config.diesel_energy_kwh_per_l)
    cycle_cost = 2.0 * battery_energy_kwh * stress
    fade_cost = config.soh_fade_reward_weight * health_cost_multiplier * total_degradation_increment
    friction_brake_kwh = friction_force * mean_speed * dt_s / 3.6e6
    tracking_cost = 0.03 * abs(point.speed_kph - state.speed_kph)
    unmet_power_kw = unmet_force * mean_speed / 1000.0
    reward = -(100.0 * (fuel_l + battery_equivalent_l) + cycle_cost + fade_cost + 1.5 * friction_brake_kwh
              + 0.02 * max(0.0, state.battery_temp_c - 38.0) ** 2 + tracking_cost + 0.05 * unmet_power_kw)
    engine_wheel_kw = engine_force * mean_speed / 1000.0
    regen_wheel_kw = regen_force * mean_speed / 1000.0
    metrics = {
        "speed_kph": state.speed_kph, "target_speed_kph": point.speed_kph,
        "acceleration_m_s2": actual_acceleration, "target_acceleration_m_s2": acceleration_request,
        "road_load_force_n": road_force, "engine_axle_force_n": engine_force,
        "electric_axle_force_n": motor_force, "engine_power_kw": engine_shaft_kw,
        "engine_wheel_power_kw": engine_wheel_kw, "motor_power_kw": motor_wheel_kw,
        "regen_power_kw": regen_battery_kw, "regen_wheel_power_kw": regen_wheel_kw,
        "friction_brake_power_kw": friction_force * mean_speed / 1000.0,
        "friction_brake_kwh": friction_brake_kwh, "requested_wheel_power_kw": point.demand_kw,
        "unmet_power_kw": unmet_power_kw, "fuel_l": fuel_l,
        "battery_equivalent_fuel_l": battery_equivalent_l,
        "battery_throughput_kwh": battery_energy_kwh, "electric_gear": float(electric_gear),
        "electric_path_efficiency": electric_path_efficiency,
        "soh_loss_percentage_points": total_degradation_increment,
        "calendar_soh_loss_percentage_points": calendar_increment,
        "cycle_soh_loss_percentage_points": degradation_increment,
        "equivalent_full_cycles_increment": battery_energy_kwh / (2.0 * config.battery_capacity_kwh),
        "cycle_aging_stress_factor": stress,
        "cycle_fade_reward_cost": fade_cost,
        "battery_temperature_c": state.battery_temp_c,
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


class HealthAwareMPCPolicy:
    """Short-horizon beam-search MPC using the same P4 plant and health reward.

    The finite action set is searched over a short preview of the known drive
    cycle. Only the first action is applied; optimization is repeated next step.
    """

    def __init__(
        self,
        horizon_steps: int = 5,
        beam_width: int = 8,
        discount: float = 0.98,
        health_cost_multiplier: float = 1.0,
        terminal_soc_weight: float = 2.0,
        terminal_temperature_weight: float = 2.0,
    ) -> None:
        if (
            horizon_steps < 1
            or beam_width < 1
            or health_cost_multiplier < 0.0
            or terminal_soc_weight < 0.0
            or terminal_temperature_weight < 0.0
        ):
            raise ValueError("MPC horizon/beam must be positive and cost weights nonnegative")
        self.horizon_steps = horizon_steps
        self.beam_width = beam_width
        self.discount = discount
        self.health_cost_multiplier = health_cost_multiplier
        self.terminal_soc_weight = terminal_soc_weight
        self.terminal_temperature_weight = terminal_temperature_weight

    def select_action(
        self,
        state: EpisodeState,
        cycle: list[DrivePoint],
        index: int,
        config: PlantConfig,
    ) -> int:
        """Return the first action of the highest-return health-aware rollout."""
        beam: list[tuple[float, EpisodeState, int]] = [(0.0, copy_episode_state(state), 0)]
        horizon = min(self.horizon_steps, len(cycle) - index)
        for depth in range(horizon):
            point: DrivePoint = cycle[index + depth]
            expanded: list[tuple[float, EpisodeState, int]] = []
            for accumulated_return, predicted_state, first_action in beam:
                for action_index in range(N_ACTIONS):
                    action: int = int(action_index)
                    candidate_state = copy_episode_state(predicted_state)
                    step_reward, _ = transition(
                        candidate_state, point, action, config,
                        health_cost_multiplier=self.health_cost_multiplier,
                    )
                    chosen_first = action if depth == 0 else first_action
                    expanded.append((
                        accumulated_return + (self.discount**depth) * step_reward,
                        candidate_state,
                        chosen_first,
                    ))
            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = expanded[: self.beam_width]

        if not beam:
            return 0
        # A finite horizon otherwise rewards emptying the battery just outside
        # the preview. The terminal terms make the controller charge-sustaining
        # and discourage ending a rollout hot, without pretending to know the
        # complete future mission.
        def terminal_value(item: tuple[float, EpisodeState, int]) -> float:
            accumulated_return, predicted_state, _ = item
            soc_cost = self.terminal_soc_weight * abs(predicted_state.soc_pct - state.soc_pct)
            temperature_cost = self.terminal_temperature_weight * max(
                0.0, predicted_state.battery_temp_c - 35.0
            ) ** 2
            return accumulated_return - soc_cost - temperature_cost

        return int(max(beam, key=terminal_value)[2])


def train_q_learning(
    episodes: int = 10000,
    seed: int = 2026,
    steps_per_episode: int = 120,
    alpha: float = 0.16,
    gamma: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.04,
) -> tuple[TabularQPolicy, list[dict[str, float | int | str]]]:
    """Train a Double Q-learning EMS with randomized missions and health states.

    Two independent estimators reduce the optimistic max bias of ordinary
    Q-learning (van Hasselt et al., 2016). The exported policy is their mean,
    retaining the existing portable CSV format.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = np.random.default_rng(seed)
    action_rng = random.Random(seed)
    q_a = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    q_b = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
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
            speed_kph=cycle[0].speed_kph,
        )
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-5.0 * episode / max(episodes - 1, 1))
        total_reward = 0.0

        for step, point in enumerate(cycle):
            observation = _encode_point(state, point)
            if action_rng.random() < epsilon:
                action = action_rng.randrange(N_ACTIONS)
            else:
                action = int(np.argmax(q_a[observation] + q_b[observation]))
            reward, _ = transition(state, point, action, config)
            total_reward += reward
            done = step == len(cycle) - 1
            if done:
                next_max = 0.0
            else:
                next_point = cycle[step + 1]
                next_observation = _encode_point(state, next_point)
                if action_rng.random() < 0.5:
                    next_action = int(np.argmax(q_a[next_observation]))
                    next_max = float(q_b[next_observation, next_action])
                else:
                    next_action = int(np.argmax(q_b[next_observation]))
                    next_max = float(q_a[next_observation, next_action])
            if action_rng.random() < 0.5:
                q_a[observation, action] += alpha * (
                    reward + gamma * next_max - q_a[observation, action]
                )
            else:
                q_b[observation, action] += alpha * (
                    reward + gamma * next_max - q_b[observation, action]
                )

        logs.append({
            "episode": episode + 1,
            "scenario": scenario,
            "epsilon": epsilon,
            "return": total_reward,
            "algorithm": "double_q_learning",
        })

    return TabularQPolicy((q_a + q_b) / 2.0), logs


def _fixed_policy_action(name: str, state: EpisodeState, point: DrivePoint) -> int:
    if point.brake_pedal > 0.0:
        return 7  # maximum safe regen request, electric axle gear 2
    if name == "engine_only_regen":
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
    policy: TabularQPolicy | HealthAwareMPCPolicy | str,
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
        speed_kph=cycle[0].speed_kph,
    )
    initial_soh_pct = state.soh_pct
    trace: list[dict[str, float]] = []
    speed_tracking_error_sum = 0.0
    friction_brake_energy_kwh = 0.0
    unmet_traction_energy_kwh = 0.0
    for index, point in enumerate(cycle):
        if isinstance(policy, HealthAwareMPCPolicy):
            action = policy.select_action(state, cycle, index, config)
        elif isinstance(policy, TabularQPolicy):
            action = policy.select_action(state, point)
        else:
            action = _fixed_policy_action(policy, state, point)
        reward, control = transition(state, point, action, config)
        speed_tracking_error_sum += abs(point.speed_kph - state.speed_kph)
        friction_brake_energy_kwh += control["friction_brake_kwh"]
        unmet_traction_energy_kwh += control["unmet_power_kw"] * config.timestep_s / 3600.0
        if capture_trace:
            trace.append(
                {
                    "time_s": float(index * config.timestep_s),
                    "speed_kph": control["speed_kph"],
                    "target_speed_kph": point.speed_kph,
                    "grade_pct": point.grade_pct,
                    "demand_kw": point.demand_kw,
                    "accelerator_pedal": point.accelerator_pedal,
                    "brake_pedal": point.brake_pedal,
                    "engine_power_kw": control["engine_power_kw"],
                    "engine_wheel_power_kw": control["engine_wheel_power_kw"],
                    "motor_power_kw": control["motor_power_kw"],
                    "regen_power_kw": control["regen_power_kw"],
                    "regen_wheel_power_kw": control["regen_wheel_power_kw"],
                    "friction_brake_power_kw": control["friction_brake_power_kw"],
                    "road_load_force_n": control["road_load_force_n"],
                    "electric_gear": control["electric_gear"],
                    "soc_pct": state.soc_pct,
                    "soh_pct": state.soh_pct,
                    "equivalent_full_cycles": state.battery_throughput_kwh / (2.0 * config.battery_capacity_kwh),
                    "battery_temp_c": state.battery_temp_c,
                    "peak_battery_temp_c": state.peak_battery_temp_c,
                    "time_above_35_c_s": state.time_above_35_c_s,
                    "calendar_aging_proxy": state.calendar_aging_proxy,
                    "cumulative_fuel_l": state.fuel_l,
                    "step_reward": reward,
                }
            )

    score = state.cumulative_reward
    distance = max(state.distance_km, 1.0e-9)
    equivalent_fuel_l = state.fuel_l + (initial_soc_pct - state.soc_pct) / 100.0 * config.battery_capacity_kwh / (
        config.reference_engine_efficiency * config.diesel_energy_kwh_per_l
    )
    if isinstance(policy, TabularQPolicy):
        policy_name = "trained_q_learning"
    elif isinstance(policy, HealthAwareMPCPolicy):
        policy_name = "health_aware_mpc" if policy.health_cost_multiplier == 1.0 else "health_priority_mpc"
    else:
        policy_name = policy
    metrics: dict[str, Any] = {
        "scenario": scenario,
        "policy": policy_name,
        "seed": seed,
        "distance_km": state.distance_km,
        "mean_speed_tracking_error_kph": speed_tracking_error_sum / max(len(cycle), 1),
        "friction_brake_energy_kwh": friction_brake_energy_kwh,
        "unmet_traction_energy_kwh": unmet_traction_energy_kwh,
        "fuel_l": state.fuel_l,
        "fuel_l_per_100km": state.fuel_l / distance * 100.0,
        "charge_sustaining_equivalent_fuel_l": equivalent_fuel_l,
        "charge_sustaining_equivalent_l_per_100km": equivalent_fuel_l / distance * 100.0,
        "initial_soc_pct": initial_soc_pct,
        "final_soc_pct": state.soc_pct,
        "soc_change_pct": state.soc_pct - initial_soc_pct,
        "battery_discharge_kwh": state.electric_discharge_kwh,
        "battery_throughput_kwh": state.battery_throughput_kwh,
        "equivalent_full_cycles": state.battery_throughput_kwh / (2.0 * config.battery_capacity_kwh),
        "initial_soh_pct": initial_soh_pct,
        "final_soh_pct": state.soh_pct,
        "soh_loss_percentage_points": initial_soh_pct - state.soh_pct,
        "cycle_aging_proxy": state.degradation_proxy,
        "regen_recovered_kwh": state.recovered_energy_kwh,
        "battery_degradation_proxy": state.degradation_proxy,
        "calendar_aging_proxy": state.calendar_aging_proxy,
        "peak_battery_temp_c": state.peak_battery_temp_c,
        "time_above_35_c_s": state.time_above_35_c_s,
        "final_battery_temp_c": state.battery_temp_c,
        "episode_return": score,
    }
    return metrics, trace


def evaluate_policies(
    policy: TabularQPolicy,
    seeds: tuple[int, ...] = tuple(range(4101, 4111)),
    steps: int = 120,
) -> list[dict[str, Any]]:
    """Compare Q learning and health-aware MPC to engine-only and rule baselines."""
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in seeds:
            eval_seed = seed + scenario_index * 100
            for candidate in (
                "engine_only_regen", "rule_based", policy,
                HealthAwareMPCPolicy(),
                HealthAwareMPCPolicy(health_cost_multiplier=25.0),
            ):
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    episodes = np.asarray([int(row["episode"]) for row in training_log])
    returns = np.asarray([float(row["return"]) for row in training_log])
    smooth_window = max(1, min(500, len(returns) // 10))
    smooth = np.convolve(returns, np.ones(smooth_window) / smooth_window, mode="valid")
    axes[0].plot(episodes[: len(smooth)], smooth, color="#126782", linewidth=1.8)
    axes[0].set(title="Training progress (moving-average return)", xlabel="Episode", ylabel="Return")
    axes[0].grid(alpha=0.25)

    policies = EVALUATION_POLICIES
    colors = ("#8b9aa7", "#e09f3e", "#137c8b", "#805ad5", "#c05621")
    scenarios = list(SCENARIOS)
    x = np.arange(len(scenarios))
    width = 0.15
    for offset, (name, color) in enumerate(zip(policies, colors)):
        means = [
            float(np.mean([row["charge_sustaining_equivalent_l_per_100km"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name]))
            for scenario in scenarios
        ]
        stds = [
            float(np.std([row["charge_sustaining_equivalent_l_per_100km"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name], ddof=1))
            for scenario in scenarios
        ]
        bar_x = x + (offset - (len(policies) - 1) / 2.0) * width
        axes[1].bar(bar_x, means, width, yerr=stds, capsize=2, label=name.replace("_", " "), color=color)

        mean_efc = [
            float(np.mean([row["equivalent_full_cycles"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name]))
            for scenario in scenarios
        ]
        axes[2].bar(bar_x, mean_efc, width, label=name.replace("_", " "), color=color)
        mean_soh_loss = [
            float(np.mean([row["soh_loss_percentage_points"] for row in evaluation if row["scenario"] == scenario and row["policy"] == name]))
            for scenario in scenarios
        ]
        axes[3].bar(bar_x, mean_soh_loss, width, label=name.replace("_", " "), color=color)

    axes[1].set(title="Charge-sustaining equivalent", ylabel="Equivalent diesel (L/100 km)", xticks=x, xticklabels=[s.replace("_", "\n") for s in scenarios])
    axes[2].set(title="Battery throughput", ylabel="Equivalent full cycles per route", xticks=x, xticklabels=[s.replace("_", "\n") for s in scenarios])
    axes[3].set(title="Cycle-aging proxy", ylabel="SoH loss (percentage points / route)", xticks=x, xticklabels=[s.replace("_", "\n") for s in scenarios])
    for axis in axes[1:]:
        axis.legend(frameon=False, fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Rear-e-axle P4 EMS: Q-learning vs predictive health-aware control", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.parent / "benchmark_dashboard.png", dpi=170, bbox_inches="tight")
    workspace_root = Path(__file__).resolve().parents[2]
    if path.parent.resolve() == (workspace_root / "artifacts").resolve():
        fig.savefig(workspace_root / "benchmark_dashboard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_vehicle_trajectory_plot(policy: TabularQPolicy, path: Path, seed: int = 4101) -> None:
    """Plot the actual speed and separate axle/brake powers from one held-out P4 route."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, trace = simulate_episode("mixed_route", seed, policy, steps=120, capture_trace=True)
    time_s = np.asarray([row["time_s"] for row in trace])
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(time_s, [row["target_speed_kph"] for row in trace], "--", label="Target speed")
    axes[0].plot(time_s, [row["speed_kph"] for row in trace], label="Simulated vehicle speed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_s, [row["engine_wheel_power_kw"] for row in trace], label="Front engine axle (assumed)")
    axes[1].plot(time_s, [row["motor_power_kw"] for row in trace], label="Rear electric axle")
    axes[1].plot(time_s, [-row["regen_wheel_power_kw"] for row in trace], label="Rear e-axle regen")
    axes[1].plot(time_s, [-row["friction_brake_power_kw"] for row in trace], label="Friction brakes")
    axes[1].set_ylabel("Wheel power (kW)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_s, [row["road_load_force_n"] / 1000.0 for row in trace], color="#805ad5")
    axes[2].set_ylabel("Road load (kN)")
    axes[2].grid(alpha=0.25)

    axes[3].plot(time_s, [row["soc_pct"] for row in trace], label="Battery SoC")
    axes[3].set_ylabel("SoC (%)")
    axes[3].grid(alpha=0.25)
    axes[4].plot(time_s, [row["soh_pct"] for row in trace], label="SoH proxy")
    axes[4].set_ylabel("SoH proxy (%)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(alpha=0.25)
    fig.suptitle("Assumed P4 truck: front engine axle + rear electric axle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.parent.parent / "performance_dashboard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for policy in EVALUATION_POLICIES:
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
                    "mean_battery_throughput_kwh": float(np.mean([row["battery_throughput_kwh"] for row in subset])),
                    "mean_equivalent_full_cycles": float(np.mean([row["equivalent_full_cycles"] for row in subset])),
                    "mean_soh_loss_percentage_points": float(np.mean([row["soh_loss_percentage_points"] for row in subset])),
                    "mean_calendar_aging_proxy": float(np.mean([row["calendar_aging_proxy"] for row in subset])),
                    "mean_peak_battery_temp_c": float(np.mean([row["peak_battery_temp_c"] for row in subset])),
                    "mean_time_above_35_c_s": float(np.mean([row["time_above_35_c_s"] for row in subset])),
                    "mean_tracking_error_kph": float(np.mean([row["mean_speed_tracking_error_kph"] for row in subset])),
                    "mean_friction_brake_energy_kwh": float(np.mean([row["friction_brake_energy_kwh"] for row in subset])),
                    "mean_unmet_traction_energy_kwh": float(np.mean([row["unmet_traction_energy_kwh"] for row in subset])),
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
    rollout_metrics, rollout_trace = simulate_episode("mixed_route", 4101, policy, steps=120, capture_trace=True)
    write_csv(root / "p4_mixed_route_trace.csv", rollout_trace)
    save_vehicle_trajectory_plot(policy, root / "p4_vehicle_trajectory.png", seed=4101)
    metadata = {
        "algorithm": "Double Q-learning (epsilon-greedy; mean of two independent estimators)",
        "additional_controller": "health-aware receding-horizon beam-search MPC with terminal SoC/temperature costs; 5-step preview, beam width 8; tested on evaluation only, not learned",
        "episodes": episodes,
        "seed": seed,
        "steps_per_episode": steps_per_episode,
        "observation_bins": OBSERVATION_BINS,
        "observation_shape": OBSERVATION_SHAPE,
        "action_count": N_ACTIONS,
        "propulsion_actions": {"assist_fraction": ASSIST_FRACTIONS, "electric_gear": [1, 2]},
        "braking_actions": {"regen_fraction": REGEN_FRACTIONS, "electric_gear": [1, 2]},
        "evaluated_controllers": [
            "engine-only propulsion plus rear-axle regen",
            "SOC-threshold rule-based control",
            "Double Q-learning (SoC, SoH, speed, demand, grade, temperature, braking observations)",
            "health-aware receding-horizon beam-search MPC (5 steps; beam width 8; terminal SoC/temperature costs; perfect short route preview)",
            "life-priority beam-search MPC (same horizon/preview; 25x health-fade shadow-price sensitivity case; not calibrated battery economics)",
        ],
        "plant_config": asdict(PlantConfig()),
        "architecture": "P4 parallel hybrid: rear axle driven by the independent electric motor; engine drives the other/front axle (front location remains an explicit assumption)",
        "reward_objective": "charge-sustaining fuel-equivalent consumption + throughput cost + illustrative EFC-based SoH-fade cost with SoC/temperature/C-rate stress + friction brake work + thermal violation + speed-tracking and unmet-traction penalties",
        "health_proxy": "equivalent full cycles = total charge-plus-discharge energy throughput / (2 * nominal capacity); cycle fade plus an explicit illustrative calendar-fade term with temperature and SoC stress; coefficients are not chemistry-specific and cannot estimate service life",
        "scientific_basis": [
            "Wang et al. (2018), Predictive energy management for hybrid electric vehicles considering extension of the battery life, Proc IMechE Part D 232(4), 499-510, doi:10.1177/0954407017703229",
            "Wu et al. (2021), Battery Thermal- and Health-Constrained Energy Management for Hybrid Electric Bus Based on Soft Actor-Critic DRL Algorithm, IEEE TII 17(6), 3751-3761, doi:10.1109/TII.2020.3014599",
            "Schmalstieg et al. (2014), A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries, Journal of Power Sources 257, 325-334, doi:10.1016/j.jpowsour.2014.02.012",
        ],
        "model_scope": "longitudinal force model with assumed generic heavy-truck properties; not calibrated to a specific Sisu model or validated for real-time vehicle control",
        "validation_rollout": rollout_metrics,
    }
    (root / "q_policy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return policy, training_log, summary


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("Scenario       Policy               CS-eq    SoC d   EFC     SoH d   Track err")
    print("                                   L/100km     (%)          (pp)       (km/h)")
    print("-" * 84)
    for row in rows:
        print(
            f"{row['scenario']:<14} {row['policy']:<20} "
            f"{row['mean_charge_sustaining_equivalent_l_per_100km']:>7.2f} "
            f"{row['mean_soc_change_pct']:>8.3f} "
            f"{row['mean_equivalent_full_cycles']:>6.4f} "
            f"{row['mean_soh_loss_percentage_points']:>8.5f} "
            f"{row['mean_tracking_error_kph']:>9.2f}"
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
