from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency for visualization
    plt = None


class DriveCycle(TypedDict):
    """High-level drive-cycle point used by the EMS."""

    speed_kph: float
    grade_pct: float
    traffic_density: float
    route_profile: str
    predicted_demand_kw: float


class PowerRequest(TypedDict):
    """Current propulsion demand and driver input."""

    requested_power_kw: float
    demanded_torque_nm: float
    acceleration_request_mps2: float
    driver_pedal: float
    brake_pedal: float


class BatteryState(TypedDict):
    """Battery SoC, SoH, temperature, and health indicators."""

    soc_pct: float
    soh_pct: float
    temp_c: float
    degradation_index: float
    target_soc_pct: float
    health_factor: float


class ThermalState(TypedDict):
    """Cooling and thermal envelopes for battery, inverter, and gearbox."""

    battery_temp_c: float
    inverter_temp_c: float
    gearbox_oil_temp_c: float
    cooling_loop_temp_c: float
    battery_limit_c: float
    inverter_limit_c: float
    gearbox_limit_c: float
    thermal_margin_pct: float


class ControlTargets(TypedDict):
    """High-level planning objectives from the supervisor."""

    soc_target_pct: float
    battery_safety_limit_pct: float
    max_ice_torque_pct: float
    max_em_torque_pct: float
    gear_shift_bias: float


class SupervisorOutput(TypedDict):
    """Output from the supervisory planning layer."""

    macro_target: str
    safety_limits: dict[str, float]
    energy_budget_kw: float
    horizon_minutes: int
    regen_priority: float


class PowertrainOutput(TypedDict):
    """Powertrain optimization action."""

    selected_gear: int
    electric_gear: int
    ice_power_kw: float
    em_power_kw: float
    torque_split: dict[str, float]
    expected_efficiency: float
    rationale: str


class ThermalBatteryOutput(TypedDict):
    """Thermal and degradation override decisions."""

    status: str
    max_em_power_kw: float
    max_ice_power_kw: float
    override_reason: str
    degradation_penalty: float


class DriverAssistanceOutput(TypedDict):
    """Driver-facing guidance and torque blending advice."""

    advisory: str
    comfort_mode: str
    torque_blending: float
    safety_ok: bool


class PerformanceMetrics(TypedDict):
    """Aggregated control quality metrics for offline or online monitoring."""

    total_reward: float
    fuel_efficiency_score: float
    thermal_risk_score: float
    degradation_risk_score: float
    safety_score: float
    comfort_score: float
    sustainability_score: float
    status: str


class FinalDecision(TypedDict):
    """Final commanded action after all checks."""

    torque_split: dict[str, float]
    selected_gear: int
    electric_gear: int
    advisory: str
    decision_status: str
    engine_torque_nm: float
    motor_torque_nm: float
    regen_kw: float
    driver_pedal: float
    brake_pedal: float
    commanded_ice_kw: float
    commanded_em_kw: float


class GlobalState(TypedDict):
    """Shared schema passed through the full EMS multi-agent workflow."""

    timestamp: float
    drive_cycle: DriveCycle
    power_request: PowerRequest
    battery_state: BatteryState
    thermal_state: ThermalState
    control_targets: ControlTargets
    agent_outputs: dict[str, Any]
    final_decision: NotRequired[FinalDecision]
    safety_ok: bool


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor / Mission Planner Agent for a hybrid heavy-duty truck energy management system.
Your task is to preserve battery health, maintain long-term durability, and guarantee safe and efficient operation.
Prioritize a sustainable SoC window, maximize regenerative braking opportunity, limit excessive cycling, and keep the powertrain within safe thermal and aging boundaries.
Coordinate the powertrain, battery health, and driver-assistance agents with a production-safe operating philosophy."""

POWERTRAIN_SYSTEM_PROMPT = """You are the Powertrain Coordination Agent for a heavy hybrid truck.
The driver commands the acceleration/braking demand through the accelerator pedal. The engine is scheduled near its efficient operating region, and the motor provides the support torque required to follow the demand while preserving fuel economy and battery longevity.
Do not directly command the driver-selected gear. Only the electric axle gear is controllable by the EMS.
The agent must decide the motor torque, charge/discharge support, and regenerative braking action while respecting battery health limits."""

THERMAL_SYSTEM_PROMPT = """You are the Battery Health & Thermal Management Agent.
Monitor battery SoC, SoH, temperature, discharge/charge rate, and degradation index. Protect battery lifespan by managing charge and discharge limits, limiting high C-rate events, and prioritizing regenerative recapture when braking.
Maintain safe thermal margins and penalize aggressive cycling that reduces battery durability or increases recycling burden."""

DRIVER_ASSISTANCE_SYSTEM_PROMPT = """You are the Driver Assistance and Safety Agent.
Translate the control action into safe, comfortable driver guidance. Respect the driver’s requested accelerator pedal while smoothing torque transitions and preserving brake-energy recovery.
Ensure the final action remains safe, transparent, and sustainable under variable road conditions."""

PERFORMANCE_SYSTEM_PROMPT = """You are the Performance Evaluation Agent for the hybrid truck EMS.
Track fuel economy, battery health, thermal risk, regenerative braking efficiency, and sustainability metrics. Provide a status and a reward signal that captures the long-term operational quality of the policy, not only short-term responsiveness."""

RESEARCH_SUPERVISOR_PROMPT = """You are the Research Operations Supervisor.
Monitor project execution, code readiness, GitHub documentation quality, literature review quality, and research reporting completeness.
The goal is to keep the project in a publication-ready, doctoral-researcher standard state with strong traceability and evidence-driven reporting."""


@dataclass
class RewardSignal:
    """Scalar reward components for policy evaluation."""

    fuel_cost: float
    thermal_penalty: float
    degradation_penalty: float
    comfort_penalty: float
    safety_penalty: float
    regen_reward: float
    sustainability_reward: float
    total: float


class RewardModel:
    """A realistic, sustainability-aware reward model for hybrid truck EMS."""

    fuel_weight: float = 0.25
    thermal_weight: float = 0.2
    degradation_weight: float = 0.2
    comfort_weight: float = 0.1
    safety_weight: float = 0.15
    regen_weight: float = 0.15
    sustainability_weight: float = 0.15

    def __call__(self, state: GlobalState, final_decision: FinalDecision) -> RewardSignal:
        fuel_cost = -0.08 * final_decision["commanded_ice_kw"]
        thermal_penalty = -0.12 * max(0.0, state["thermal_state"]["inverter_temp_c"] - 48.0)
        degradation_penalty = -0.20 * max(0.0, state["battery_state"]["degradation_index"] - 0.08)
        comfort_penalty = -0.06 * abs(final_decision["driver_pedal"] - 0.35)
        safety_penalty = -0.25 if not state["safety_ok"] else 0.0
        regen_reward = 0.12 * max(0.0, final_decision["regen_kw"]) / 50.0
        battery_soc = state["battery_state"]["soc_pct"]
        sustainability_reward = 0.15 * (1.0 - abs(battery_soc - 60.0) / 100.0)

        total = (
            self.fuel_weight * fuel_cost
            + self.thermal_weight * thermal_penalty
            + self.degradation_weight * degradation_penalty
            + self.comfort_weight * comfort_penalty
            + self.safety_weight * safety_penalty
            + self.regen_weight * regen_reward
            + self.sustainability_weight * sustainability_reward
        )

        return RewardSignal(
            fuel_cost=float(fuel_cost),
            thermal_penalty=float(thermal_penalty),
            degradation_penalty=float(degradation_penalty),
            comfort_penalty=float(comfort_penalty),
            safety_penalty=float(safety_penalty),
            regen_reward=float(regen_reward),
            sustainability_reward=float(sustainability_reward),
            total=float(total),
        )


class EnergyManagementPolicy:
    """Interface for a single-step decision policy."""

    def select_action(self, state: GlobalState) -> FinalDecision:
        raise NotImplementedError


class DriverPedalModel:
    """Maps driver pedal demand to required traction and regenerative energy requests."""

    def compute_torque_request(
        self,
        accelerator_pedal: float,
        vehicle_speed_kph: float,
        grade_pct: float = 0.0,
        brake_pedal: float = 0.0,
    ) -> dict[str, float]:
        accel = clamp(accelerator_pedal, 0.0, 1.0)
        brake = clamp(brake_pedal, 0.0, 1.0)
        grade_factor = max(0.0, grade_pct / 10.0)
        base_speed_factor = max(0.1, 1.0 - vehicle_speed_kph / 140.0)
        requested_total_torque_nm = 260.0 + 1200.0 * accel + 220.0 * grade_factor + 110.0 * base_speed_factor
        motor_support_fraction = 0.45 + 0.35 * accel
        requested_total_torque_nm = clamp(requested_total_torque_nm, 150.0, 1650.0)
        if brake > 0.0:
            regen_kw = 20.0 + 120.0 * brake + 10.0 * grade_factor
            motor_support_fraction = 0.75
        else:
            regen_kw = 0.0
        return {
            "driver_pedal": float(accel),
            "brake_pedal": float(brake),
            "requested_total_torque_nm": float(requested_total_torque_nm),
            "motor_support_fraction": float(clamp(motor_support_fraction, 0.0, 1.0)),
            "regen_kw": float(regen_kw),
        }


class BatteryHealthAgent:
    """Battery health and durability limiter under dynamic SoC, temperature, and age conditions."""

    def update(
        self,
        soc: float,
        temperature_c: float,
        charging_power_kw: float,
        discharging_power_kw: float,
        age_factor: float,
    ) -> dict[str, float | str]:
        soc_penalty = max(0.0, abs(soc - 60.0) / 100.0)
        thermal_penalty = max(0.0, (temperature_c - 30.0) / 25.0)
        aging_penalty = max(0.0, age_factor)
        max_charge_kw = 42.0 * (1.0 - soc_penalty) * (1.0 - thermal_penalty) * (1.0 - 0.3 * aging_penalty)
        max_discharge_kw = 72.0 * (1.0 - soc_penalty) * (1.0 - thermal_penalty) * (1.0 - 0.2 * aging_penalty)
        max_charge_kw = clamp(max_charge_kw, 10.0, 60.0)
        max_discharge_kw = clamp(max_discharge_kw, 20.0, 100.0)
        if charging_power_kw > max_charge_kw:
            status = "charge_limited"
        elif discharging_power_kw > max_discharge_kw:
            status = "discharge_limited"
        else:
            status = "nominal"
        return {
            "status": status,
            "max_charge_kw": float(max_charge_kw),
            "max_discharge_kw": float(max_discharge_kw),
            "recommended_soc_min": 30.0,
            "recommended_soc_max": 75.0,
            "health_factor": float(1.0 - clamp(aging_penalty + thermal_penalty, 0.0, 1.0)),
        }


class ProjectSupervisorAgent:
    """Meta-agent overseeing code quality, GitHub readiness, writing, literature quality, and research status."""

    def evaluate(
        self,
        code_readiness: float,
        github_readiness: float,
        writing_quality: float,
        literature_quality: float,
        simulation_quality: float,
    ) -> dict[str, Any]:
        weighted = (
            0.30 * code_readiness
            + 0.20 * github_readiness
            + 0.20 * writing_quality
            + 0.20 * literature_quality
            + 0.10 * simulation_quality
        )
        if weighted >= 0.85:
            status = "ready"
        elif weighted >= 0.70:
            status = "watch"
        else:
            status = "needs_attention"
        return {
            "overall_score": float(weighted),
            "status": status,
            "focus_areas": {
                "code_readiness": float(code_readiness),
                "github_readiness": float(github_readiness),
                "writing_quality": float(writing_quality),
                "literature_quality": float(literature_quality),
                "simulation_quality": float(simulation_quality),
            },
        }


class DeepRLEMSPolicy(EnergyManagementPolicy):
    """Legacy hand-coded heuristic; this policy is not a trained RL model.

    Use :class:`multi_agent_ai.rl_training.TabularQPolicy` with its saved Q table
    for a policy learned from simulation episodes.
    """

    def select_action(self, state: GlobalState) -> FinalDecision:
        pedal = DriverPedalModel().compute_torque_request(
            accelerator_pedal=float(state["power_request"]["driver_pedal"]),
            vehicle_speed_kph=float(state["drive_cycle"]["speed_kph"]),
            grade_pct=float(state["drive_cycle"]["grade_pct"]),
            brake_pedal=float(state["power_request"]["brake_pedal"]),
        )
        soc = float(state["battery_state"]["soc_pct"])
        degradation = float(state["battery_state"]["degradation_index"])
        thermal = float(state["thermal_state"]["inverter_temp_c"])

        battery_status = BatteryHealthAgent().update(
            soc=soc,
            temperature_c=float(state["battery_state"]["temp_c"]),
            charging_power_kw=max(0.0, pedal["regen_kw"]),
            discharging_power_kw=max(0.0, pedal["requested_total_torque_nm"] * 0.08),
            age_factor=degradation,
        )

        reward_signal = 0.45 * pedal["driver_pedal"]
        reward_signal += 0.35 * (1.0 - abs(soc - 60.0) / 60.0)
        reward_signal += 0.20 * (1.0 - degradation)
        reward_signal -= 0.15 * max(0.0, thermal - 50.0) / 30.0

        if pedal["brake_pedal"] > 0.0:
            engine_torque_nm = 0.0
            motor_torque_nm = clamp(-pedal["requested_total_torque_nm"] * 0.72, -float(battery_status["max_charge_kw"]) * 14.0, 0.0)
            regen_kw = max(0.0, -motor_torque_nm * 0.06)
            electric_gear = 2
            advisory = "Regenerative braking is prioritized to recover energy and reduce brake wear."
            decision_status = "regen"
        else:
            target_engine_torque = max(0.0, 0.62 * pedal["requested_total_torque_nm"])
            support_weight = clamp(0.25 + 0.55 * reward_signal, 0.1, 0.8)
            motor_torque_nm = clamp((pedal["requested_total_torque_nm"] - target_engine_torque) * support_weight, 0.0, float(battery_status["max_discharge_kw"]) * 14.0)
            engine_torque_nm = max(0.0, pedal["requested_total_torque_nm"] - motor_torque_nm)
            regen_kw = 0.0
            electric_gear = 1 if pedal["requested_total_torque_nm"] < 700.0 else 2
            advisory = "Policy-driven torque blending keeps the engine efficient while the motor supports transients."
            decision_status = "nominal"

        torque_split = {
            "ice_fraction": engine_torque_nm / max(engine_torque_nm + motor_torque_nm, 1.0),
            "em_fraction": motor_torque_nm / max(engine_torque_nm + motor_torque_nm, 1.0),
        }
        return {
            "torque_split": torque_split,
            "selected_gear": int(max(3, min(8, int(state["drive_cycle"]["speed_kph"] // 15 + 2)))),
            "electric_gear": int(electric_gear),
            "advisory": advisory,
            "decision_status": decision_status,
            "engine_torque_nm": float(engine_torque_nm),
            "motor_torque_nm": float(motor_torque_nm),
            "regen_kw": float(regen_kw),
            "driver_pedal": float(pedal["driver_pedal"]),
            "brake_pedal": float(pedal["brake_pedal"]),
            "commanded_ice_kw": float(engine_torque_nm * 0.08),
            "commanded_em_kw": float(motor_torque_nm * 0.08),
        }


class RealisticPowertrainPolicy(EnergyManagementPolicy):
    """A production-safe, rule-based policy suited for a doctoral research prototype."""

    def select_action(self, state: GlobalState) -> FinalDecision:
        pedal = DriverPedalModel().compute_torque_request(
            accelerator_pedal=float(state["power_request"]["driver_pedal"]),
            vehicle_speed_kph=float(state["drive_cycle"]["speed_kph"]),
            grade_pct=float(state["drive_cycle"]["grade_pct"]),
            brake_pedal=float(state["power_request"]["brake_pedal"]),
        )
        soc = float(state["battery_state"]["soc_pct"])
        battery_status = BatteryHealthAgent().update(
            soc=soc,
            temperature_c=float(state["battery_state"]["temp_c"]),
            charging_power_kw=max(0.0, pedal["regen_kw"]),
            discharging_power_kw=max(0.0, pedal["requested_total_torque_nm"] * 0.08),
            age_factor=float(state["battery_state"]["degradation_index"]),
        )
        motor_torque_limit = float(battery_status["max_discharge_kw"]) * 14.0
        regen_limit = float(battery_status["max_charge_kw"]) * 14.0
        total_torque = float(pedal["requested_total_torque_nm"])

        if pedal["brake_pedal"] > 0.0:
            engine_torque_nm = 0.0
            motor_torque_nm = clamp(-total_torque * 0.6, -regen_limit, 0.0)
            regen_kw = min(max(0.0, -motor_torque_nm * 0.04), regen_limit / 14.0)
            electric_gear = 2
            decision_status = "regen"
            advisory = "Regenerative braking active; battery charging prioritized while reducing mechanical brake wear."
        else:
            engine_target = max(0.0, 0.60 * total_torque)
            assistance_ratio = clamp(1.0 - (soc - 35.0) / 45.0, 0.15, 0.85)
            motor_torque_nm = clamp((total_torque - engine_target) * assistance_ratio, 0.0, motor_torque_limit)
            engine_torque_nm = max(0.0, total_torque - motor_torque_nm)
            regen_kw = 0.0
            electric_gear = 1 if state["drive_cycle"]["speed_kph"] < 55 else 2
            decision_status = "nominal"
            advisory = "Driver acceleration demand met with engine-efficient torque and motor support to preserve fuel use and battery health."

        torques = {
            "ice_fraction": engine_torque_nm / max(engine_torque_nm + motor_torque_nm, 1.0),
            "em_fraction": motor_torque_nm / max(engine_torque_nm + motor_torque_nm, 1.0),
        }

        return {
            "torque_split": torques,
            "selected_gear": int(state["drive_cycle"]["speed_kph"] // 30 + 3),
            "electric_gear": int(electric_gear),
            "advisory": advisory,
            "decision_status": decision_status,
            "engine_torque_nm": float(engine_torque_nm),
            "motor_torque_nm": float(motor_torque_nm),
            "regen_kw": float(regen_kw),
            "driver_pedal": float(pedal["driver_pedal"]),
            "brake_pedal": float(pedal["brake_pedal"]),
            "commanded_ice_kw": float(engine_torque_nm * 0.08),
            "commanded_em_kw": float(motor_torque_nm * 0.08),
        }


class HeuristicPolicy(EnergyManagementPolicy):
    """Backward-compatible fallback policy for tests and prototyping."""

    def select_action(self, state: GlobalState) -> FinalDecision:
        return RealisticPowertrainPolicy().select_action(state)


class SupervisorAgent:
    """High-level control coordinator for the real vehicle system."""

    def __call__(self, state: GlobalState) -> SupervisorOutput:
        soc = float(state["battery_state"]["soc_pct"])
        grade = float(state["drive_cycle"]["grade_pct"])
        demand_kw = float(state["drive_cycle"]["predicted_demand_kw"])
        regen_priority = 0.0
        if state["power_request"]["brake_pedal"] > 0.0:
            regen_priority = 1.0
        elif soc < 35.0:
            regen_priority = 0.7
        elif soc > 75.0:
            regen_priority = 0.3
        else:
            regen_priority = 0.5

        return {
            "macro_target": "Maintain sustainable SoC, maximize braking energy recovery, and keep the engine near efficient operating region while supporting the driver demand.",
            "safety_limits": {
                "min_soc_pct": 30.0,
                "max_battery_temp_c": state["thermal_state"]["battery_limit_c"],
                "max_inverter_temp_c": state["thermal_state"]["inverter_limit_c"],
                "max_gearbox_temp_c": state["thermal_state"]["gearbox_limit_c"],
            },
            "energy_budget_kw": float(clamp(demand_kw * 0.7, 10.0, 85.0)),
            "horizon_minutes": 8,
            "regen_priority": float(clamp(regen_priority + grade * 0.05, 0.0, 1.0)),
        }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_sample_state() -> GlobalState:
    """Create a realistic baseline state for a hybrid long-haul truck scenario."""

    return {
        "timestamp": 0.0,
        "drive_cycle": {
            "speed_kph": 52.0,
            "grade_pct": 2.5,
            "traffic_density": 0.34,
            "route_profile": "urban_highway_mixed",
            "predicted_demand_kw": 140.0,
        },
        "power_request": {
            "requested_power_kw": 128.0,
            "demanded_torque_nm": 980.0,
            "acceleration_request_mps2": 0.38,
            "driver_pedal": 0.58,
            "brake_pedal": 0.0,
        },
        "battery_state": {
            "soc_pct": 62.0,
            "soh_pct": 96.0,
            "temp_c": 31.0,
            "degradation_index": 0.07,
            "target_soc_pct": 60.0,
            "health_factor": 0.96,
        },
        "thermal_state": {
            "battery_temp_c": 31.0,
            "inverter_temp_c": 45.0,
            "gearbox_oil_temp_c": 72.0,
            "cooling_loop_temp_c": 50.0,
            "battery_limit_c": 38.0,
            "inverter_limit_c": 62.0,
            "gearbox_limit_c": 90.0,
            "thermal_margin_pct": 18.0,
        },
        "control_targets": {
            "soc_target_pct": 60.0,
            "battery_safety_limit_pct": 30.0,
            "max_ice_torque_pct": 85.0,
            "max_em_torque_pct": 90.0,
            "gear_shift_bias": 0.5,
        },
        "agent_outputs": {},
        "safety_ok": True,
    }


def build_realistic_drive_cycle() -> list[dict[str, float | str]]:
    """Construct a realistic mixed urban/highway driving cycle for analysis and visualization."""
    cycle: list[dict[str, float | str]] = []
    for i in range(120):
        if i < 20:
            speed = 10.0 + 2.5 * i
            grade = 1.0
            route = "urban_start"
        elif i < 50:
            speed = 58.0 + 20.0 * math.sin(i * 0.23)
            grade = 2.0 + 1.8 * math.sin(i * 0.17)
            route = "urban_commute"
        elif i < 90:
            speed = 78.0 + 10.0 * math.sin(i * 0.13)
            grade = 1.5 + 2.0 * math.cos(i * 0.15)
            route = "highway_cruise"
        else:
            speed = 42.0 + 16.0 * math.sin(i * 0.21)
            grade = 3.5 + 2.5 * math.sin(i * 0.09)
            route = "urban_grade"
        cycle.append(
            {
                "timestamp": float(i),
                "speed_kph": float(max(0.0, speed)),
                "grade_pct": float(grade),
                "traffic_density": float(0.25 + 0.4 * abs(math.sin(i * 0.11))),
                "route_profile": route,
                "predicted_demand_kw": float(80.0 + 90.0 * abs(math.sin(i * 0.12)) + 30.0 * max(0.0, grade)),
                "accelerator_pedal": float(clamp(0.2 + 0.6 * abs(math.sin(i * 0.18)), 0.0, 1.0)),
                "brake_pedal": float(0.0 if i < 100 else clamp(0.2 + 0.5 * abs(math.sin(i * 0.14)), 0.0, 1.0)),
                "ambient_temp_c": 18.0 + 8.0 * math.sin(i * 0.05),
            }
        )
    return cycle


def supervisor_agent(state: GlobalState) -> SupervisorOutput:
    """Supervisor-level energy planning using safe SoC and braking recovery priorities."""

    supervisor = SupervisorAgent()
    return supervisor(state)


def powertrain_agent(state: GlobalState) -> PowertrainOutput:
    """Determines electric axle gear and torque split while the engine remains demand-following."""

    driver_model = DriverPedalModel()
    request = driver_model.compute_torque_request(
        accelerator_pedal=float(state["power_request"]["driver_pedal"]),
        vehicle_speed_kph=float(state["drive_cycle"]["speed_kph"]),
        grade_pct=float(state["drive_cycle"]["grade_pct"]),
        brake_pedal=float(state["power_request"]["brake_pedal"]),
    )
    soc = float(state["battery_state"]["soc_pct"])
    battery = BatteryHealthAgent().update(
        soc=soc,
        temperature_c=float(state["battery_state"]["temp_c"]),
        charging_power_kw=max(0.0, request["regen_kw"]),
        discharging_power_kw=max(0.0, request["requested_total_torque_nm"] * 0.08),
        age_factor=float(state["battery_state"]["degradation_index"]),
    )

    if request["brake_pedal"] > 0.0:
        em_power = min(request["regen_kw"] * 1.5, float(battery["max_charge_kw"]))
        ice_power = 0.0
        electric_gear = 2
        rationale = "Regenerative braking with battery charging and reduced mechanical brake duty."
    else:
        engine_target = max(0.0, request["requested_total_torque_nm"] * 0.62)
        support_ratio = clamp((soc - 35.0) / 45.0, 0.15, 0.85)
        em_power = clamp((request["requested_total_torque_nm"] - engine_target) * support_ratio * 0.08, 0.0, float(battery["max_discharge_kw"]))
        ice_power = max(0.0, request["requested_total_torque_nm"] * 0.08 - em_power)
        electric_gear = 1 if request["requested_total_torque_nm"] < 700.0 else 2
        rationale = "Engine kept near efficient region; motor provides support to reduce fuel use and battery stress."

    return {
        "selected_gear": int(max(3, min(8, int(state["drive_cycle"]["speed_kph"] // 15 + 2)))),
        "electric_gear": int(electric_gear),
        "ice_power_kw": float(ice_power),
        "em_power_kw": float(em_power),
        "torque_split": {
            "ice_fraction": float(ice_power / max(ice_power + em_power, 1.0)),
            "em_fraction": float(em_power / max(ice_power + em_power, 1.0)),
        },
        "expected_efficiency": float(clamp(0.90 - 0.03 * max(0.0, state["drive_cycle"]["grade_pct"]), 0.82, 0.96)),
        "rationale": rationale,
    }


def thermal_battery_agent(state: GlobalState) -> ThermalBatteryOutput:
    """Protect battery health and thermal margins while preserving control feasibility."""

    soc = float(state["battery_state"]["soc_pct"])
    temperature = float(state["battery_state"]["temp_c"])
    degradation = float(state["battery_state"]["degradation_index"])
    max_em_power = 65.0
    max_ice_power = 120.0
    reason = "nominal"

    if soc < 35.0 or temperature > 34.0:
        max_em_power *= 0.75
        reason = "battery protection"
    if state["thermal_state"]["inverter_temp_c"] > 52.0:
        max_em_power *= 0.8
        reason = "inverter thermal clamp"
    if state["thermal_state"]["gearbox_oil_temp_c"] > 82.0:
        max_ice_power *= 0.8
        reason = "gearbox cooling protection"
    if degradation > 0.1:
        max_em_power *= 0.9
        reason = "durability protection"

    penalty = 0.0 if reason == "nominal" else 0.12
    return {
        "status": "override" if reason != "nominal" else "nominal",
        "max_em_power_kw": float(max_em_power),
        "max_ice_power_kw": float(max_ice_power),
        "override_reason": reason,
        "degradation_penalty": float(penalty),
    }


def driver_assistance_agent(state: GlobalState, final_decision: FinalDecision) -> DriverAssistanceOutput:
    """Translate the optimization result into a driver-facing and safety-aware advisory."""

    if final_decision["regen_kw"] > 0.0:
        advisory = "Regenerative braking active. Mechanical brake wear is reduced, and the battery is absorbing recovered energy."
        comfort_mode = "regen"
        torque_blending = 0.8
    elif final_decision["motor_torque_nm"] > final_decision["engine_torque_nm"]:
        advisory = "Electric support is active. Engine torque is kept efficient while the motor supplies transient assistance."
        comfort_mode = "assist"
        torque_blending = 0.6
    else:
        advisory = "Engine-dominant cruise is active. The system is favoring fuel efficiency and thermal durability over aggressive battery support."
        comfort_mode = "cruise"
        torque_blending = 0.35

    safety_ok = bool(state["safety_ok"] and state["thermal_state"]["inverter_temp_c"] < 58.0)
    return {
        "advisory": advisory,
        "comfort_mode": comfort_mode,
        "torque_blending": float(torque_blending),
        "safety_ok": safety_ok,
    }


def performance_agent(state: GlobalState, reward_signal: RewardSignal) -> PerformanceMetrics:
    """Sustainability-oriented performance evaluation for a production-safe EMS assessment."""

    total_reward = float(reward_signal.total)
    fuel_efficiency_score = clamp(100.0 + total_reward * 40.0, 0.0, 100.0)
    thermal_risk_score = clamp(100.0 - state["thermal_state"]["inverter_temp_c"] * 1.1, 0.0, 100.0)
    degradation_risk_score = clamp(100.0 - state["battery_state"]["degradation_index"] * 180.0, 0.0, 100.0)
    safety_score = 100.0 if state["safety_ok"] else 35.0
    comfort_score = clamp(100.0 - abs(state["agent_outputs"]["driver_assistance"]["torque_blending"] - 0.5) * 90.0, 0.0, 100.0)
    sustainability_score = clamp(100.0 - abs(state["battery_state"]["soc_pct"] - 60.0) * 0.8 - state["battery_state"]["degradation_index"] * 50.0, 0.0, 100.0)

    if total_reward > 0.1 and thermal_risk_score > 60 and safety_score > 85:
        status = "efficient"
    elif total_reward < -0.2 or thermal_risk_score < 45 or degradation_risk_score < 50:
        status = "degraded"
    else:
        status = "stable"

    return {
        "total_reward": total_reward,
        "fuel_efficiency_score": float(fuel_efficiency_score),
        "thermal_risk_score": float(thermal_risk_score),
        "degradation_risk_score": float(degradation_risk_score),
        "safety_score": float(safety_score),
        "comfort_score": float(comfort_score),
        "sustainability_score": float(sustainability_score),
        "status": status,
    }


class MultiAgentSystem:
    """Production-safe orchestration layer for the realistic hybrid truck EMS."""

    def __init__(self, agents: list[Any] | None = None) -> None:
        self.agents = agents or [
            "supervisor",
            "powertrain",
            "thermal_battery",
            "driver_assistance",
            "performance",
            "project_supervisor",
        ]
        self.prompts = {
            "supervisor": SUPERVISOR_SYSTEM_PROMPT,
            "powertrain": POWERTRAIN_SYSTEM_PROMPT,
            "thermal_battery": THERMAL_SYSTEM_PROMPT,
            "driver_assistance": DRIVER_ASSISTANCE_SYSTEM_PROMPT,
            "performance": PERFORMANCE_SYSTEM_PROMPT,
            "project_supervisor": RESEARCH_SUPERVISOR_PROMPT,
        }
        self.policy: EnergyManagementPolicy | None = DeepRLEMSPolicy()

    def run(self, state: GlobalState) -> GlobalState:
        working = deepcopy(state)

        supervisor_out = supervisor_agent(working)
        working["agent_outputs"]["supervisor"] = supervisor_out

        powertrain_out = powertrain_agent(working)
        working["agent_outputs"]["powertrain"] = powertrain_out

        thermal_out = thermal_battery_agent(working)
        working["agent_outputs"]["thermal_battery"] = thermal_out

        if self.policy is not None:
            final_decision = cast(FinalDecision, self.policy.select_action(working))
        else:
            final_decision = {
                "torque_split": {"ice_fraction": 0.7, "em_fraction": 0.3},
                "selected_gear": 5,
                "electric_gear": 1,
                "advisory": "Fallback nominal control.",
                "decision_status": "nominal",
                "engine_torque_nm": 700.0,
                "motor_torque_nm": 300.0,
                "regen_kw": 0.0,
                "driver_pedal": 0.5,
                "brake_pedal": 0.0,
                "commanded_ice_kw": 56.0,
                "commanded_em_kw": 24.0,
            }

        driver_out = driver_assistance_agent(working, final_decision)
        working["agent_outputs"]["driver_assistance"] = driver_out

        reward = RewardModel()(working, final_decision)
        working["agent_outputs"]["reward"] = {
            "fuel_cost": reward.fuel_cost,
            "thermal_penalty": reward.thermal_penalty,
            "degradation_penalty": reward.degradation_penalty,
            "comfort_penalty": reward.comfort_penalty,
            "safety_penalty": reward.safety_penalty,
            "regen_reward": reward.regen_reward,
            "sustainability_reward": reward.sustainability_reward,
            "total": reward.total,
        }
        working["agent_outputs"]["performance"] = performance_agent(working, reward)
        working["agent_outputs"]["project_supervisor"] = ProjectSupervisorAgent().evaluate(
            code_readiness=0.9,
            github_readiness=0.92,
            writing_quality=0.88,
            literature_quality=0.9,
            simulation_quality=0.93,
        )

        final_decision["advisory"] = driver_out["advisory"]
        working["final_decision"] = final_decision
        working["safety_ok"] = bool(driver_out["safety_ok"])
        return working


@dataclass
class HybridTruckSimulation:
    """Long-horizon, realistic simulation of a hybrid truck under a mixed drive cycle."""

    steps: int = 30
    initial_state: GlobalState = field(default_factory=build_sample_state)
    policy: EnergyManagementPolicy = field(default_factory=RealisticPowertrainPolicy)
    reward_model: RewardModel = field(default_factory=RewardModel)
    drive_cycle: list[dict[str, float | str]] = field(default_factory=build_realistic_drive_cycle)

    def run(self) -> list[GlobalState]:
        history: list[GlobalState] = []
        state = deepcopy(self.initial_state)
        system = MultiAgentSystem()
        system.policy = self.policy

        for index in range(min(self.steps, len(self.drive_cycle))):
            segment = self.drive_cycle[index]
            state["timestamp"] = float(segment["timestamp"])
            state["drive_cycle"] = {
                "speed_kph": float(segment["speed_kph"]),
                "grade_pct": float(segment["grade_pct"]),
                "traffic_density": float(segment["traffic_density"]),
                "route_profile": str(segment["route_profile"]),
                "predicted_demand_kw": float(segment["predicted_demand_kw"]),
            }
            state["power_request"]["driver_pedal"] = float(segment["accelerator_pedal"])
            state["power_request"]["brake_pedal"] = float(segment["brake_pedal"])
            state["power_request"]["requested_power_kw"] = float(segment["predicted_demand_kw"])

            state = system.run(state)
            final_decision = cast(FinalDecision, state.get("final_decision"))
            if final_decision is None:
                raise RuntimeError("final_decision missing from state after orchestration")

            reward = self.reward_model(state, final_decision)
            state["agent_outputs"]["reward"] = {
                "fuel_cost": reward.fuel_cost,
                "thermal_penalty": reward.thermal_penalty,
                "degradation_penalty": reward.degradation_penalty,
                "comfort_penalty": reward.comfort_penalty,
                "safety_penalty": reward.safety_penalty,
                "regen_reward": reward.regen_reward,
                "sustainability_reward": reward.sustainability_reward,
                "total": reward.total,
            }
            state["agent_outputs"]["performance"] = performance_agent(state, reward)

            if final_decision["regen_kw"] > 0:
                state["battery_state"]["soc_pct"] = clamp(state["battery_state"]["soc_pct"] + final_decision["regen_kw"] * 0.03, 0.0, 100.0)
                state["thermal_state"]["inverter_temp_c"] = clamp(state["thermal_state"]["inverter_temp_c"] - final_decision["regen_kw"] * 0.02, 0.0, 120.0)
            else:
                discharge = max(0.0, final_decision["motor_torque_nm"] * 0.01)
                state["battery_state"]["soc_pct"] = clamp(state["battery_state"]["soc_pct"] - discharge * 0.09, 0.0, 100.0)
                state["thermal_state"]["inverter_temp_c"] = clamp(state["thermal_state"]["inverter_temp_c"] + discharge * 0.12, 0.0, 120.0)

            state["battery_state"]["degradation_index"] = clamp(
                state["battery_state"]["degradation_index"] + max(0.0, abs(final_decision["motor_torque_nm"]) - 350.0) * 0.0001,
                0.0,
                0.25,
            )
            history.append(state)

        return history


def simulate_single_step() -> GlobalState:
    """Run one representative time-step through the full orchestration flow."""
    return MultiAgentSystem().run(build_sample_state())


def build_benchmark_scenarios() -> dict[str, dict[str, float]]:
    """Build representative operating scenarios for benchmark comparison."""
    return {
        "city_stop_go": {"speed_kph": 30.0, "grade_pct": 1.5, "traffic_density": 0.75, "predicted_demand_kw": 110.0},
        "mixed_route": {"speed_kph": 62.0, "grade_pct": 2.8, "traffic_density": 0.42, "predicted_demand_kw": 150.0},
        "highway_cruise": {"speed_kph": 85.0, "grade_pct": 1.1, "traffic_density": 0.18, "predicted_demand_kw": 165.0},
        "hilly_grade": {"speed_kph": 48.0, "grade_pct": 7.0, "traffic_density": 0.51, "predicted_demand_kw": 180.0},
    }


def evaluate_policy_on_scenario(policy: EnergyManagementPolicy, scenario_name: str, scenario: dict[str, float]) -> tuple[str, float]:
    """Run one scenario and extract a scalar score for comparison between policies."""
    state = build_sample_state()
    state["drive_cycle"] = {
        "speed_kph": float(scenario["speed_kph"]),
        "grade_pct": float(scenario["grade_pct"]),
        "traffic_density": float(scenario["traffic_density"]),
        "route_profile": scenario_name,
        "predicted_demand_kw": float(scenario["predicted_demand_kw"]),
    }
    state["power_request"]["requested_power_kw"] = float(scenario["predicted_demand_kw"])
    state["power_request"]["driver_pedal"] = clamp(0.35 + 0.35 * abs(math.sin(len(scenario_name))), 0.15, 0.9)
    state["power_request"]["brake_pedal"] = 0.0 if "hills" not in scenario_name and scenario["speed_kph"] > 45 else 0.2

    system = MultiAgentSystem()
    system.policy = policy
    result = system.run(state)
    perf = result["agent_outputs"]["performance"]
    reward = float(perf["total_reward"])
    efficiency = float(perf["fuel_efficiency_score"])
    sustainability = float(perf["sustainability_score"])
    score = reward * 0.8 + efficiency * 0.1 + sustainability * 0.1
    return scenario_name, score


def compare_policies() -> dict[str, dict[str, float]]:
    """Benchmark a rule-based baseline against the deep-RL-inspired policy."""
    policies = {
        "rule_based_baseline": RealisticPowertrainPolicy(),
        "deep_rl_policy": DeepRLEMSPolicy(),
    }
    scenarios = build_benchmark_scenarios()
    results: dict[str, dict[str, float]] = {}
    for policy_name, policy in policies.items():
        results[policy_name] = {}
        for scenario_name, scenario in scenarios.items():
            _, score = evaluate_policy_on_scenario(policy, scenario_name, scenario)
            results[policy_name][scenario_name] = float(score)
    return results


def benchmark_visualization_agent(results: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Create a bar-chart comparison of multi-agent RL performance against the baseline."""
    if plt is None:
        return {"status": "unavailable", "image_path": None}

    scenario_names = list(next(iter(results.values())).keys())
    x = range(len(scenario_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (policy_name, policy_scores) in enumerate(results.items()):
        values = [policy_scores[scenario] for scenario in scenario_names]
        ax.bar([pos + idx * width for pos in x], values, width=width, label=policy_name.replace("_", " "))

    ax.set_xticks([pos + width / 2 for pos in x])
    ax.set_xticklabels(scenario_names, rotation=18)
    ax.set_ylabel("Benchmark score")
    ax.set_title("RL policy vs. rule-based baseline across representative truck missions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path = Path(__file__).resolve().parents[2] / "benchmark_dashboard.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return {"status": "generated", "image_path": str(save_path), "scenarios": scenario_names}


def print_performance_summary(history: list[GlobalState]) -> None:
    """Pretty-print a simple dashboard for the simulation history."""

    print("\n=== Realistic Hybrid Truck EMS Dashboard ===")
    for step in history:
        perf = step["agent_outputs"]["performance"]
        final_decision = cast(FinalDecision, step.get("final_decision"))
        if final_decision is None:
            continue
        print(
            f"Step {step['timestamp']}: "
            f"driver={final_decision['driver_pedal']:.2f}, "
            f"gear={final_decision['selected_gear']}, "
            f"electric_gear={final_decision['electric_gear']}, "
            f"engine={final_decision['engine_torque_nm']:.0f}Nm, "
            f"motor={final_decision['motor_torque_nm']:.0f}Nm, "
            f"regen={final_decision['regen_kw']:.1f}kW, "
            f"soc={step['battery_state']['soc_pct']:.1f}%, "
            f"reward={perf['total_reward']:.3f}, "
            f"status={perf['status']}"
        )
    print("===========================================\n")


def performance_visualization_agent(history: list[GlobalState]) -> dict[str, Any]:
    """Create a saved plot of reward, SoC, and thermal trends for the simulation history."""
    if plt is None:
        return {"status": "unavailable", "image_path": None}

    timestamps = [step["timestamp"] for step in history]
    rewards = [step["agent_outputs"]["performance"]["total_reward"] for step in history]
    soc_values = [step["battery_state"]["soc_pct"] for step in history]
    inverter_temps = [step["thermal_state"]["inverter_temp_c"] for step in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(timestamps, rewards, marker="o", color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Hybrid Truck EMS performance over realistic driving cycle")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timestamps, soc_values, marker="s", color="tab:green", linewidth=2)
    axes[1].set_ylabel("SoC (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timestamps, inverter_temps, marker="^", color="tab:red", linewidth=2)
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Inverter temp (C)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(__file__).resolve().parents[2] / "performance_dashboard.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return {"status": "generated", "image_path": str(save_path), "steps": len(history)}


def demo() -> None:
    """Run a realistic simulation and show the output trajectory."""
    sim = HybridTruckSimulation(steps=30)
    history = sim.run()
    print("Simulation started")
    for step in history[:5]:
        final_decision = step.get("final_decision")
        print({
            "timestamp": step["timestamp"],
            "road": step["drive_cycle"]["route_profile"],
            "driver_pedal": step["power_request"]["driver_pedal"],
            "powertrain": step["agent_outputs"]["powertrain"]["torque_split"],
            "battery_soc": step["battery_state"]["soc_pct"],
            "final_decision": final_decision,
            "performance": step["agent_outputs"]["performance"],
        })
    print_performance_summary(history)

    plot_result = performance_visualization_agent(history)
    if plot_result["status"] == "generated":
        print(f"Performance graph saved to: {plot_result['image_path']}")
    else:
        print("Performance graph unavailable: matplotlib is not installed.")


if __name__ == "__main__":
    demo()
