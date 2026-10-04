from __future__ import annotations

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
    """Inputs describing the upcoming road scenario."""

    speed_kph: float
    grade_pct: float
    traffic_density: float
    route_profile: str
    predicted_demand_kw: float


class PowerRequest(TypedDict):
    """Current propulsion power and transient request."""

    requested_power_kw: float
    demanded_torque_nm: float
    acceleration_request_mps2: float


class BatteryState(TypedDict):
    """Battery SoC, SoH, and degradation indicators."""

    soc_pct: float
    soh_pct: float
    temp_c: float
    degradation_index: float
    target_soc_pct: float


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


class PowertrainOutput(TypedDict):
    """Powertrain optimization action."""

    selected_gear: int
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
    status: str


class FinalDecision(TypedDict):
    """Final commanded action after all checks."""

    torque_split: dict[str, float]
    selected_gear: int
    advisory: str
    decision_status: str
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


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor / Planner Agent for a hybrid heavy-vehicle energy management system.
Your task is to read the current traffic, grade, and power demand and create a safe and efficient macro plan.
Keep battery SoC within the desired operating band, respect safety margins, and coordinate the strategic goals for the powertrain and thermal agents.
Respond with safe, concise control targets and a driver-visible macro target."""

POWERTRAIN_SYSTEM_PROMPT = """You are the Powertrain Optimization Agent for a heavy hybrid truck.
Manage the torque split between the ICE primary axle and the EM secondary axle while respecting gear selection, efficiency, and smooth torque blending.
Minimize BSFC and transient spikes while preserving driveability and satisfying the planner targets.
Output the selected gear and the ICE/EM power split as a structured decision."""

THERMAL_SYSTEM_PROMPT = """You are the Thermal & Battery Health Agent.
Monitor battery SoC, SoH, battery temperature, inverter thermal state, and gearbox cooling state.
If thermal limits or accelerated degradation risks are detected, override or penalize powertrain actions.
Protect safety margins and battery health while allowing the system to return to nominal operation when the thermal envelope is acceptable."""

DRIVER_ASSISTANCE_SYSTEM_PROMPT = """You are the Driver Assistance & Safety Agent for a hybrid truck.
Convert system decisions into driver advisories and confirm that torque blending remains smooth, safe, and comfortable.
Raise caution if system limits are approached and provide a final safety recommendation to the driver."""

PERFORMANCE_SYSTEM_PROMPT = """You are the Performance Evaluation Agent for the hybrid truck energy management system.
Track reward trends, fuel efficiency, thermal risk, battery degradation, and comfort/safety metrics.
Report whether the current policy is stable, efficient, or in a degraded operating zone so operators can interpret the system behavior."""


@dataclass
class RewardSignal:
    """Scalar reward components for policy evaluation."""

    fuel_cost: float
    thermal_penalty: float
    degradation_penalty: float
    comfort_penalty: float
    safety_penalty: float
    total: float


class RewardModel:
    """A lightweight RL-style reward function for a hybrid truck EMS."""

    fuel_weight: float = 0.40
    thermal_weight: float = 0.25
    degradation_weight: float = 0.20
    comfort_weight: float = 0.10
    safety_weight: float = 0.05

    def __call__(self, state: GlobalState, final_decision: FinalDecision) -> RewardSignal:
        fuel_cost = -0.04 * final_decision["commanded_ice_kw"]
        thermal_penalty = -0.06 * max(0.0, state["thermal_state"]["inverter_temp_c"] - 50.0)
        degradation_penalty = -0.12 * max(0.0, state["battery_state"]["degradation_index"] - 0.10)
        comfort_penalty = -0.05 * abs(final_decision["torque_split"]["em_fraction"] - 0.30)
        safety_penalty = -0.20 if not state["safety_ok"] else 0.0

        total = (
            self.fuel_weight * fuel_cost
            + self.thermal_weight * thermal_penalty
            + self.degradation_weight * degradation_penalty
            + self.comfort_weight * comfort_penalty
            + self.safety_weight * safety_penalty
        )

        return RewardSignal(
            fuel_cost=float(fuel_cost),
            thermal_penalty=float(thermal_penalty),
            degradation_penalty=float(degradation_penalty),
            comfort_penalty=float(comfort_penalty),
            safety_penalty=float(safety_penalty),
            total=float(total),
        )


class EnergyManagementPolicy:
    """Interface for a single-step decision policy."""

    def select_action(self, state: GlobalState) -> FinalDecision:
        raise NotImplementedError


class HeuristicPolicy(EnergyManagementPolicy):
    """A simple policy used as the default RL-like baseline."""

    def select_action(self, state: GlobalState) -> FinalDecision:
        powertrain = powertrain_agent(state)
        thermal = thermal_battery_agent(state)
        final_ice = clamp(powertrain["ice_power_kw"], 0.0, thermal["max_ice_power_kw"])
        final_em = clamp(powertrain["em_power_kw"], 0.0, thermal["max_em_power_kw"])
        torque_split = {
            "ice_fraction": final_ice / max(final_ice + final_em, 1.0),
            "em_fraction": final_em / max(final_ice + final_em, 1.0),
        }
        driver = driver_assistance_agent(
            state,
            {
                "torque_split": torque_split,
                "selected_gear": powertrain["selected_gear"],
                "advisory": "",
                "decision_status": "nominal" if thermal["status"] == "nominal" else "restricted",
                "commanded_ice_kw": float(final_ice),
                "commanded_em_kw": float(final_em),
            },
        )
        return {
            "torque_split": torque_split,
            "selected_gear": powertrain["selected_gear"],
            "advisory": driver["advisory"],
            "decision_status": "nominal" if thermal["status"] == "nominal" else "restricted",
            "commanded_ice_kw": float(final_ice),
            "commanded_em_kw": float(final_em),
        }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_sample_state() -> GlobalState:
    """Create a representative high-load hybrid truck operating point."""

    return {
        "timestamp": 42.0,
        "drive_cycle": {
            "speed_kph": 72.0,
            "grade_pct": 4.8,
            "traffic_density": 0.35,
            "route_profile": "urban_hill",
            "predicted_demand_kw": 145.0,
        },
        "power_request": {
            "requested_power_kw": 132.0,
            "demanded_torque_nm": 920.0,
            "acceleration_request_mps2": 0.42,
        },
        "battery_state": {
            "soc_pct": 58.5,
            "soh_pct": 96.0,
            "temp_c": 30.5,
            "degradation_index": 0.12,
            "target_soc_pct": 60.0,
        },
        "thermal_state": {
            "battery_temp_c": 32.0,
            "inverter_temp_c": 49.0,
            "gearbox_oil_temp_c": 78.0,
            "cooling_loop_temp_c": 52.0,
            "battery_limit_c": 40.0,
            "inverter_limit_c": 65.0,
            "gearbox_limit_c": 95.0,
            "thermal_margin_pct": 14.0,
        },
        "control_targets": {
            "soc_target_pct": 60.0,
            "battery_safety_limit_pct": 35.0,
            "max_ice_torque_pct": 85.0,
            "max_em_torque_pct": 80.0,
            "gear_shift_bias": 0.55,
        },
        "agent_outputs": {},
        "safety_ok": True,
    }


def supervisor_prompt(state: GlobalState) -> str:
    return (
        f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
        f"Current drive cycle: speed={state['drive_cycle']['speed_kph']} kph, "
        f"grade={state['drive_cycle']['grade_pct']}%, traffic_density={state['drive_cycle']['traffic_density']}, "
        f"predicted_demand_kw={state['drive_cycle']['predicted_demand_kw']}.\n"
        f"Battery: SoC={state['battery_state']['soc_pct']}%, target={state['battery_state']['target_soc_pct']}%, SoH={state['battery_state']['soh_pct']}%.\n"
        f"Thermal: battery={state['thermal_state']['battery_temp_c']}C, inverter={state['thermal_state']['inverter_temp_c']}C, gearbox={state['thermal_state']['gearbox_oil_temp_c']}C."
    )


def supervisor_agent(state: GlobalState) -> SupervisorOutput:
    """Set macro targets and safety boundaries."""

    demand_kw = state["drive_cycle"]["predicted_demand_kw"]
    soc = state["battery_state"]["soc_pct"]
    target_soc = state["battery_state"]["target_soc_pct"]
    safety_limit = state["control_targets"]["battery_safety_limit_pct"]

    if demand_kw > 150:
        macro_target = "Maintain SoC near the target band while allowing limited discharge for hill climb support."
    else:
        macro_target = "Maintain SoC near target and favor efficient engine-dominant cruising."

    if soc < target_soc:
        energy_budget_kw = clamp((target_soc - soc) * 2.5, 6.0, 18.0)
    else:
        energy_budget_kw = clamp((soc - target_soc) * 1.5, 4.0, 12.0)

    return {
        "macro_target": macro_target,
        "safety_limits": {
            "min_soc_pct": safety_limit,
            "max_battery_temp_c": state["thermal_state"]["battery_limit_c"],
            "max_inverter_temp_c": state["thermal_state"]["inverter_limit_c"],
            "max_gearbox_temp_c": state["thermal_state"]["gearbox_limit_c"],
        },
        "energy_budget_kw": float(energy_budget_kw),
        "horizon_minutes": 5,
    }


def powertrain_prompt(state: GlobalState) -> str:
    return (
        f"{POWERTRAIN_SYSTEM_PROMPT}\n\n"
        f"Power request: {state['power_request']['requested_power_kw']} kW, torque={state['power_request']['demanded_torque_nm']} Nm.\n"
        f"Drive conditions: speed={state['drive_cycle']['speed_kph']} kph, grade={state['drive_cycle']['grade_pct']}%.\n"
        f"Planner targets: SoC band={state['battery_state']['target_soc_pct']}%, max ICE torque={state['control_targets']['max_ice_torque_pct']}%, max EM torque={state['control_targets']['max_em_torque_pct']}%."
    )


def powertrain_agent(state: GlobalState) -> PowertrainOutput:
    """Select ICE/EM split and gear ratio for efficient propulsion."""

    requested_kw = state["power_request"]["requested_power_kw"]
    speed_kph = state["drive_cycle"]["speed_kph"]
    grade_pct = state["drive_cycle"]["grade_pct"]
    soc = state["battery_state"]["soc_pct"]

    if grade_pct > 5.0 or speed_kph < 40:
        ice_share = 0.55
        em_share = 0.45
        selected_gear = 4
    elif speed_kph >= 60:
        ice_share = 0.72
        em_share = 0.28
        selected_gear = 6
    else:
        ice_share = 0.65
        em_share = 0.35
        selected_gear = 5

    if soc < 40.0:
        em_share *= 0.5
        ice_share = 1.0 - em_share

    ice_power = requested_kw * ice_share
    em_power = requested_kw * em_share
    efficiency = clamp(0.90 - max(0.0, grade_pct - 3.0) * 0.02, 0.78, 0.94)

    return {
        "selected_gear": int(selected_gear),
        "ice_power_kw": float(ice_power),
        "em_power_kw": float(em_power),
        "torque_split": {
            "ice_fraction": float(ice_share),
            "em_fraction": float(em_share),
        },
        "expected_efficiency": float(efficiency),
        "rationale": "Engine-dominant cruise under moderate load with EM torque support during grade or transient demand.",
    }


def thermal_prompt(state: GlobalState) -> str:
    return (
        f"{THERMAL_SYSTEM_PROMPT}\n\n"
        f"Battery SoC={state['battery_state']['soc_pct']}%, SoH={state['battery_state']['soh_pct']}%, temp={state['battery_state']['temp_c']}C.\n"
        f"Thermal envelopes: battery={state['thermal_state']['battery_temp_c']}C/{state['thermal_state']['battery_limit_c']}C, inverter={state['thermal_state']['inverter_temp_c']}C/{state['thermal_state']['inverter_limit_c']}C, gearbox={state['thermal_state']['gearbox_oil_temp_c']}C/{state['thermal_state']['gearbox_limit_c']}C."
    )


def thermal_battery_agent(state: GlobalState) -> ThermalBatteryOutput:
    """Protect battery and thermal system through operating envelope overrides."""

    batt_temp = state["thermal_state"]["battery_temp_c"]
    inverter_temp = state["thermal_state"]["inverter_temp_c"]
    gearbox_temp = state["thermal_state"]["gearbox_oil_temp_c"]
    soc = state["battery_state"]["soc_pct"]
    degradation = state["battery_state"]["degradation_index"]

    max_em_power = 65.0
    max_ice_power = 120.0
    reason = "nominal"

    if batt_temp > 35.0 or soc < 35.0:
        max_em_power *= 0.7
        reason = "battery protection"
    if inverter_temp > 55.0:
        max_em_power *= 0.8
        reason = "inverter thermal clamp"
    if gearbox_temp > 85.0:
        max_ice_power *= 0.8
        reason = "gearbox cooling protection"
    if degradation > 0.15:
        max_em_power *= 0.9
        reason = "degradation reduction"

    penalty = 0.0
    if reason != "nominal":
        penalty = 0.12

    return {
        "status": "override" if reason != "nominal" else "nominal",
        "max_em_power_kw": float(max_em_power),
        "max_ice_power_kw": float(max_ice_power),
        "override_reason": reason,
        "degradation_penalty": float(penalty),
    }


def driver_prompt(state: GlobalState, final_decision: FinalDecision) -> str:
    return (
        f"{DRIVER_ASSISTANCE_SYSTEM_PROMPT}\n\n"
        f"Final recommended command: ICE={final_decision['commanded_ice_kw']} kW, EM={final_decision['commanded_em_kw']} kW, gear={final_decision['selected_gear']}.\n"
        f"Driver comfort target: smooth torque blending and stable acceleration under speed {state['drive_cycle']['speed_kph']} kph."
    )


def driver_assistance_agent(state: GlobalState, final_decision: FinalDecision) -> DriverAssistanceOutput:
    """Translate the optimized output to driver-advisory guidance."""

    ice_kw = final_decision["commanded_ice_kw"]
    em_kw = final_decision["commanded_em_kw"]
    torque_blending = clamp((em_kw / max(ice_kw + em_kw, 1.0)), 0.0, 1.0)
    safety_ok = state["safety_ok"] and (state["thermal_state"]["inverter_temp_c"] < 60.0)

    if state["drive_cycle"]["grade_pct"] > 6.0:
        advisory = "Moderate hill climb: EM support active, maintain steady pedal input to avoid torque transients."
        comfort_mode = "assist"
    elif torque_blending > 0.42:
        advisory = "Smooth electric assist engaged; keep torque delivery gradual for driver comfort."
        comfort_mode = "balanced"
    else:
        advisory = "Engine-dominant operation is stable; expect efficient cruise with moderate smoothness."
        comfort_mode = "cruise"

    return {
        "advisory": advisory,
        "comfort_mode": comfort_mode,
        "torque_blending": float(torque_blending),
        "safety_ok": bool(safety_ok),
    }


def performance_agent(state: GlobalState, reward_signal: RewardSignal) -> PerformanceMetrics:
    """Evaluate how well the current control policy is performing."""

    total_reward = float(reward_signal.total)
    fuel_efficiency_score = clamp(100.0 + total_reward * 25.0, 0.0, 100.0)
    thermal_risk_score = clamp(100.0 - state["thermal_state"]["inverter_temp_c"] * 1.2, 0.0, 100.0)
    degradation_risk_score = clamp(100.0 - state["battery_state"]["degradation_index"] * 200.0, 0.0, 100.0)
    safety_score = 100.0 if state["safety_ok"] else 35.0
    comfort_score = clamp(100.0 - abs(state["agent_outputs"]["driver_assistance"]["torque_blending"] - 0.3) * 100.0, 0.0, 100.0)

    if total_reward > 0 and thermal_risk_score > 60 and safety_score > 80:
        status = "efficient"
    elif total_reward < -0.5 or thermal_risk_score < 45:
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
        "status": status,
    }


class MultiAgentSystem:
    """Sequential orchestration layer for the hybrid-truck EMS agents."""

    def __init__(self, agents: list[Any] | None = None) -> None:
        self.agents = agents or [
            "supervisor",
            "powertrain",
            "thermal_battery",
            "driver_assistance",
            "performance",
        ]
        self.prompts = {
            "supervisor": SUPERVISOR_SYSTEM_PROMPT,
            "powertrain": POWERTRAIN_SYSTEM_PROMPT,
            "thermal_battery": THERMAL_SYSTEM_PROMPT,
            "driver_assistance": DRIVER_ASSISTANCE_SYSTEM_PROMPT,
            "performance": PERFORMANCE_SYSTEM_PROMPT,
        }
        self.policy: EnergyManagementPolicy | None = HeuristicPolicy()

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
            commanded_ice_kw = clamp(
                powertrain_out["ice_power_kw"],
                0.0,
                thermal_out["max_ice_power_kw"],
            )
            commanded_em_kw = clamp(
                powertrain_out["em_power_kw"],
                0.0,
                thermal_out["max_em_power_kw"],
            )
            final_decision = cast(
                FinalDecision,
                {
                    "torque_split": {
                        "ice_fraction": commanded_ice_kw / max(commanded_ice_kw + commanded_em_kw, 1.0),
                        "em_fraction": commanded_em_kw / max(commanded_ice_kw + commanded_em_kw, 1.0),
                    },
                    "selected_gear": powertrain_out["selected_gear"],
                    "advisory": "",
                    "decision_status": "nominal" if thermal_out["status"] == "nominal" else "restricted",
                    "commanded_ice_kw": float(commanded_ice_kw),
                    "commanded_em_kw": float(commanded_em_kw),
                },
            )

        driver_out = driver_assistance_agent(working, final_decision)
        working["agent_outputs"]["driver_assistance"] = driver_out

        reward = RewardModel()(working, final_decision)
        working["agent_outputs"]["reward"] = {
            "fuel_cost": reward.fuel_cost,
            "thermal_penalty": reward.thermal_penalty,
            "degradation_penalty": reward.degradation_penalty,
            "comfort_penalty": reward.comfort_penalty,
            "safety_penalty": reward.safety_penalty,
            "total": reward.total,
        }
        working["agent_outputs"]["performance"] = performance_agent(working, reward)

        final_decision["advisory"] = driver_out["advisory"]
        working["final_decision"] = final_decision
        working["safety_ok"] = bool(driver_out["safety_ok"])

        return working


@dataclass
class HybridTruckSimulation:
    """Rolling optimization loop for a time-stepped heavy-truck EMS simulation."""

    steps: int = 3
    initial_state: GlobalState = field(default_factory=build_sample_state)
    policy: EnergyManagementPolicy = field(default_factory=HeuristicPolicy)
    reward_model: RewardModel = field(default_factory=RewardModel)

    def run(self) -> list[GlobalState]:
        history: list[GlobalState] = []
        state = deepcopy(self.initial_state)
        system = MultiAgentSystem()
        system.policy = self.policy

        for _ in range(self.steps):
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
                "total": reward.total,
            }
            state["agent_outputs"]["performance"] = performance_agent(state, reward)
            state["battery_state"]["soc_pct"] = clamp(
                state["battery_state"]["soc_pct"] - final_decision["commanded_em_kw"] * 0.01,
                0.0,
                100.0,
            )
            state["thermal_state"]["inverter_temp_c"] = clamp(
                state["thermal_state"]["inverter_temp_c"] + final_decision["commanded_em_kw"] * 0.06,
                0.0,
                120.0,
            )
            state["timestamp"] += 1.0
            history.append(state)

        return history


def simulate_single_step() -> GlobalState:
    """Run one representative time-step through the full orchestration flow."""

    system = MultiAgentSystem()
    sample = build_sample_state()
    return system.run(sample)


def print_performance_summary(history: list[GlobalState]) -> None:
    """Pretty-print a simple dashboard for the simulation history."""

    print("\n=== Hybrid Truck EMS Performance Dashboard ===")
    for step in history:
        perf = step["agent_outputs"]["performance"]
        final_decision = cast(FinalDecision, step.get("final_decision"))
        if final_decision is None:
            continue
        print(
            f"Step {step['timestamp']}: "
            f"gear={final_decision['selected_gear']}, "
            f"ICE={final_decision['commanded_ice_kw']:.1f}kW, "
            f"EM={final_decision['commanded_em_kw']:.1f}kW, "
            f"status={perf['status']}, "
            f"reward={perf['total_reward']:.3f}, "
            f"safety={perf['safety_score']:.1f}, "
            f"comfort={perf['comfort_score']:.1f}"
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
    axes[0].set_title("Hybrid Truck EMS performance over time")
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
    """Run a visible console demo with a few steps of the EMS simulation."""
    sim = HybridTruckSimulation(steps=3)
    history = sim.run()
    print("Simulation started")
    for step in history:
        final_decision = step.get("final_decision")
        print({
            "timestamp": step["timestamp"],
            "supervisor": step["agent_outputs"]["supervisor"]["macro_target"],
            "powertrain": step["agent_outputs"]["powertrain"]["torque_split"],
            "thermal": step["agent_outputs"]["thermal_battery"]["status"],
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
