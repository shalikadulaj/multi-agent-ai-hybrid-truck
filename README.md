# Efficient and Sustainable Multi-Agent EMS for a Heavy-Duty Hybrid Truck

This project studies a multi-agent energy management strategy for a heavy-duty hybrid truck. The design is inspired by the electrification work associated with a heavy-vehicle context, with the goal of managing engine and motor power so that the truck remains efficient, safe, and durable across realistic operating conditions.

The system is organized around a layered control architecture in which a supervisory planner interprets the mission, the powertrain policy manages engine and motor torque split, the battery-health controller constrains charge and discharge, and the performance layer evaluates energy use, thermal stress, regenerative recovery, and long-term sustainability. The workflow is intentionally modular so it can be extended toward stronger RL or MPC-based optimization without sacrificing transparency or explainability.

## 1. Research framing

Heavy-duty hybrid trucks operate under very different constraints from passenger vehicles. Long mission durations, variable road grade, high transient torque demands, and the cost of battery degradation mean that energy management must balance efficiency, durability, thermal safety, and driver comfort in a coordinated way.

This repository proposes a multi-agent control structure designed for that challenge. Rather than relying on a single rule set, the decision problem is divided into specialized agents that work toward a common objective. That structure makes the project more suitable for research communication, benchmarking, and further extension toward policy optimization.

## 2. Control architecture

The system is composed of the following layers:

- Mission and safety supervisor: defines strategic objectives, safe SoC boundaries, and regenerative priorities.
- Powertrain coordination policy: allocates engine torque and motor support while respecting efficiency and driver demand.
- Battery health and thermal protection: limits charging and discharging under degradation and temperature constraints.
- Driver-assistance layer: translates the final decision into smooth and understandable guidance for the operator.
- Performance and benchmark layer: evaluates reward, thermal safety, fuel efficiency, and sustainability across missions.

This layered logic keeps the controller readable while preserving a realistic multi-agent structure.

## 3. RL-informed policy design

The current control policy is structured as an RL-informed EMS rather than a static rule-only controller. It uses a reward-aware decision pattern to balance:

- efficient engine operation,
- motor support during acceleration and transient events,
- regenerative braking during deceleration,
- SoC preservation within safe operating limits,
- battery degradation and thermal protection,
- smooth driver response and overall system comfort.

The result is a control policy that is closer to realistic hybrid-truck decision-making than a basic heuristic and is positioned as a benchmark foundation for more advanced RL or MPC implementations.

## 4. Benchmarking and public performance view

The project includes a benchmark suite based on representative truck missions:

- city stop-and-go driving,
- mixed urban/highway operation,
- sustained highway cruise,
- hill-climb and grade-heavy routes.

These scenarios are intended to show how the system behaves under different mission demands and how a multi-agent approach compares with a simpler baseline policy.

### Visual outputs

The repository includes the following plots:

- `performance_dashboard.png` for the time-series trajectory of reward, SoC, and inverter temperature.
- `benchmark_dashboard.png` for the policy comparison across the benchmark scenarios.

These images show the system’s operational behavior in a format suitable for GitHub presentation and project reporting.

## 5. How to run the project

From the repository root:

```powershell
cd "c:\Users\kwsha\OneDrive - University of Oulu and Oamk\MVD\Multi-agent AI\Oct 4"
.\venv\Scripts\python -m multi_agent_ai.main
```

Or, with the environment active:

```bash
python -m multi_agent_ai.main
```

To run the validation tests:

```bash
python -m pytest -q
```

## 6. Repository structure

- `src/multi_agent_ai/agents.py` — EMS logic, benchmark suite, reward model, and multi-agent orchestration
- `src/multi_agent_ai/main.py` — simulation and benchmark runner
- `src/multi_agent_ai/__init__.py` — public package exports
- `tests/test_multi_agent_ai.py` — validation tests
- `performance_dashboard.png` — time-series result plot
- `benchmark_dashboard.png` — multi-policy benchmark comparison
- `scripts/github_uploader.py` — repository upload helper

## 7. Scientific background and literature context

The project sits at the intersection of hybrid vehicle energy management, battery-aware control, and multi-agent optimization. The design is informed by research in electrified powertrains and coordinated control for complex energy systems.

### 7.1 Multi-agent coordination for complex control problems

A large part of the difficulty in hybrid vehicle control comes from the fact that efficiency, safety, thermal limits, and battery health are tightly coupled. Multi-agent coordination is therefore a natural design choice for decomposing the problem into manageable control objectives.

Busoniu, Babuška, and De Schutter (2008) provide a core reference for the multi-agent reinforcement learning landscape, while also motivating the high-level organization used in this repository.

### 7.2 Hybrid energy management and power-split control

Hybrid trucks require a coordinated strategy that balances fuel use, power delivery, and battery state management. The literature on hybrid-electric vehicle energy management remains highly relevant because it establishes the conceptual foundation for torque allocation, energy recovery, and operating-window constraints.

Borhan et al. (2012) showed how model predictive control can coordinate engine and motor actions under powertrain constraints, while Tie and Tan (2013) reviewed broader energy management strategies for hybrid and fuel-cell systems. Their work informs the design of the present policy structure and the emphasis on realistic operating constraints.

### 7.3 Battery health and durability-aware control

Battery degradation is a major concern in electrified truck operation. The combination of temperature, high-rate current events, and SoC variation can materially affect service life and lifecycle cost. This is especially important in heavy-duty applications where battery replacement and material impacts are significant.

Wu et al. (2018) demonstrated the value of thermal- and health-aware energy management for hybrid electric buses. The same principle is reflected in the battery health layer of this project, which constrains charge and discharge windows and prevents aggressive cycling from dominating the policy decisions.

### 7.4 Regenerative braking and energy recovery

Regenerative braking is a central element of hybrid vehicle efficiency. When braking, a portion of the kinetic energy can be captured and stored instead of being lost as heat. This reduces mechanical brake wear and improves the energy efficiency of the vehicle mission.

### 7.5 RL and policy optimization for EMS

Reinforcement learning remains a strong direction for adaptive energy management because the task is dynamic and strongly coupled. The reward model in this project includes fuel efficiency, thermal protection, battery degradation penalties, regenerative benefit, and sustainability-aware operation, so the policy is not optimized for a single short-term objective.

The repository therefore acts as a realistic benchmark environment and research-oriented control prototype, rather than a final production controller.

## 8. Representative references

1. Busoniu, L., Babuška, R., and De Schutter, B. (2008). A comprehensive survey of multi-agent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews, 38(2), 156–172.
2. Borhan, H. A., Vahidi, A., Phillips, A. M., et al. (2012). MPC-based energy management of a power-split hybrid electric vehicle. IEEE Transactions on Control Systems Technology, 20(3), 593–603.
3. Tie, S. F., and Tan, C. W. (2013). A review of energy management strategies for fuel cell hybrid electric vehicles. Renewable and Sustainable Energy Reviews, 20, 82–102.
4. Wu, J., He, H., Peng, J., Li, Y., and Zhang, Y. (2018). Battery thermal- and health-constrained energy management for hybrid electric bus based on soft actor-critic. Energy, 164, 705–714.
5. Lewis, F. L., and Vrabie, D. (2009). Reinforcement learning and adaptive dynamic programming for feedback control. IEEE Circuits and Systems Magazine, 9(3), 32–50.

## 9. Project status and direction

The repository demonstrates a realistic hybrid-truck EMS built around multi-agent coordination and RL-informed policy structure. The system includes driver-demand tracking, engine/motor torque management, regenerative braking, battery-health protection, benchmark scenarios, and visual performance outputs suitable for public reporting.

The project is intended as a solid research foundation that can evolve toward deeper RL training, stronger benchmark comparisons, and more detailed vehicle-level optimization in later stages.

## 10. License

This project is intended for research, academic demonstration, and engineering experimentation.
