# Multi-Agent AI System for Hybrid Truck Energy Management

This project implements a lightweight multi-agent energy management system (EMS) for a hybrid heavy-duty truck. It models a simplified but realistic control architecture in which multiple specialized agents cooperate to choose engine and electric motor power split, manage thermal limits, and evaluate system performance. The system is deliberately designed as an educational and research prototype that is easy to inspect, extend, and visualize.

## 1. Project overview

The core idea is to emulate a hierarchical control stack inspired by multi-agent reinforcement learning and classical vehicle energy management. Instead of relying on a single monolithic controller, the system separates decision-making into distinct roles:

- Supervisor agent: sets strategic goals and safety limits
- Powertrain agent: chooses power split and gear ratio
- Thermal/battery health agent: enforces thermal and degradation constraints
- Driver-assistance agent: converts decisions into a driver-facing advisory
- Performance agent: computes reward and quality metrics

This creates a multi-agent control loop in which each agent contributes a partial decision and the system assembles a final action.

## 2. System architecture

The architecture is designed around the following state flow:

1. A drive-cycle state is created from vehicle speed, grade, traffic density, and predicted power demand.
2. The supervisor sets the macro target and safety budget.
3. The powertrain agent determines torque split and gear.
4. The thermal/battery health agent applies override constraints if the system approaches thermal or degradation limits.
5. The driver-assistance agent evaluates comfort and safety before finalizing the advisory.
6. A reward model evaluates the action using a simple RL-inspired objective.
7. The performance agent aggregates reward, efficiency, thermal risk, and comfort/safety into a status label for the current step.

This approach is not a production-safe vehicle control system, but it captures the main structure used in research on multi-agent and learning-based EMS.

## 3. Simulation logic

The project exposes a rolling simulation through the `HybridTruckSimulation` class. Each time step updates the truck state, re-evaluates the agent outputs, and accumulates performance history. The simulation writes visible output to the console and saves a performance chart in the workspace.

The main execution entry point is:

```bash
python -m multi_agent_ai.main
```

The generated chart is saved as:

- `performance_dashboard.png`

## 4. Control behavior and interpretation

The current policy is heuristic rather than fully learned. It acts as a baseline controller that approximates RL-style decision-making using explicit operating rules. This makes it suitable for demonstration and experimentation, while leaving room for future replacement with:

- a more advanced rule-based strategy,
- dynamic programming or MPC-based optimization,
- reinforcement learning policy optimization,
- multi-agent coordination with communication or negotiation.

The current output shows the actual behavior of the controller in a short time sequence. For example, the system may choose a gear of 6, split the power between the internal combustion engine and motor, and then compute a negative total reward when the system drifts toward degraded operation. This is useful because it reveals how the model interprets efficiency, thermal risk, and comfort trade-offs, rather than merely printing a final command.

## 5. Repository structure

- `src/multi_agent_ai/agents.py` — core agent logic, reward model, performance evaluator, and simulation loop
- `src/multi_agent_ai/main.py` — visible simulation entry point
- `src/multi_agent_ai/__init__.py` — project exports
- `tests/test_multi_agent_ai.py` — basic validation scaffold
- `performance_dashboard.png` — generated performance chart
- `scripts/github_uploader.py` — helper script for preparing a GitHub push

## 6. How to run the project

Windows / PowerShell:

```powershell
cd "c:\Users\kwsha\OneDrive - University of Oulu and Oamk\MVD\Multi-agent AI\Oct 4"
.\venv\Scripts\python -m multi_agent_ai.main
```

Or, after activating the environment:

```bash
python -m multi_agent_ai.main
```

## 7. Literature review and background

The design of this system is grounded in several research areas that are central to hybrid vehicle energy management and multi-agent decision-making.

### 7.1 Multi-agent systems and coordination

Multi-agent systems are widely used when a complex control task can be decomposed into interacting subproblems. The idea is that specialized agents can share local information while coordinating toward a global objective. In vehicle control, this decomposition is especially useful because energy efficiency, thermal safety, battery health, and driver comfort are not independent variables. A useful reference is the survey by Busoniu et al. (2008), which systematically reviews multi-agent reinforcement learning and coordination frameworks. This work explains why decentralized or hierarchical control can be useful when a problem contains local constraints and global performance objectives.

The present project follows that same principle: each agent focuses on a narrow part of the decision space, while the final decision is assembled as a combined control action.

### 7.2 Energy management in hybrid electric vehicles

Hybrid vehicle energy management is a classic control problem because it requires balancing fuel economy, emissions, battery state-of-charge, and powertrain constraints. Traditional approaches include rule-based control, dynamic programming, and model predictive control. Review papers on hybrid and fuel-cell vehicle energy management emphasize that the challenge is no longer only about power sharing, but also about resource preservation, thermal constraints, and real-time feasibility.

Tie and Tan (2013) provide a review of energy management strategies for fuel-cell hybrid electric vehicles and show how strategy design depends on both system hardware and operating conditions. Similarly, Borhan et al. (2012) present a model predictive control framework for a power-split hybrid vehicle, showing how optimization can manage engine and motor coordination while considering powertrain dynamics and constraints.

This project follows the same spirit, but with a simplified heuristic and reward-based formulation rather than full optimization over a high-fidelity vehicle model.

### 7.3 Reinforcement learning for energy management

Reinforcement learning is attractive for energy management because it can learn a policy from interacting with a dynamical environment without requiring a complete analytical solution at every step. This is especially important in vehicles with nonlinear constraints and uncertain driving conditions. The reward formulation used here approximates a reinforcement-learning objective by penalizing fuel cost, battery degradation, thermal stress, and discomfort.

Recent literature has applied deep reinforcement learning to hybrid vehicle energy management. For example, Wu et al. (2018) develop a battery thermal- and health-constrained reinforcement-learning method for hybrid electric buses, showing that health-aware and thermal-aware objectives can materially affect the learned policy. This aligns with the current system, which explicitly includes thermal risk and degradation penalties in the reward function.

### 7.4 Why this project matters

The value of this project is pedagogical and architectural. It demonstrates how an energy management problem can be decomposed into specialized agents, each with a limited focus, while sharing state and constraints. This is a useful step toward more realistic multi-agent control systems for electrified vehicles, grid-aware charging, and fleet-level energy coordination.

This simplified prototype is not meant to replace a high-fidelity automotive control model, but it captures the main ideas used in academic research on hybrid energy management, multi-agent learning, and performance monitoring.

## 8. Representative references

1. Busoniu, L., Babuska, R., & De Schutter, B. (2008). A comprehensive survey of multi-agent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews, 38(2), 156–172.
2. Tie, S. F., & Tan, C. W. (2013). A review of energy management strategies for fuel cell hybrid electric vehicles. Renewable and Sustainable Energy Reviews, 20, 82–102.
3. Borhan, H. A., Vahidi, A., Phillips, A. M., et al. (2012). MPC-based energy management of a power-split hybrid electric vehicle. IEEE Transactions on Control Systems Technology, 20(3), 593–603.
4. Wu, J., He, H., Peng, J., Li, Y., & Zhang, Y. (2018). Battery thermal- and health-constrained energy management for hybrid electric bus based on soft actor-critic. Energy, 164, 705–714.
5. Kessels, J. T. B. A., et al. (2007). Energy management for hybrid electric vehicles using stochastic dynamic programming. In Proceedings of the IEEE Intelligent Vehicles Symposium.
6. Lewis, F. L., & Vrabie, D. (2009). Reinforcement learning and adaptive dynamic programming for feedback control. IEEE Circuits and Systems Magazine, 9(3), 32–50.

These references provide a foundational baseline for the conceptual design, policy structure, and control objectives used in this project.

## 9. Future work

The next logical steps for this project are:

- replace the heuristic rule generator with a learned RL policy,
- add a more realistic battery degradation model,
- compare multiple policies under the same drive cycle,
- extend the system to multiple connected vehicles or a fleet-level EMS,
- integrate remote monitoring and dashboarding for real-time visualization.

## 10. Summary

This repository presents a compact multi-agent energy management prototype for a hybrid truck. It is intentionally simple enough for learning and experimentation while still reflecting the key architectural patterns discussed in the literature: multi-agent decision decomposition, reward-based evaluation, thermal and battery protection, and performance monitoring. The system already demonstrates visible output and graph generation, which makes it suitable for both internal development and presentation to stakeholders or academic supervisors.

## 11. License

This project is intended for educational and research use.
