# Assumed two-axle P4 hybrid truck model

## Architecture being represented

For this project, the topology is represented as a **P4-type parallel hybrid** per the user's installation detail: the **rear axle is the independent electric e-axle**, while the engine mechanically drives the other axle. This model labels the other axle as front for its two-axle abstraction; that front-engine-axle placement remains an assumption pending vehicle drawings. During braking, the rear e-axle can generate negative wheel force and recover part of the energy; friction brakes supply the remainder.

This encodes the topology requested for the project. It does **not** verify that a specific Sisu production vehicle has these component ratings or this exact axle arrangement.

## Longitudinal vehicle equation

The vehicle is advanced at a one-second fixed step. The equivalent mass includes a factor for rotating components:

$$
m_{eq}\dot v = F_{eng,wheel}+F_{em,wheel}-F_{regen,wheel}-F_{friction}-F_{rr}-F_{aero}-F_{grade}.
$$

The forward road-load terms are:

$$
F_{rr}=C_{rr}mg\cos\theta,\quad
F_{aero}=\tfrac12\rho C_d A v^2,\quad
F_{grade}=mg\sin\theta.
$$

The speed trace is a driver-requested reference. The simulated vehicle speed is a state integrated from the net wheel force; target acceleration plus speed-tracking feedback gives the requested longitudinal acceleration. Requested wheel force is split between engine-axle and electric-axle actions, subject to rated power and battery limits. A speed-dependent electric-axle ratio efficiency differentiates its two discrete gear choices.

Engine shaft demand is calculated from engine-axle wheel power and an assumed axle efficiency. Fuel use is estimated with a generic engine-efficiency curve. The e-axle's electrical energy is calculated from its wheel power and an assumed conversion efficiency. In braking, the requested deceleration determines total braking force; the policy's regenerative fraction is constrained by motor and charge-power limits, and friction brakes fill the residual force.

## Assumed nominal parameters

These values are **engineering starting assumptions for a representative loaded heavy truck**, not specifications measured from a Sisu truck. Replace them with approved vehicle/component data before making quantitative claims.

| Parameter | Nominal model value | Notes |
|---|---:|---|
| Gross vehicle mass | 32,000 kg | Assumed operating mass; real mission mass varies |
| Rotating-mass factor | 1.05 | Approximate wheel/driveline inertia correction |
| Wheel radius | 0.50 m | Nominal; currently metadata only, no tire slip model |
| Rolling resistance coefficient | 0.007 | Road/tire assumption |
| Drag coefficient | 0.62 | Assumed heavy-truck body |
| Frontal area | 9.0 m² | Assumed |
| Air density | 1.20 kg/m³ | Nominal ambient |
| Engine maximum shaft power | 360 kW | Assumed cap, not engine map |
| Rear e-axle maximum wheel-side power | 160 kW | Assumed cap |
| Rear e-axle regenerative power cap | 120 kW | Assumed cap |
| Battery energy capacity | 120 kWh | Assumed pack capacity |
| Engine axle efficiency | 0.94 | Fixed assumed driveline efficiency |
| Rear electric axle nominal efficiency | 0.93 | Gear-dependent multiplier also applied |
| Nominal timestep | 1 s | Fixed-step model |
| Maximum reference acceleration | 0.30 m/s² | Deliberately below vehicle cap to allow speed-feedback headroom |
| Maximum reference deceleration | 0.60 m/s² | Before the physical/service-deceleration cap |
| Maximum modeled acceleration | 0.65 m/s² | Assumed longitudinal limit |
| Maximum modeled service deceleration | 1.20 m/s² | Assumed; braking model has no ABS or tire-force limit |

Exact values, user-specified rear e-axle placement, front engine-axle assumption, controller settings, and training seed are saved with each run in `artifacts/q_policy_metadata.json`.

## Battery-health proxy and reward

The RL observation includes SoC, SoH, demand, vehicle speed, grade, battery temperature, and braking state. The learned controller uses Double Q-learning: two Q estimators are maintained and the estimator selected for evaluation is separated from the estimator used for value evaluation. This reduces the max-operator overestimation bias reported for ordinary Q-learning, but does not remove the need for validation or guarantee improved real-vehicle performance. The run reports cumulative charge/discharge throughput and equivalent full cycles (EFC), defined here as total absolute charge-plus-discharge energy divided by twice nominal battery energy. It also reports peak temperature, time above 35 °C, and a calendar-aging proxy.

An **illustrative cycle-fade proxy** applies an assumed SoH loss of 0.01 percentage points per EFC, multiplied by stress terms for SoC, temperature (assumed Q10=2 above 25 °C), and C-rate above 0.5 C. A separate illustrative calendar term accumulates with elapsed time, SoC above 60%, and the same temperature multiplier. These parameter values and stress equations are prototype assumptions, not values calibrated to a Sisu cell or pack and not a chemistry-independent aging law. They cannot estimate warranty life, years of use, or true remaining useful life.

Peer-reviewed literature motivates considering health: Wang et al. describe a battery-fading index for predictive HEV energy management; Wu et al. study thermal- and health-constrained DRL energy management; Schmalstieg et al. present a chemistry-specific holistic aging model for NMC 18650 cells. Those papers motivate tracking health and stress but **do not calibrate the coefficients used in this repository**. Cell-specific data are required.

The reward includes charge-sustaining fuel-equivalent energy (so battery depletion is not a free fuel gain), battery throughput and the explicit fade proxy, thermal-limit cost, friction-brake work, speed-tracking error, and unmet wheel power:

$$
r_t=-\left[100\left(\Delta V_{fuel}+\frac{\Delta E_{batt}}{\eta_{ref}H_{diesel}}\right)
+w_{cyc}E_{throughput}+w_{fade}\Delta SoH+w_{brake}E_{friction}+w_{temp}C_{temp}+w_{track}C_{track}+w_{unmet}P_{unmet}\right].
$$

Here positive $\Delta E_{batt}$ means energy removed from the battery; charging makes it negative. Current implementation coefficients: fuel-equivalent use is scaled by 100; throughput stress by 2.0; fade by 25,000 reward units per SoH percentage point; friction-brake energy by 1.5; thermal exceedance above 38 °C by 0.02 per squared degree; speed error by 0.03 per km/h per step; and unmet wheel power by 0.05 per kW. The large fade shadow price is a declared tuning assumption; Q and MPC use the same value. Coefficients require sensitivity analysis and calibration before research conclusions.

## Health-aware predictive controller and bounded contribution

Double Q-learning remains the learned comparator, but its coarse state discretization can be data-inefficient. The benchmark therefore adds a receding-horizon beam-search MPC: it evaluates the same eight discrete rear-e-axle actions over a **five-step (5 s) speed/grade cycle preview**, keeps the eight highest-return partial sequences, applies only the first action, then replans. It uses the P4 model's fuel-equivalent, fade, friction-brake, thermal, tracking, and traction terms, plus terminal SoC deviation and terminal high-temperature costs. This MPC sees perfect short preview from the generated cycle; the preview advantage must be acknowledged in comparisons, and a real controller would require a navigation/sensor preview source plus runtime verification.

The project's defensible contribution is a reproducible rear-e-axle P4 evaluation framework that compares Double Q-learning and a short-horizon health-costed predictive controller using fuel-equivalent use, tracking error, EFC, cycle/calendar SoH proxies, thermal exposure, and friction-brake work. This is a project-specific integration and benchmark contribution; **it is not claimed to be first or state of the art**. Such claims require a systematic literature review and broader matched baselines.

## Included dynamics and important exclusions

Included: target-speed tracking, longitudinal inertia, grade/rolling/aerodynamic resistance, engine/e-axle power caps, separate axle force contributions, battery SoC-dependent power limits, regenerative/friction brake blending, fuel-use estimate, first-order battery-temperature, cycle-throughput, and calendar-aging proxies.

Not included: validated Sisu engine or motor maps, torque-speed envelopes, transmission shift logic, axle differential/traction limits, tire-road slip, load transfer, vehicle yaw/lateral dynamics, engine turbo/transient states, detailed battery electrochemistry, validated calendar aging, cooling-system dynamics, driver traffic interaction, or hardware/functional-safety behavior. The electric gear is currently represented by a coarse efficiency difference, not a motor RPM/torque map. Component power requests and the prescribed speed cycle remain simplified inputs.

Use results only to compare controller behavior **within this assumed model**. Do not interpret fuel estimates or SoH proxies as Sisu test results, and do not connect the policy to actuators. Calibration requires approved axle layout, measured mass/load, coast-down/road-load coefficients, tire/wheel data, engine fuel map, rear motor/inverter maps, battery chemistry and cell/pack aging data, current/thermal limits, axle ratios, and measured drive-cycle traces. Validate speed, acceleration, fuel, DC power, SoC, temperature, battery capacity fade, and brake energy against independent vehicle or validated plant data across held-out missions.

## Peer-reviewed sources checked

1. Wang, X., Ma, S., and Wang, J. (2018). “Predictive energy management for hybrid electric vehicles considering extension of the battery life.” *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering*, 232(4), 499–510. DOI: [10.1177/0954407017703229](https://doi.org/10.1177/0954407017703229).
2. Wu, J. et al. (2021). “Battery Thermal- and Health-Constrained Energy Management for Hybrid Electric Bus Based on Soft Actor-Critic DRL Algorithm.” *IEEE Transactions on Industrial Informatics*, 17(6), 3751–3761. DOI: [10.1109/TII.2020.3014599](https://doi.org/10.1109/TII.2020.3014599).
3. Schmalstieg, J., Käbitz, S., Ecker, M., and Sauer, D. U. (2014). “A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion batteries.” *Journal of Power Sources*, 257, 325–334. DOI: [10.1016/j.jpowsour.2014.02.012](https://doi.org/10.1016/j.jpowsour.2014.02.012).
4. van Hasselt, H., Guez, A., and Silver, D. (2016). “Deep Reinforcement Learning with Double Q-learning.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1). DOI: [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295).
