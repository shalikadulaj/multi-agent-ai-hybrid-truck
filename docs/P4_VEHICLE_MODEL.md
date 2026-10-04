# Assumed two-axle P4 hybrid truck model

## Architecture being represented

For this project, the proposed topology is represented as a **P4-type parallel hybrid**: a conventional engine and transmission deliver torque to one driven axle, while a mechanically independent electric motor and e-axle deliver torque to a second driven axle. During braking, the e-axle can generate negative wheel force and recover part of the energy; friction brakes supply the remainder. The model does not assume which physical axle is engine-driven versus electrically driven, because that installation detail has not been supplied.

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
| Electric axle maximum wheel-side power | 160 kW | Assumed cap |
| Electric regenerative power cap | 120 kW | Assumed cap |
| Battery energy capacity | 120 kWh | Assumed pack capacity |
| Engine axle efficiency | 0.94 | Fixed assumed driveline efficiency |
| Electric axle nominal efficiency | 0.93 | Gear-dependent multiplier also applied |
| Nominal timestep | 1 s | Fixed-step model |
| Maximum reference acceleration | 0.30 m/s² | Deliberately below vehicle cap to allow speed-feedback headroom |
| Maximum reference deceleration | 0.60 m/s² | Before the physical/service-deceleration cap |
| Maximum modeled acceleration | 0.65 m/s² | Assumed longitudinal limit |
| Maximum modeled service deceleration | 1.20 m/s² | Assumed; braking model has no ABS or tire-force limit |

Exact values and the training seed are saved with each run in `artifacts/q_policy_metadata.json`.

## Reward used for Q-learning

The Q-learning reward is updated for the P4 model. It includes charge-sustaining fuel-equivalent energy (so the policy cannot claim a free gain by discharging the battery), battery throughput/degradation proxy, thermal-limit cost, friction-brake work, speed-tracking error, and unmet wheel power. The intended per-step structure is:

$$
r_t=-\left[100\left(\dot V_{fuel}+\frac{\Delta E_{batt}}{\eta_{ref}H_{diesel}}\right)
+w_{cyc}E_{throughput}+w_{brake}E_{friction}+w_{temp}C_{temp}+w_{track}C_{track}+w_{unmet}P_{unmet}\right].
$$

Here positive $\Delta E_{batt}$ means energy removed from the battery; charging makes it negative. Thus stored battery energy reduces the equivalent-fuel charge for that step. The weights and battery-aging relationship are tunable research assumptions, not safety requirements or calibrated lifecycle economics. See `reward` terms in `src/multi_agent_ai/rl_training.py` for the executable calculation.

Current implementation coefficients: fuel-equivalent use is scaled by 100; throughput stress by 2.0; friction-brake energy by 1.5; thermal exceedance above 38 °C by 0.02 per squared degree; speed error by 0.03 per km/h per step; and unmet wheel power by 0.05 per kW. These coefficients were selected for this prototype's numeric reward scale and require sensitivity analysis and calibration before research conclusions.

## Included dynamics and important exclusions

Included: target-speed tracking, longitudinal inertia, grade/rolling/aerodynamic resistance, engine/e-axle power caps, separate axle force contributions, battery SoC-dependent power limits, regenerative/friction brake blending, fuel-use estimate, and first-order battery-temperature/throughput proxies.

Not included: validated Sisu engine or motor maps, torque-speed envelopes, transmission shift logic, axle differential/traction limits, tire-road slip, load transfer, vehicle yaw/lateral dynamics, engine turbo/transient states, detailed battery electrochemistry or aging, cooling-system dynamics, driver traffic interaction, or hardware/functional-safety behavior. The electric gear is currently represented by a coarse efficiency difference, not a motor RPM/torque map. Component power requests and the prescribed speed cycle therefore remain simplified inputs.

Use the results to compare controller behavior **within this assumed model**. Do not interpret the fuel estimates as Sisu test results, and do not connect the policy to actuators. Calibration requires measured mass/load, coast-down or road-load coefficients, tire/wheel data, engine fuel map, motor/inverter maps, battery OCV/resistance/current/thermal limits, axle ratios, and measured drive-cycle/speed traces. Validation should compare speed, acceleration, fuel, DC power, SoC, temperature, and brake energy against independent vehicle or validated plant data across held-out missions.
