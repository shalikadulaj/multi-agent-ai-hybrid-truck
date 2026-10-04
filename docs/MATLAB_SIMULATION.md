# Run the trained EMS from MATLAB

This workflow assumes **MATLAB**. It runs the trained Q-learning policy and the low-order truck plant in Python, writes a timestamped CSV rollout, then imports and visualizes the results in MATLAB. It is useful for algorithm review and MATLAB plotting; it is **not** a Simulink vehicle plant, HIL integration, or hardware validation. “Mtalba” is interpreted as MATLAB here; if it means another program, these MATLAB-specific steps will not apply.

## 1. Prepare the Python environment

From PowerShell at the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest
```

If a compatible environment already exists, activate or use its Python executable instead. MATLAB releases support specific Python versions; check the MATLAB documentation for your installed release before selecting the interpreter.

## 2. Train and inspect the saved evaluation

```powershell
.\.venv\Scripts\python.exe -m multi_agent_ai.rl_training --episodes 1000 --seed 2026
```

The run writes the Q table, metadata, episode log, per-episode benchmark data, summary CSV, and result plot under `artifacts/`. The exact episode count and seed are recorded in `q_policy_metadata.json`.

## 3. Plot a trained-policy rollout in MATLAB

Open MATLAB, set its current folder to the repository root, and run:

```matlab
run('matlab/run_trained_ems.m')
```

The script calls the repository's `.venv` Python executable, evaluates a saved policy on a reproducible mixed-route cycle, exports `artifacts/matlab_trajectory.csv`, and plots speed, engine/motor/regen power, battery SoC and temperature, and cumulative fuel use.

If your Python environment is elsewhere, edit `pythonExe` near the top of `matlab/run_trained_ems.m`. Verify that `pythonExe -m multi_agent_ai.rl_training --help` works in that same environment.

## 4. Use the trained policy in Simulink (research integration)

Use the MATLAB script as an offline functional check before building a closed-loop model. For a Simulink integration, keep the EMS action interface discrete and explicit:

- propulsion: four motor-assist fractions (`0`, `0.20`, `0.40`, `0.60`) crossed with e-axle gear 1 or 2;
- braking: four requested regen fractions (`0`, `0.33`, `0.67`, `1.0`) crossed with e-axle gear 1 or 2;
- driver-selected truck gear remains an input and must not be controlled by this EMS;
- enforce battery SoC/current/temperature, engine/motor limits, torque rate, and brake blending in an independent safety supervisor.

For an initial MATLAB prototype, read `artifacts/q_policy.csv` with `readtable`, reproduce the documented binning in `src/multi_agent_ai/rl_training.py`, map the discretized state to `state_index`, and select the largest of `q_action_0` through `q_action_7`. Keep the Python-versus-MATLAB action outputs regression-tested before trusting a port. The Python environment must remain the reference implementation until that validation is in place.

## Scope and safety

The included longitudinal/powertrain model is a low-order research model with nominal, uncalibrated parameters. It is not a validated Sisu truck digital twin. This repository's plots are software simulation results, not measurements from a physical truck. Do not connect this policy directly to vehicle actuators. Real deployment requires measured vehicle and battery maps, independent functional-safety analysis, real-time implementation, HIL/bench testing, and supervised vehicle trials.
