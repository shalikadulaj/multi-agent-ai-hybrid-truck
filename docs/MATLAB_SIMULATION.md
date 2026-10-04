# Run the trained EMS from MATLAB

This workflow assumes **MATLAB**. It runs the trained Q-learning policy and the low-order truck plant in Python, writes a timestamped CSV rollout, then imports and visualizes the results in MATLAB. It is useful for algorithm review and MATLAB plotting; it is **not** a Simulink vehicle plant, HIL integration, or hardware validation. “Mtalba” is interpreted as MATLAB here; if it means another program, these MATLAB-specific steps will not apply.

## Connect VS Code Copilot to MATLAB

This workspace is set up to use MathWorks' official [MATLAB MCP Server](https://github.com/matlab/matlab-mcp-server) with VS Code Copilot. On this PC, the local configuration targets MATLAB R2024b and the official MCP server v0.14.0. It starts an isolated `nodesktop` MATLAB session in this repository and disables the MCP server's optional telemetry. The `.tools/` server binary and `.vscode/mcp.json` configuration are intentionally git-ignored because they are local machine paths; the custom agent definition is in `.github/agents/matlab-simulation.agent.md`.

One-time setup:

1. In VS Code, open the repository as the workspace folder.
2. Open the Command Palette (`Ctrl+Shift+P`) and run **MCP: List Servers** (or open the **MCP Servers** view).
3. Select `matlab` and choose **Start Server**. Review and accept VS Code's workspace/MCP trust prompt. On its first tool call, the server may start MATLAB R2024b; MATLAB must be licensed and able to start for your Windows user.
4. In Copilot Chat's agent picker, select **MATLAB Simulation Engineer**. If it does not appear, run **Developer: Reload Window**, then reopen the agent picker.
5. Give it a task such as: “Run the existing trained EMS rollout in MATLAB, check the MATLAB script, and report the generated CSV and PNG paths.” The agent uses MATLAB MCP tools and the project script; it should report tool errors rather than claim an unexecuted run succeeded.

VS Code must remain open while an agent session is running. This creates an agent that can complete a requested simulation without you manually operating MATLAB, but it is not a continuously running background service: you still start the agent task, and VS Code may ask for tool permission depending on your approval settings. Keep approvals enabled for unfamiliar code. Do not configure auto-approval for arbitrary MATLAB evaluation; MATLAB code can access files and launch processes under your Windows account.

If MATLAB is installed at another release path, edit the `--matlab-root` value in `.vscode/mcp.json`. The current path was checked against `C:\Program Files\MATLAB\R2024b`. If you clone this repo elsewhere, also change the MCP executable path and initial working folder in that local configuration. Download the MCP server only from the linked MathWorks GitHub release page.

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

Verified on this workstation with MATLAB R2024b (24.2.0.2712019): `run('matlab/run_trained_ems.m')` completed through MATLAB batch mode, generated the CSV and `artifacts/matlab_trajectory.png`, and `checkcode('matlab/run_trained_ems.m')` returned no diagnostics. The launcher probes `venv` and `.venv` and uses the first one that can import the project module.

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
