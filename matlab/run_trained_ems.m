function run_trained_ems()
% Run the saved Python Q-learning EMS and plot its rollout in MATLAB.
% This invokes the repository's Python plant model; it is not Simulink HIL.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
pythonCandidates = {fullfile(repoRoot, 'venv', 'Scripts', 'python.exe'), ...
                    fullfile(repoRoot, '.venv', 'Scripts', 'python.exe')};
pythonExe = '';
for candidateIndex = 1:numel(pythonCandidates)
    if isfile(pythonCandidates{candidateIndex})
        checkCommand = sprintf('"%s" -c "import multi_agent_ai.rl_training" >NUL 2>&1', ...
            pythonCandidates{candidateIndex});
        if system(checkCommand) == 0
            pythonExe = pythonCandidates{candidateIndex};
            break;
        end
    end
end
if isempty(pythonExe)
    error(['No project-ready Python found in venv or .venv. Install the project with ' ...
           'python -m pip install -e . in the intended environment, or edit this script.']);
end

modelPath = fullfile(repoRoot, 'artifacts', 'q_policy.csv');
trajectoryPath = fullfile(repoRoot, 'artifacts', 'matlab_trajectory.csv');
plotPath = fullfile(repoRoot, 'artifacts', 'matlab_trajectory.png');
if ~isfile(modelPath)
    error('Trained Q table not found: %s. Run the training command in docs/MATLAB_SIMULATION.md first.', modelPath);
end

command = sprintf(['cd /d "%s" && "%s" -m multi_agent_ai.rl_training ' ...
    '--evaluate-only --model "%s" --scenario mixed_route --seed 4101 ' ...
    '--steps 120 --trajectory-csv "%s"'], ...
    repoRoot, pythonExe, modelPath, trajectoryPath);
[status, output] = system(command);
fprintf('%s\n', output);
if status ~= 0
    error('Python rollout failed with status %d.', status);
end

T = readtable(trajectoryPath);
figure('Name', 'Trained hybrid-truck EMS rollout', 'Color', 'w');
layout = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, 'Q-learning EMS — Python plant rollout viewed in MATLAB');

nexttile;
plot(T.time_s, T.speed_kph, 'LineWidth', 1.3);
ylabel('Speed (km/h)'); grid on;

nexttile;
plot(T.time_s, T.engine_power_kw, 'LineWidth', 1.2); hold on;
plot(T.time_s, T.motor_power_kw, 'LineWidth', 1.2);
plot(T.time_s, T.regen_power_kw, 'LineWidth', 1.2);
ylabel('Power (kW)'); legend('Engine', 'Motor', 'Regen', 'Location', 'eastoutside'); grid on;

nexttile;
yyaxis left;
plot(T.time_s, T.soc_pct, 'LineWidth', 1.3); ylabel('Battery SoC (%)');
yyaxis right;
plot(T.time_s, T.battery_temp_c, '--', 'LineWidth', 1.1); ylabel('Battery temp (°C)');
grid on;

nexttile;
plot(T.time_s, T.cumulative_fuel_l, 'LineWidth', 1.3);
xlabel('Time (s)'); ylabel('Fuel (L, cumulative)'); grid on;

exportgraphics(layout, plotPath, 'Resolution', 180);
fprintf('MATLAB plot saved to: %s\n', plotPath);
end
