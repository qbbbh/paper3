% Parameters
num_vehicles = 5;        % Number of vehicles
num_timesteps = 1000;    % Total simulation time (ms)
B_RF = 1e6;              % RF bandwidth (Hz)
B_VLC = 2e6;             % VLC bandwidth (Hz)
N0_RF = 8e-23;           % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;          % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF; % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_max_RF = 20;           % Maximum RF power (W)
P_max_VLC = 30;          % Maximum VLC power (W)
P_min_RF = 0.1;          % Minimum RF power (W)
P_min_VLC = 20;          % Minimum VLC power (W)
R_min_RF = 1e3;          % Minimum RF rate (bps)
R_min_VLC = 1e3;         % Minimum VLC rate (bps)

% % Load channel gain matrices (g_RF and g_VLC)
% load('channel_gain_RF.mat');  % RF channel gains (5x5x1000)
% load('channel_gain_VLC.mat'); % VLC channel gains (5x5x1000)

% Initialize actions and power allocations
x_RF = zeros(num_vehicles, num_vehicles); % RF selection matrix
x_VLC = zeros(num_vehicles, num_vehicles); % VLC selection matrix
P_RF = ones(num_vehicles, num_vehicles) * P_max_RF / 2; % Initial RF power
P_VLC = ones(num_vehicles, num_vehicles) * P_max_VLC / 2; % Initial VLC power

% Initialize results tracking
EE_history = zeros(num_timesteps, 1);         % Track total EE at each timestep
EE_per_vehicle = zeros(num_timesteps, num_vehicles); % EE per vehicle at each timestep
iteration_count = zeros(num_timesteps, 1);    % Iteration count at each timestep

for t = 1:num_timesteps
    % Update channel gains for the current time step
    g_RF_current = gRF(:, :, t); % RF gains at time t
    g_VLC_current = gVLC(:, :, t); % VLC gains at time t
    
    % Initialize convergence criteria
    max_iter = 50; % Maximum iterations
    epsilon = 1e-6; % Convergence threshold
    iter = 0;
    delta = Inf;

    % Iterative optimization
    while iter < max_iter && delta > epsilon
        % Store previous power values for convergence check
        P_RF_prev = P_RF;
        P_VLC_prev = P_VLC;

        % Update actions (x_RF, x_VLC) based on EE and constraints
        for i = 1:num_vehicles
            for j = 1:num_vehicles
                if i ~= j % Avoid self-communication
                    % Calculate effective EE for RF and VLC
        interference_RF = sum(g_RF_current(:, j) .* P_RF(:, j)) - g_RF_current(i, j) * P_RF(i, j);
        interference_RF = max(interference_RF, 1e-9); % 防止分母为零
        SINR_RF(i,j) = g_RF_current(i, j) * P_RF(i, j) / (interference_RF + noise_RF);
                % 计算 VLC 的干扰和 SINR
        interference_VLC = sum(g_VLC_current(:, j) .* P_VLC(:, j)) - g_VLC_current(i, j) * P_VLC(i, j);
        interference_VLC = max(interference_VLC, 1e-9); % 防止分母为零
        SINR_VLC(i,j) = g_VLC_current(i, j) * P_VLC(i, j) / (interference_VLC + noise_VLC);
                            % 计算VLC等效SINR
        equ_SINR_VLC = 2 * ((1 + SINR_VLC(i,j)/3.3)^(B_VLC/B_RF) - 1);
                    
                    % 信道选择
                    if equ_SINR_VLC > SINR_RF
                        x_RF(i, j) = 0;
                        x_VLC(i, j) = 1;  % 选择VLC
                    else
                        x_RF(i, j) = 1;
                        x_VLC(i, j) = 0;  % 选择RF
                    end
                    % EE_RF_ij = C_RF(i, j) / max(P_RF(i, j), P_min_RF);
                    % EE_VLC_ij = C_VLC(i, j) / max(P_VLC(i, j), P_min_VLC);

                 end
             end
         end
      

        % Update power allocations
        P_RF = min(P_max_RF, max(P_min_RF, P_RF + randn(size(P_RF)) * 0.1));
        P_VLC = min(P_max_VLC, max(P_min_VLC, P_VLC + randn(size(P_VLC)) * 0.1));

        % Check convergence
        delta = max(abs(P_RF(:) - P_RF_prev(:))) + max(abs(P_VLC(:) - P_VLC_prev(:)));
        iter = iter + 1;
    end

    % Record iteration count
    iteration_count(t) = iter;


        % 计算即时奖励（假设奖励与能量效率相关）
        reward = calculate_reward(C_RF, C_VLC, P_RF, P_VLC, R_min_RF, R_min_VLC);

        % 更新状态（假设状态与信噪比有关，离散化到num_states范围）
        next_state = min(num_states, max(1, round(SINR * num_states)));

        % % 更新Q表
        % Q_table(state(i), action, i) = Q_table(state(i), action, i) + ...
        %     alpha * (reward + gamma * max(Q_table(next_state, :, i)) - Q_table(state(i), action, i));

    % Calculate capacities and total EE
    [C_RF, C_VLC] = calculate_capacity(g_RF_current, g_VLC_current, P_RF, P_VLC, noise_RF, noise_VLC, B_RF, B_VLC,SINR_RF,SINR_VLC);
    EE_total_RF = sum(C_RF(:) ./ max(P_RF(:), P_min_RF) .* x_RF(:)); % Total EE for RF
    EE_total_VLC = sum(C_VLC(:) ./ max(P_VLC(:), P_min_VLC) .* x_VLC(:)); % Total EE for VLC
    EE_history(t) = EE_total_RF + EE_total_VLC;

    % Calculate EE for each vehicle
    for i = 1:num_vehicles
        EE_per_vehicle(t, i) = sum(C_RF(i, :) ./ max(P_RF(i, :), P_min_RF) .* x_RF(i, :)) + ...
                               sum(C_VLC(i, :) ./ max(P_VLC(i, :), P_min_VLC) .* x_VLC(i, :));
    end
end
disp('Optimization complete.');
% Plot total EE over time
figure;
plot(1:num_timesteps, EE_history, 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Total EE (bps/W)');
title('Total Energy Efficiency over Time');
grid on;

% Plot EE per vehicle over time
figure;
plot(1:num_timesteps, EE_per_vehicle, 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Energy Efficiency per Vehicle (bps/W)');
title('Energy Efficiency per Vehicle over Time');
legend('Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4', 'Vehicle 5');
grid on;
% Plot BER per vehicle over time
figure;
plot(1:num_timesteps, avg_BER_history, 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Average BER');
title('Average Bit Error Rate per Vehicle over Time');
legend('Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4', 'Vehicle 5');
grid on;

% Plot iteration count over time
figure;
plot(1:num_timesteps, iteration_count, 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Iteration Count');
title('Algorithm Iteration Count over Time');
grid on;
function [C_RF, C_VLC] = calculate_capacity(g_RF, g_VLC, P_RF, P_VLC, noise_RF, noise_VLC, B_RF, B_VLC,SINR_RF,SINR_VLC)
    % % SINR for RF
    % SINR_RF = P_RF .* g_RF ./ (noise_RF + sum(P_RF .* g_RF, 2) - diag(P_RF .* g_RF));
    % 
    % % SINR for VLC
    % SINR_VLC = (P_VLC .* g_VLC).^2 ./ (noise_VLC + sum(P_VLC .* g_VLC, 2) - diag(P_VLC .* g_VLC));

    % Capacity calculations
    C_RF = B_RF * log2(1 + SINR_RF);
    C_VLC = B_VLC * log2(1 + SINR_VLC);
end
function reward = calculate_reward(C_RF, C_VLC, P_RF, P_VLC, R_min_RF, R_min_VLC)
    % Calculate energy efficiency
    EE_RF = C_RF ./ P_RF;
    EE_VLC = C_VLC ./ P_VLC;

    % Apply QoS constraints
    penalty_RF = max(0, R_min_RF - C_RF); % QoS penalty for RF
    penalty_VLC = max(0, R_min_VLC - C_VLC); % QoS penalty for VLC

    % Combine rewards
    reward = sum(EE_RF(:)) + sum(EE_VLC(:)) - sum(penalty_RF(:)) - sum(penalty_VLC(:));
end
