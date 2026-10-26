%% Alternating Optimization for Power Allocation in RF/VLC Heterogeneous Network 
% 交替优化，结果在EE_alter1和EE_alterrow中 没有辅助变量

% Clear workspace and command window
clear;
clc;

% Parameters Initialization
n = 5;                 % Number of vehicles
time_steps = 1000;     % Number of time steps
P_max = 30;            % Maximum power (W)
P_min = 0.1;           % Minimum power (W)
B_RF = 1e6;            % RF带宽（单位：Hz）
B_VLC = 2e6;           % VLC带宽（单位：Hz）
N0_RF = 8e-23;         % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;        % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF;   % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_VLC_min = 20;        % Minimum VLC power (W)
P_RF_min = 20;         % Minimum RF power (W)
max_iter = 100;        % Increased Maximum iterations for alternating optimization
epsilon = 1e-6;        % Relaxed Convergence threshold

% Initialize power allocation matrices
P_RF = zeros(n, time_steps);
P_VLC = zeros(n, time_steps);

% Initialize channel selection matrices (1 for VLC, 0 for RF)
channel_choice = zeros(n, n, time_steps);

% Initialize SINR matrices
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% Initialize channel gains (these should be updated based on vehicle positions)
% For demonstration, we initialize with random values
gRF = rand(n, n, time_steps) * 1e-3;  % RF channel gains
gVLC = rand(n, n, time_steps) * 1e-3; % VLC channel gains

% Ensure no self-communication by setting diagonal to zero
for t = 1:time_steps
    gRF(:, :, t) = gRF(:, :, t) - diag(diag(gRF(:, :, t)));
    gVLC(:, :, t) = gVLC(:, :, t) - diag(diag(gVLC(:, :, t)));
end

% Initialize total energy efficiency record
EE_alter1 = zeros(1, time_steps);
EE_alterrow = zeros(n, time_steps); % To store per-vehicle EE

% Main Loop over time steps
for t = 1:time_steps
    fprintf('Time Step %d\n', t);
    
    % Initialize power allocation for this time step
    if t == 1
        % Initial power allocation (could be random or set to maximum)
        P_RF_current = P_max * ones(n, 1);
        P_VLC_current = P_max * ones(n, 1);
        % Initial channel choice (could be random)
        channel_choice_current = randi([0, 1], n, n);
        for i = 1:n
            channel_choice_current(i, i) = 0;  % No self-communication
        end
    else
        % Use previous time step's allocation as initial guess
        P_RF_current = P_RF(:, t-1);
        P_VLC_current = P_VLC(:, t-1);
        channel_choice_current = channel_choice(:, :, t-1);
    end
    
    % Alternating Optimization Iteration
    for iter = 1:max_iter
        if iter > 1
            P_RF_prev = P_RF_current;
            P_VLC_prev = P_VLC_current;
        end
        
        % Step 1: Fix channel choice, optimize power allocation
        % Define optimization variables and constraints
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
        
        % Optimization options
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxIterations', 100, 'MaxFunctionEvaluations', 1000);
        
        % Objective function
        obj_fun = @(x) -compute_total_EE(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
        
        % Perform optimization using fmincon
        [x_opt, ~, exitflag, output] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);
        
        % Check if optimization was successful
        if exitflag <= 0
            fprintf('Optimization did not converge at time step %d, iteration %d. Exit flag: %d\n', t, iter, exitflag);
            disp(output.message);
        end
        
        % Update power allocations
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);
        
        % Step 2: Fix power allocation, optimize channel choice
        % Save old channel choices for convergence check
        old_channel_choice = channel_choice_current;
        
        % For each communication pair, decide RF or VLC based on energy efficiency
        channel_choice_new = zeros(n, n);
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % Compute SINR for RF
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
                    
                    % Compute SINR for VLC
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    
                    % Compute equivalent SINR for VLC
                    equ_SINR_VLC = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                    
                    % Channel selection
                    if equ_SINR_VLC > SINR_RF(i, j, t)
                        channel_choice_new(i, j) = 1;  % 选择VLC
                    else
                        channel_choice_new(i, j) = 0;  % 选择RF
                    end
                end
            end
        end
        
        % Calculate differences for convergence
        power_diff_RF = 0;
        power_diff_VLC = 0;
        channel_diff = 0;
        if iter > 1
            power_diff_RF = max(abs(P_RF_current - P_RF_prev));
            power_diff_VLC = max(abs(P_VLC_current - P_VLC_prev));
            channel_diff = max(max(abs(channel_choice_new - old_channel_choice)));
        end
        
        % Update channel choices
        channel_choice_current = channel_choice_new;
        
        % Check for convergence
        if iter > 1
            if power_diff_RF < epsilon && power_diff_VLC < epsilon && channel_diff < 1e-6
                fprintf('Converged at time step %d, iteration %d\n', t, iter);
                                IterationCount(t) = iter;
                break;
            end
        end
    end
    
    % Save results for this time step
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;
    channel_choice(:, :, t) = channel_choice_current;
    
    % Compute total energy efficiency for this time step
    EE_alter1(t) = compute_total_EE([P_RF_current; P_VLC_current], gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
    
    % Compute energy efficiency for each vehicle
    EE_vehicle = zeros(1, n);
    for i = 1:n
        total_eta = 0;
        for j = 1:n
            if i ~= j
                if channel_choice_current(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC_val = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC_val);
                    power = P_VLC_current(i);
                else  % RF
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF_val = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
                    rate = B_RF * log2(1 + SINR_RF_val);
                    power = P_RF_current(i);
                end
                if power > 0
                    eta_ij = rate / power;
                else
                    eta_ij = 0;
                end
                total_eta = total_eta + eta_ij;
            end
        end
        EE_vehicle(i) = total_eta;  % Energy efficiency of vehicle i
    end
    EE_alterrow(:, t) = EE_vehicle;
    fprintf('Time Step %d: Total Energy Efficiency = %.4f bps/W\n', t, EE_alter1(t));
end

% Function to compute total energy efficiency
function EE_total = compute_total_EE(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n)
    P_RF = x(1:n);
    P_VLC = x(n+1:end);
    EE_total = 0;
    for i = 1:n
        total_eta = 0;
        for j = 1:n
            if i ~= j
                % Compute SINR and rate based on channel choice
                if channel_choice(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j) .* P_VLC) - gVLC(i, j) * P_VLC(i);
                    SINR_VLC = gVLC(i, j) * P_VLC(i) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC(i);
                else  % RF
                    interference_RF = sum(gRF(:, j) .* P_RF) - gRF(i, j) * P_RF(i);
                    SINR_RF = gRF(i, j) * P_RF(i) / (interference_RF + noise_RF);
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF(i);
                end
                if power > 0
                    eta_ij = rate / power;
                else
                    eta_ij = 0;
                end
                total_eta = total_eta + eta_ij;
            end
        end
        EE_total = EE_total + total_eta;
    end
end
