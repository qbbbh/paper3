%% Ablation Study 1: Only Alternating Channel Selection (No Auxiliary Variables)
% Clear workspace and command window仅使用交替优化信道选择（不使用辅助变量）
clear;
clc;

% Parameters Initialization
n = 5;                  % Number of vehicles
time_steps = 10;        % Number of time steps (use smaller number for testing)
P_max = 30;             % Maximum power (W)
P_min = 1;              % Minimum power (W)
B_RF = 1e6;             % RF bandwidth (Hz)
B_VLC = 2e6;            % VLC bandwidth (Hz)
N0_RF = 8e-23;          % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;         % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF;    % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_VLC_min = 20;         % Minimum VLC power (W)
P_RF_min = 1;           % Minimum RF power (W)
P_total = P_max;        % Total power per vehicle
max_iter = 50;          % Maximum iterations for alternating optimization
epsilon = 1e-6;         % Convergence threshold

% Minimum transmission rates (QoS) for RF and VLC
R_min_RF = 1;   % Example value, adjust as needed (bps)
R_min_VLC = 1;  % Example value, adjust as needed (bps)

% Initialize power allocation matrices
P_RF = zeros(n, time_steps);
P_VLC = zeros(n, time_steps);

% Initialize channel selection matrices (1 for VLC, 0 for RF)
channel_choice = zeros(n, n, time_steps);

% Initialize SINR matrices
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% Initialize channel gains (gRF and gVLC)
% Initialize channel gains with random positive values
gRF = rand(n, n, time_steps) * 1e-3;  % RF channel gains
gVLC = rand(n, n, time_steps) * 1e-3; % VLC channel gains

% Ensure no self-communication by setting diagonal to zero
for t = 1:time_steps
    gRF(:, :, t) = gRF(:, :, t) - diag(diag(gRF(:, :, t)));
    gVLC(:, :, t) = gVLC(:, :, t) - diag(diag(gVLC(:, :, t)));
end

% Define Optimization Options
options = optimoptions('fmincon', ...
    'Display', 'off', ...                % Set to 'iter-detailed' for debugging
    'Algorithm', 'sqp', ...              % Choose SQP algorithm
    'MaxIterations', 100, ...            % Maximum iterations
    'OptimalityTolerance', 1e-6, ...     % Optimality tolerance
    'StepTolerance', 1e-6);               % Step tolerance

% Initialize total energy efficiency record
EE_altopt = zeros(1, time_steps);
IterationCount = zeros(1, time_steps); % Record iteration count for each time step

%% Main Loop over Time Steps
for t = 1:time_steps
    fprintf('Time Step %d\n', t);
    
    % Initialize power allocation for this time step
    if t == 1
        % Initial power allocation (set to minimum to allow optimization)
        P_RF_current = P_RF_min * ones(n, 1);
        P_VLC_current = P_VLC_min * ones(n, 1);
        % Initial channel choice (random)
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
        % Save previous power allocations for convergence check
        if iter > 1
            P_RF_prev = P_RF_current;
            P_VLC_prev = P_VLC_current;
        end
        
        % Step 1: Fix channel choice, optimize power allocation
        % Define optimization variables and constraints
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
        
        % Define objective function: maximize EE (minimize -EE)
        obj_fun = @(x) -compute_total_EE_basic(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
        
        % Define non-linear constraints
        nonlcon = @(x) nonlinear_constraints_basic(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n, R_min_RF, R_min_VLC, P_total);
        
        % Perform optimization using fmincon with non-linear constraints
        [x_opt, ~, exitflag, output] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, nonlcon, options);
        
        % Check if optimization was successful
        if exitflag <= 0
            fprintf('Optimization did not converge at time step %d, iteration %d. Exit flag: %d\n', t, iter, exitflag);
            disp(output.message);
        end
        
        % Update power allocations
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);
        
        % Step 2: Fix power allocation, optimize channel choice
        % For each communication pair, decide RF or VLC based on energy efficiency
        channel_choice_new = zeros(n, n);
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % Compute SINR for RF
                    interference_RF = sum(gRF(i, :) .* P_RF_current) - gRF(i, j) * P_RF_current(j);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(j) / (interference_RF + noise_RF);
                    
                    % Compute SINR for VLC
                    interference_VLC = sum(gVLC(i, :) .* P_VLC_current) - gVLC(i, j) * P_VLC_current(j);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(j) / (interference_VLC + noise_VLC);
                    
                    % Compute equivalent SINR for VLC
                    equ_SINR_VLC = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                    
                    % Channel selection based on energy efficiency
                    if equ_SINR_VLC > SINR_RF(i, j, t)
                        channel_choice_new(i, j) = 1;  % Select VLC
                    else
                        channel_choice_new(i, j) = 0;  % Select RF
                    end
                end
            end
        end
        
        % Check for convergence
        if iter > 1
            power_diff_RF = max(abs(P_RF_current - P_RF_prev));
            power_diff_VLC = max(abs(P_VLC_current - P_VLC_prev));
            channel_diff = max(max(abs(channel_choice_new - channel_choice_current)));
            if power_diff_RF < epsilon && power_diff_VLC < epsilon && channel_diff < epsilon
                fprintf('Converged at time step %d, iteration %d\n', t, iter);
                IterationCount(t) = iter;
                break;
            end
        end
        
        % Update variables for next iteration
        channel_choice_current = channel_choice_new;
    end
    
    % Save results for this time step
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;
    channel_choice(:, :, t) = channel_choice_current;
    
    % Compute total energy efficiency for this time step
    EE_altopt(t) = compute_total_EE_basic([P_RF_current; P_VLC_current], gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
end

% Calculate EE_alt for comparison (not used in this variant)
EE_altrow = sum(zeros(n, n, time_steps), 2); % Placeholder
EE_alt = sum(EE_altrow, 1); % Placeholder

%% Function to compute total energy efficiency (Basic Version)
function EE_total = compute_total_EE_basic(x_opt, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n)
    P_RF_current = x_opt(1:n);
    P_VLC_current = x_opt(n+1:end);
    EE_total = 0;
    for i = 1:n
        for j = 1:n
            if i ~= j
                % Compute SINR and rate based on channel choice
                if channel_choice(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j) .* P_VLC_current) - gVLC(i, j) * P_VLC_current(j);
                    SINR_VLC = gVLC(i, j) * P_VLC_current(j) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_current(j);  % Using transmitter's power
                else  % RF
                    interference_RF = sum(gRF(:, j) .* P_RF_current) - gRF(i, j) * P_RF_current(j);
                    SINR_RF = gRF(i, j) * P_RF_current(j) / (interference_RF + noise_RF);
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF_current(j);  % Using transmitter's power
                end
                if power > 0
                    eta_ij = rate / power;
                else
                    eta_ij = 0;
                end
                EE_total = EE_total + eta_ij;
            end
        end
    end
end

%% Function to compute non-linear constraints (Basic Version)
function [c, ceq] = nonlinear_constraints_basic(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n, R_min_RF, R_min_VLC, P_total)
    % x: [P_RF; P_VLC]
    P_RF = x(1:n);       % n x 1
    P_VLC = x(n+1:end);  % n x 1
    
    c = [];
    ceq = [];
    
    % C6 & C7: Minimum transmission rates
    for j = 1:n
        for i = 1:n
            if i ~= j
                if channel_choice(i, j) == 0  % RF
                    interference_RF = sum(gRF(i, :) .* P_RF) - gRF(i, j) * P_RF(j);
                    SINR_RF = gRF(i, j) * P_RF(j) / (interference_RF + noise_RF);
                    rate_RF = B_RF * log2(1 + SINR_RF);
                    if isnan(rate_RF) || isinf(rate_RF) || imag(rate_RF) ~= 0
                        rate_RF = 0; % Handle invalid values
                        fprintf('Warning: Invalid rate_RF at i=%d, j=%d\n', i, j);
                    end
                    c(end+1) = R_min_RF - rate_RF;  % rate_RF >= R_min_RF
                else  % VLC
                    interference_VLC = sum(gVLC(i, :) .* P_VLC) - gVLC(i, j) * P_VLC(j);
                    SINR_VLC = gVLC(i, j) * P_VLC(j) / (interference_VLC + noise_VLC);
                    rate_VLC = B_VLC * log2(1 + SINR_VLC);
                    if isnan(rate_VLC) || isinf(rate_VLC) || imag(rate_VLC) ~= 0
                        rate_VLC = 0; % Handle invalid values
                        fprintf('Warning: Invalid rate_VLC at i=%d, j=%d\n', i, j);
                    end
                    c(end+1) = R_min_VLC - rate_VLC;  % rate_VLC >= R_min_VLC
                end
            end
        end
    end
    
    % C3 & C4: Total power per vehicle
    for i = 1:n
        c(end+1) = (P_RF(i) + P_VLC(i)) - P_total;  % P_RF + P_VLC <= P_total
        if (P_RF(i) + P_VLC(i)) > P_total + 1e-6
            fprintf('Warning: Total power for vehicle %d exceeds P_total: %f > %f\n', i, P_RF(i) + P_VLC(i), P_total);
        end
    end
end
