%% Alternating Optimization with Auxiliary Variables for Power Allocation in RF/VLC Heterogeneous Network 
% 结果在 EE_altrow 和 EE_alt  EE_altopt对不上
% Parameters Initialization
n = 5;                  % Number of vehicles
time_steps = 100;      % Number of time steps
P_max = 50;             % Maximum power (W)
% P_min = 75;              % Minimum power (W)
B_RF = 1e6;             % RF bandwidth (Hz)
B_VLC = 2e6;            % VLC bandwidth (Hz)
N0_RF = 8e-23;          % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;         % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF; % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_VLC_min = 15;         % Minimum VLC power (W)
P_RF_min = 1;           % Minimum RF power (W)
P_total = P_max;        % Assuming total power per vehicle is P_max
max_iter = 50;          % Maximum iterations for alternating optimization
epsilon = 1e-6;         % Convergence threshold

% Minimum transmission rates (QoS) for RF and VLC
R_min_RF = 1e3;   % Example value, adjust as needed (bps)
R_min_VLC = 1e3;  % Example value, adjust as needed (bps)

% Initialize power allocation matrices
P_RF = zeros(n, time_steps);
P_VLC = zeros(n, time_steps);

% Initialize channel selection matrices (1 for VLC, 0 for RF)
channel_choice = zeros(n, n, time_steps);

% Initialize SINR matrices
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% Initialize auxiliary variables (三维数组)
alpha = ones(n, n, time_steps);  % Initial \(\alpha_{ij,t}\)
beta = ones(n, n, time_steps);   % Initial \(\beta_{ij,t}\)

% Initialize total energy efficiency record
EE_altopt = zeros(1, time_steps);
IterationCount = zeros(1, time_steps); % Record iteration count for each time step

% Ensure no self-communication by setting diagonal to zero
% for t = 1:time_steps
%     gRF(:, :, t) = gRF(:, :, t) - diag(diag(gRF(:, :, t)));
%     gVLC(:, :, t) = gVLC(:, :, t) - diag(diag(gVLC(:, :, t)));
% end

%% Define Optimization Options
options = optimoptions('fmincon', ...
    'Display', 'off', ...                % Set to 'iter-detailed' for debugging
    'Algorithm', 'sqp', ...              % Choose SQP algorithm
    'MaxIterations', 100, ...            % Maximum iterations
    'OptimalityTolerance', 1e-6, ...     % Optimality tolerance
    'StepTolerance', 1e-6);               % Step tolerance

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
        
        % Step 1: Fix channel choice, optimize power allocation using auxiliary variables
        % Define optimization variables and constraints
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
        
        % Define objective function: maximize EE (minimize -EE)
        obj_fun = @(x) -compute_total_EE(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
        
        % Define non-linear constraints
        nonlcon = @(x) nonlinear_constraints(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n, R_min_RF, R_min_VLC, P_total);
        
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
                    % interference_RF = sum(gRF(i, :) .* P_RF_current') - gRF(i, j) * P_RF_current(j);
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    interference_RF = max(interference_RF, 1e-9); % 防止分母为零
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(j) / (interference_RF + noise_RF);
                    
                    % Compute SINR for VLC
                    % interference_VLC = sum(gVLC(i, :) .* P_VLC_current') - gVLC(i, j) * P_VLC_current(j);
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    interference_VLC = max(interference_VLC, 1e-9); % 防止分母为零
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(j) / (interference_VLC + noise_VLC);
                    
                    % 计算VLC等效SINR
                    equ_SINR_VLC = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                    
                    % 信道选择
                    if equ_SINR_VLC > SINR_RF(i, j, t)
                        channel_choice_new(i, j) = 1;  % 选择VLC
                    else
                        channel_choice_new(i, j) = 0;  % 选择RF
                    end
                end
            end
        end
        
        % Update auxiliary variables
        for i = 1:n
            for j = 1:n
                if i ~= j
                    if channel_choice_new(i, j) == 1  % VLC
                        rate = B_VLC * log2(1 + SINR_VLC(i, j, t));
                        power = P_VLC_current(j);  % 使用发送方j的功率
                    else  % RF
                        rate = B_RF * log2(1 + SINR_RF(i, j, t));
                        power = P_RF_current(j);  % 使用发送方j的功率
                    end
                    if power > 0
                        alpha(i, j, t) = rate / power;
                        beta(i, j, t) = 1 / power;
                    else
                        alpha(i, j, t) = 0;
                        beta(i, j, t) = 0;
                    end
                else
                    alpha(i, j, t) = 0;
                    beta(i, j, t) = 0;
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
                IterationCount10(t) = iter;
                break;
            end
        end
%%        
      for i = 1:n
        for j = 1:n
           if i ~= j
             if channel_choice_new(i, j) == 1  % VLC
                SINR_10(i,j,t)=SINR_VLC(i,j,t);
             else
                SINR_10(i,j,t)=SINR_RF(i,j,t);
             end
             SINR_total(i,t)=(sum(SINR_10(i,:,t)));
             SINR_AVE(t)=10*log10(sum(SINR_total(:,t))/n);
             
             % BER_AVE(i,t)=0.5*erfc(sqrt(SINR_total(i,t)));
             % BER_10(t)=mean(BER_AVE(:,t),1);
           end
        end
      end
      BER_10(t)=erfc((sqrt(2*SINR_AVE(t))));
           %在迭代次数里用这段代码
%%
        % Update variables for next iteration
        channel_choice_current = channel_choice_new;
    end
    
    % Save results for this time step
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;
    channel_choice(:, :, t) = channel_choice_current;
    
    % Compute total energy efficiency for this time step
    EE_altopt(t) = compute_total_EE([P_RF_current; P_VLC_current], gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
 

end

% 修正 EE_altrow 和 EE_alt 的计算逻辑
for t=1:time_steps
    for i=1:n
EE_altrow(i,t) = sum(alpha(i,:,t)); % 按第二维（j）求和，结果为 n x time_steps
EE_alt = sum(EE_altrow, 1); % 按第一维（i）求和，结果为 1 x time_steps
    end
end

%% Function to compute total energy efficiency
function EE_total = compute_total_EE(x_opt, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n)
    P_RF_current = x_opt(1:n);
    P_VLC_current = x_opt(n+1:end);
    EE_total = 0;
    for i = 1:n
        for j = 1:n
            if i ~= j
                % Compute SINR and rate based on channel choice
                if channel_choice(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j) .* P_VLC_current) - gVLC(i, j) * P_VLC_current(j);
                    interference_VLC = max(interference_VLC, 1e-9); % 防止分母为零
                    SINR_VLC = gVLC(i, j) * P_VLC_current(j) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_current(j);  % 使用发送方j的功率
                else  % RF
                    interference_RF = sum(gRF(:, j) .* P_RF_current) - gRF(i, j) * P_RF_current(j);
                    interference_RF = max(interference_RF, 1e-9); % 防止分母为零
                    SINR_RF = gRF(i, j) * P_RF_current(j) / (interference_RF + noise_RF);
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF_current(j);  % 使用发送方j的功率
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

%% Function to compute non-linear constraints
function [c, ceq] = nonlinear_constraints(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n, R_min_RF, R_min_VLC, P_total)
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
                    % 修正: 转置 P_RF 使其与 gRF(i, :) 维度一致
                    interference_RF = sum(gRF(i, :) .* P_RF') - gRF(i, j) * P_RF(j);
                    interference_RF = max(interference_RF, 1e-9); % 防止分母为零
                    SINR_RF = gRF(i, j) * P_RF(j) / (interference_RF + noise_RF);
                    rate_RF = B_RF * log2(1 + SINR_RF);
                    if isnan(rate_RF) || isinf(rate_RF) || imag(rate_RF) ~= 0
                        rate_RF = 0; % 处理无效值
                    end
                    c(end+1) = R_min_RF - rate_RF;  % rate_RF >= R_min_RF
                else  % VLC
                    % 修正: 转置 P_VLC 使其与 gVLC(i, :) 维度一致
                    interference_VLC = sum(gVLC(i, :) .* P_VLC') - gVLC(i, j) * P_VLC(j);
                    interference_VLC = max(interference_VLC, 1e-9); % 防止分母为零
                    SINR_VLC = gVLC(i, j) * P_VLC(j) / (interference_VLC + noise_VLC);
                    rate_VLC = B_VLC * log2(1 + SINR_VLC);
                    if isnan(rate_VLC) || isinf(rate_VLC) || imag(rate_VLC) ~= 0
                        rate_VLC = 0; % 处理无效值
                    end
                    c(end+1) = R_min_VLC - rate_VLC;  % rate_VLC >= R_min_VLC
                end
            end
        end
    end
    
    % C3 & C4: Total power per vehicle
    for i = 1:n
        c(end+1) = (P_RF(i) + P_VLC(i)) - P_total;  % P_RF + P_VLC <= P_total
    end
end
