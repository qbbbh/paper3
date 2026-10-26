%% Alternating Optimization with Auxiliary Variables for Power Allocation in RF/VLC Heterogeneous Network
%结果在 EE_altrow 和EE_alt  EE_altopt对不上 EE_10(t)=EE_altopt(t)+EE_altopt1(t);
% Parameters Initialization
n = 5;                  % Number of vehicles
time_steps = 500;      % Number of time steps
P_max = 30;             % Maximum power (W)
P_min =1;            % Minimum power (W)
B_RF = 1000000;         % RF bandwidth (Hz)
B_VLC = 2000000;        % VLC bandwidth (Hz)
N0_RF = 8e-23;          % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;         % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF; % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_VLC_min = 20;         % Minimum VLC power (W)
P_RF_min = 1;          % Minimum RF power (W)
max_iter = 50;          % Maximum iterations for alternating optimization
epsilon = 1e-4;         % Convergence threshold

% Initialize power allocation matrices
P_RF = zeros(n, time_steps);
P_VLC = zeros(n, time_steps);

% Initialize channel selection matrices (1 for VLC, 0 for RF)
channel_choice = zeros(n, n, time_steps);

% Initialize SINR matrices
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% Initialize auxiliary variables
alpha = ones(n, n);  % Initial \(\alpha_{ij}\)
beta = ones(n, n);   % Initial \(\beta_{ij}\)

% Initialize total energy efficiency record
EE_altopt = zeros(1, time_steps);
IterationCount = zeros(1, time_steps); % Record iteration count for each time step

%% Main Loop over Time Steps
for t = 1:time_steps
    fprintf('Time Step %d\n', t);
    
    % Initialize power allocation for this time step
    if t == 1
        % Initial power allocation (random or set to maximum)
        P_RF_current = P_max * ones(n, 1);
        P_VLC_current = P_max * ones(n, 1);
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
        % Step 1: Fix channel choice, optimize power allocation using auxiliary variables
        % Define optimization variables and constraints
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
        
        % Optimization options
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
        
        % Objective function using auxiliary variables
        obj_fun = @(x) compute_auxiliary_objective(x, alpha, beta, n,channel_choice_current);
        
        % Perform optimization using fmincon
        [x_opt, ~] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);
        
        % Update power allocations
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);
        
        % Step 2: Fix power allocation, optimize channel choice
        % For each communication pair, decide RF or VLC based on energy efficiency
        channel_choice_new = zeros(n, n);
        for i = 1:n
            for j = 1:n
                % if i ~= j
                    % Compute SINR for RF
                    interference_RF = sum(gRF(i, :, t)' .* P_RF_current(i)) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
                    interference_RF1 = sum(gRF1(i, :, t)' .* P_RF_current(i)) - gRF1(i, j, t) * P_RF_current(i);
                    SINR_RF1(i, j, t) = gRF1(i, j, t) * P_RF_current(i) / (interference_RF1 + noise_RF);

                    
                    % Compute SINR for VLC
                    interference_VLC = sum(gVLC(i, :, t)' .* P_VLC_current(i)) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    interference_VLC1 = sum(gVLC1(i, :, t)' .* P_VLC_current(i)) - gVLC1(i, j, t) * P_VLC_current(i);
                    SINR_VLC1(i, j, t) = gVLC1(i, j, t) * P_VLC_current(i) / (interference_VLC1 + noise_VLC);
             
                    % 计算VLC等效SINR
                equ_SINR_VLC(i, j, t) = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                equ_SINR_VLC1(i, j, t) = 2 * ((1 + SINR_VLC1(i, j, t)/3.3)^(B_VLC/B_RF) - 1);

                % 信道选择
                if equ_SINR_VLC(i, j, t) > SINR_RF(i, j, t)
                    channel_choice_new(i, j) = 1;  % 选择VLC
                else
                    channel_choice_new(i, j) = 0;  % 选择RF
                end
                if equ_SINR_VLC1(i, j, t) > SINR_RF1(i, j, t)
                    channel_choice_new1(i, j) = 1;  % 选择VLC
                else
                    channel_choice_new1(i, j) = 0;  % 选择RF
                end
                   
                % end
            end
        end
        
        % Update auxiliary variables
        for i = 1:n
            for j = 1:n
                % if i ~= j
                    if channel_choice_new(i, j) == 1  % VLC
                        rate = B_VLC * log2(1 + SINR_VLC(i, j, t));
                        power = P_VLC_current(i);
                        SINR_10(i,j,t)=SINR_VLC(i, j, t);
                    else  % RF
                        rate = B_RF * log2(1 + SINR_RF(i, j, t));
                        power = P_RF_current(i);
                        SINR_10(i,j,t)=SINR_RF(i, j, t);
                    end
                    if power > 0
                        alpha(i, j,t) = rate / power;
                        beta(i, j) = 1 / power;
                    end
                % else
                %     alpha(i, j,t) =0;
                % end
            end

        end

        % Check for convergence
        if iter > 1 && max(abs(channel_choice_new(:) - channel_choice_current(:))) < epsilon ...
                && max(abs(P_RF_current - P_RF_prev)) < epsilon ...
                && max(abs(P_VLC_current - P_VLC_prev)) < epsilon
            fprintf('Converged at iteration %d\n', iter);
            IterationCount(t) = iter;
            break;
        end
        
        % Update variables for next iteration
        channel_choice_current = channel_choice_new;
        channel_choice_current1 = channel_choice_new1;
        P_RF_prev = P_RF_current;
        P_VLC_prev = P_VLC_current;



    end
    
    % Save results for this time step
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;
    channel_choice(:, :, t) = channel_choice_current;
    
    % Compute total energy efficiency for this time step
    EE_altopt(t) = compute_total_EE([P_RF_current; P_VLC_current], gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n,P_RF_current,P_VLC_current);
    EE_altopt1(t) = compute_total_EE([P_RF_prev; P_VLC_prev], gRF1(:, :, t), gVLC1(:, :, t), channel_choice_current1, B_RF, B_VLC, noise_RF, noise_VLC, n,P_RF_current,P_VLC_current);
EE_10(t)=EE_altopt(t)+EE_altopt1(t);
end
for t=1:time_steps
for i=1:n
for j=1:n
EE_altrow(i,t)=sum(alpha(i,:,t));


           end

end
end
EE_alt(t)=sum(EE_altrow(:,t));
end




%% Function to compute total energy efficiency
function EE_total = compute_total_EE(x_opt, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, n,P_RF_current,P_VLC_current)
    % P_RF_current = x_opt(1:n);
    % P_VLC_current = x_opt(n+1:end);
    EE_total = 0;
    for i = 1:n
        total_eta = 0;
        for j = 1:n
            % if i ~= j
                % Compute SINR and rate based on channel choice
                if channel_choice(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j) .* P_VLC_current) - gVLC(i, j) * P_VLC_current(i);
                    SINR_VLC = gVLC(i, j) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_current(i);
                else  % RF
                    interference_RF = sum(gRF(:, j) .* P_RF_current) - gRF(i, j) * P_RF_current(i);
                    SINR_RF = gRF(i, j) * P_RF_current(i) / (interference_RF + noise_RF);
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF_current(i);
                end
                if power > 0
                    eta_ij = rate / power;
                else
                    eta_ij = 0;
                end
                total_eta = total_eta + eta_ij;
            % end
        end
        EE_total = EE_total + total_eta;
    end
end

%% Function to compute auxiliary objective
function obj = compute_auxiliary_objective(x_opt, alpha, beta, n,channel_choice_current)
    P_RF_current = x_opt(1:n);
    P_VLC_current = x_opt(n+1:end);
    obj = 0;
    for i = 1:n
        for j = 1:n
            % if i ~= j
                if channel_choice_current==1
                obj = obj + beta(i, j) * (P_VLC_current(i) ) - alpha(i, j);
                else 
                obj = obj + beta(i, j) * (P_RF_current(i) ) - alpha(i, j);
                end
            % end
        end
    end
    % obj = abs(obj);
end