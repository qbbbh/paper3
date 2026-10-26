%% Alternating Optimization for Power Allocation in RF/VLC Heterogeneous Network
%交替优化，结果在EE_alter1和EE_alterrow中 没有辅助变量  EE_alter(t)/EE_alter1(t)/EE_alter2(t);
% Parameters Initialization
n = 5;                 % Number of vehicles
time_steps = 500;      % Number of time steps
P_max = 30;            % Maximum power (W)
P_min = 0.1;           % Minimum power (W)
B_RF = 1000000; % RF带宽（单位：Hz）
B_VLC = 2000000; % VLC带宽（单位：Hz）
N0_RF = 8e-23;         % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;        % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF;   % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_VLC_min = 20;       % Minimum VLC power (W)
P_RF_min = 20;        % Minimum RF power (W)
max_iter = 50;         % Maximum iterations for alternating optimization
epsilon = 1e-4;        % Convergence threshold

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


% Initialize total energy efficiency record
EE_alter1 = zeros(1, time_steps);

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
        % Step 1: Fix channel choice, optimize power allocation
        % Define optimization variables and constraints
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
        
        % Optimization options
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
        
        % Objective function
        obj_fun = @(x) -compute_total_EE(x, gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
        
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
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
                    interference_RF1 = sum(gRF1(:, j, t) .* P_RF_current) - gRF1(i, j, t) * P_RF_current(i);
                    SINR_RF1(i, j, t) = gRF1(i, j, t) * P_RF_current(i) / (interference_RF1 + noise_RF);
                    % rate_RF = B_RF * log2(1 + SINR_RF(i, j, t));
                    % EE_RF = rate_RF / P_RF_current(i);
                    
                    % Compute SINR for VLC
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    interference_VLC1 = sum(gVLC1(:, j, t) .* P_VLC_current) - gVLC1(i, j, t) * P_VLC_current(i);
                    SINR_VLC1(i, j, t) = gVLC1(i, j, t) * P_VLC_current(i) / (interference_VLC1 + noise_VLC);

                    % rate_VLC = B_VLC * log2(1 + SINR_VLC(i, j, t));
                    % EE_VLC = rate_VLC / P_VLC_current(i);
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
        
        % Check for convergence
        if iter > 1 && max(abs(channel_choice_new(:) - channel_choice_current(:))) < epsilon ...
                && max(abs(P_RF_current - P_RF_prev)) < epsilon ...
                && max(abs(P_VLC_current - P_VLC_prev)) < epsilon
            fprintf('Converged at iteration %d\n', iter);
            IterationCountalt(t) = iter;
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
    EE_alter1(t) = compute_total_EE([P_RF_current; P_VLC_current], gRF(:, :, t), gVLC(:, :, t), channel_choice_current, B_RF, B_VLC, noise_RF, noise_VLC, n);
    EE_alter2(t) = compute_total_EE([P_RF_current; P_VLC_current], gRF1(:, :, t), gVLC1(:, :, t), channel_choice_current1, B_RF, B_VLC, noise_RF, noise_VLC, n);
    EE_alter(t)=EE_alter1(t)+EE_alter2(t);
    % Compute energy efficiency for each vehicle
    EE_vehicle = zeros(1, n);
    for i = 1:n
        total_eta = 0;
        for j = 1:n
            % if i ~= j
                if channel_choice_current(i, j) == 1  % VLC
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_current(i);
                else  % RF
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
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
                EE_vehicle(i) = total_eta;  % Energy efficiency of vehicle i
        end
    
    end
    EE_alterrow(:,t)=EE_vehicle;
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
            % if i ~= j
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
            % end
        end
        EE_total = EE_total + total_eta;
    end
end
%%代码解释：参数初始化：我们初始化所有必要的参数，包括车辆数量（n）、时间步长（time_steps）、
% 功率限制、带宽、噪声功率
% 以及交替优化的最大迭代次数。时间步骤上的主循环：主循环在每个时间步长 t 上运行。
% 每个时间步长的初始化：对于第一个时间步长，我们初始化功率分配（P_RF_current、P_VLC_current）
% 和信道选择（channel_choice_current）。对于后续时间步长，我们可以使用前一个时间步长的结果
% 作为初始猜测以加快收敛速度​​。交替优化迭代：
% 步骤 1：修复信道选择，优化功率分配我们定义优化变量（x0），
% 其中包括所有车辆的 RF 和 VLC 功率分配。功率分配的下限和上限根据
% P_RF_min、P_VLC_min 和 P_max 设置。我们定义目标函数，
% 该函数计算总能效的负数 (-EE_total)，因为 fmincon 会最小化目标函数。
% 我们使用 fmincon 进行优化，更新功率分配。
% 步骤 2：修复功率分配，优化通道选择对于每个通信对 (i, j)，
% 我们使用当前功率分配计算 RF 和 VLC 通道的能效。
% 然后，我们选择为该通信对提供更高能效的通道 (RF 或 VLC)。
% 收敛检查：我们检查迭代之间通道选择和功率分配的最大变化是否小于阈值 (epsilon)。
% 如果实现收敛，我们将跳出迭代循环。保存结果：我们保存当前时间步长的功率分配和通道选择。
% 我们计算并记录总能效。计算总能效的函数：函数 compute_total_EE 根据当前功率分配
% 和通道选择计算所有车辆的总能效。对于每辆车 i，我们将所有通信链路 (i, j) 的能量效率相加，
% 其中 j ≠ i。每条链路的能量效率计算为速率除以传输功率。
% 注意：信道增益（gRF 和 gVLC）：实际上，应根据车辆的位置和移动计算信道增益。
% 在此示例中，我们使用随机值来模拟信道增益。
% 优化选项：fmincon 的优化选项设置为使用“sqp”算法，该算法适用于带约束的非线性优化问题。
% 假设：我们假设其他车辆的干扰是根据它们当前的功率分配计算的。
% 车辆不会与自己通信（channel_choice_current(i, i) = 0）。

%其他建议：通道增益更新：为了使模拟更加逼真，您应该根据车辆的实际位置和速度在每个时间步骤
% 更新通道增益（gRF、gVLC）。功率分配初始化：您可能希望尝试不同的功率分配初始化策略，
% 以查看它如何影响收敛。收敛标准：调整收敛阈值（epsilon）和最大迭代次数（max_iter）
% 有助于控制计算时间和收敛精度之间的权衡。性能分析：运行模拟后，您可以通过绘制
% 总能效随时间的变化图以及功率分配和通道选择来分析算法的性能。
% 与其他算法的比较：要将此算法与其他算法（如广义 Dinkelbach 算法）的性能进行比较，
% 您可以实现这些算法并在相同条件下运行模拟。