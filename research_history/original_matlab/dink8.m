%% 参数初始化 需要加辅助变量 并且迭代太慢
%普通dink  结果EE_total8  EE_8 行12345代表车辆12345
n = 5;  % 车辆数量
time_steps = 10;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 1000000; % RF带宽（单位：Hz）
B_VLC = 2000000; % VLC带宽（单位：Hz）
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)
P_VLC_min = 20;  % VLC 最小功率
P_RF_min = 0.1;  % RF 最小功率

% 初始化功率分配
P_RF = P_max * ones(n, time_steps);  % 每辆车的 RF 发射功率
P_VLC = P_max * ones(n, time_steps); % 每辆车的 VLC 发射功率

% 初始化干扰和 SINR 矩阵
interference_RF = zeros(n, n, time_steps + 1);
interference_VLC = zeros(n, n, time_steps + 1);
SINR_RF = zeros(n, n, time_steps + 1);
SINR_VLC = zeros(n, n, time_steps + 1);

% 初始化信道选择矩阵，1 表示 VLC，0 表示 RF
channel_choice = zeros(n, n, time_steps + 1);
equ_SINR_VLC = zeros(n, n, time_steps + 1);

% 能耗效率记录
EE_total8 = zeros(1, time_steps);

% 初始化参数 lambda_i
lambda = zeros(n, 1);


%% 主仿真过程

for t = 1:time_steps
    %% Step 1: 计算 SINR 并进行信道选择

    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算 RF 信道的干扰和 SINR
                interference_RF(i, j, t) = sum(gRF(:, j, t) .* P_RF(:, t)) - gRF(i, j, t) * P_RF(i, t);
                SINR_RF(i, j, t) = gRF(i, j, t) * P_RF(i, t) / (interference_RF(i, j, t) + noise_RF);

                % 计算 VLC 信道的干扰和 SINR
                interference_VLC(i, j, t) = sum(gVLC(:, j, t) .* P_VLC(:, t)) - gVLC(i, j, t) * P_VLC(i, t);
                SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC(i, t) / (interference_VLC(i, j, t) + noise_VLC);

                % 计算等效 SINR
                equ_SINR_VLC(i, j, t) = 2 * ((1 + SINR_VLC(i, j, t) / 3.3)^(B_VLC / B_RF) - 1);

                % 信道选择
                if equ_SINR_VLC(i, j, t) > SINR_RF(i, j, t)
                    channel_choice(i, j, t) = 1;  % 选择 VLC
                else
                    channel_choice(i, j, t) = 0;  % 选择 RF
                end
            end
        end
    end

    %% Step 2: 使用广义 Dinkelbach 算法进行功率优化

    max_iter = 50;
    epsilon = 1e-4;

    % 初始功率分配
    P_RF_current = P_RF(:, t);
    P_VLC_current = P_VLC(:, t);

    % 初始化参数 lambda_i
    lambda = zeros(n, 1);

    for iter = 1:max_iter
        % 固定 lambda_i，优化 P_i

        % 定义优化变量 x = [P_RF; P_VLC]
        x0 = [P_RF_current; P_VLC_current];

        % 根据 channel_choice 调整功率上下限
        lb = zeros(2*n, 1);
        ub = zeros(2*n, 1);

             for i = 1:n
            % 判断车辆 i 是否需要 RF 通信
            if any(channel_choice(i, :, t) == 0 & (1:n) ~= i)
                lb(i) = P_min;
                ub(i) = P_max;
            else
                lb(i) = P_RF_min;
                ub(i) = P_RF_min;
            end

            % 判断车辆 i 是否需要 VLC 通信
            if any(channel_choice(i, :, t) == 1 & (1:n) ~= i)
                lb(n + i) = max(P_min, P_VLC_min);
                ub(n + i) = P_max;
            else
                lb(n + i) = P_VLC_min;
                ub(n + i) = P_VLC_min;
            end
          end

       

        % 优化选项
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');

        % 优化求解
        [x_opt, fval] = fmincon(@(x) objective_function_linkwise(x, gRF(:,:,t), gVLC(:,:,t), channel_choice(:,:,t), B_RF, B_VLC, noise_RF, noise_VLC, lambda, n), x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);

        % 更新 SINR 和干扰
        for i = 1:n
            for j = 1:n
                if i ~= j
                    interference_RF(i, j, t) = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF(i, j, t) + noise_RF);

                    interference_VLC(i, j, t) = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC(i, j, t) + noise_VLC);
                end
            end
        end

        % 计算链路能耗效率和更新 lambda_i
        numerator = 0;
        for i = 1:n
            total_eta = 0;
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j, t) == 1  % VLC
                        SINR = SINR_VLC(i, j, t);
                        rate = B_VLC * log2(1 + SINR);
                        power = P_VLC_current(i);
                    else  % RF
                        SINR = SINR_RF(i, j, t);
                        rate = B_RF * log2(1 + SINR);
                        power = P_RF_current(i);
                    end
                    if power > 0
                        eta_ij = rate / power;
                    else
                        eta_ij = 0;
                    end
                    total_eta = total_eta+ eta_ij;
                    numerator = numerator + (rate - lambda(i) * power);
                end
            end
            % 更新 lambda_i
            lambda(i) = total_eta;
             EE_8(:,t)=lambda;
        end

        % 检查收敛性
        if abs(numerator) < epsilon
            break;
        end

    end

    % 保存优化后的功率分配
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;

    % 计算并保存总能耗效率
    EE_total8(t) = sum(lambda);
    
    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, Total EE = %.4f\n', t, iter, EE_total8(t));

    %% 更新 t+1 时刻的信道选择（如果需要）
    if t < time_steps
        % 更新 SINR 和干扰
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 更新 RF 干扰和 SINR
                    interference_RF(i, j, t+1) = sum(gRF(:, j, t+1) .* P_RF_current) - gRF(i, j, t+1) * P_RF_current(i);
                    SINR_RF(i, j, t+1) = gRF(i, j, t+1) * P_RF_current(i) / (interference_RF(i, j, t+1) + noise_RF);

                    % 更新 VLC 干扰和 SINR
                    interference_VLC(i, j, t+1) = sum(gVLC(:, j, t+1) .* P_VLC_current) - gVLC(i, j, t+1) * P_VLC_current(i);
                    SINR_VLC(i, j, t+1) = gVLC(i, j, t+1) * P_VLC_current(i) / (interference_VLC(i, j, t+1) + noise_VLC);

                    % 计算等效 SINR
                    equ_SINR_VLC(i, j, t+1) = 2 * ((1 + SINR_VLC(i, j, t+1) / 3.3)^(B_VLC / B_RF) - 1);

                    % 信道选择
                    if equ_SINR_VLC(i, j, t+1) > SINR_RF(i, j, t+1)
                        channel_choice(i, j, t+1) = 1;  % 选择 VLC
                    else
                        channel_choice(i, j, t+1) = 0;  % 选择 RF
                    end
                end
            end
        end
    end
   
end

%% 定义目标函数

function obj = objective_function_linkwise(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, lambda, n)
    P_RF_opt = x(1:n);
    P_VLC_opt = x(n+1:end);
alpha = 0.005;  % 权重参数，可根据需要调整

    obj = 0;

    for i = 1:n
        total_eta = 0;

        for j = 1:n
            if i ~= j
                % 计算 RF 信道的干扰和 SINR
                interference_RF = sum(gRF(:, j) .* P_RF_opt) - gRF(i, j) * P_RF_opt(i);
                SINR_RF = gRF(i, j) * P_RF_opt(i) / (interference_RF + noise_RF);

                % 计算 VLC 信道的干扰和 SINR
                interference_VLC = sum(gVLC(:, j) .* P_VLC_opt) - gVLC(i, j) * P_VLC_opt(i);
                SINR_VLC = gVLC(i, j) * P_VLC_opt(i) / (interference_VLC + noise_VLC);

                % 根据信道选择计算速率和功率
                if channel_choice(i, j) == 1  % VLC
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_opt(i);
                else  % RF
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF_opt(i);
                end

                if power > 0
                    eta_ij = rate / power;
                else
                    eta_ij = 0;
                end

                % 累加目标函数
              obj = obj - (rate - alpha * lambda(i) * power);
            end
        end
    end
end
