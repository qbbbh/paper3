%% 参数初始化

n = 5;  % 车辆数量
time_steps = 10;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 10000000; % RF带宽（单位：Hz）
B_VLC = 20000000; % VLC带宽（单位：Hz）
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
EE_total9 = zeros(1, time_steps);

% 初始化辅助变量 alpha_i 和 beta_i
alpha_i = ones(n, 1);
beta_i = ones(n, 1);

% 信道增益初始化（根据实际情况更新）


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

    %% Step 2: 使用替代优化方法进行功率优化

    max_iter = 50;
    epsilon = 1e-4;

    % 初始功率分配
    P_RF_current = P_RF(:, t);
    P_VLC_current = P_VLC(:, t);

    for iter = 1:max_iter
        % Step 2.1: 固定功率，计算速率，并更新辅助变量 alpha_i 和 beta_i

        R_i = zeros(n, 1);
        P_i = P_RF_current + P_VLC_current;

        for i = 1:n
            total_rate = 0;
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j, t) == 1  % VLC
                        SINR = SINR_VLC(i, j, t);
                        rate = B_VLC * log2(1 + SINR);
                    else  % RF
                        SINR = SINR_RF(i, j, t);
                        rate = B_RF * log2(1 + SINR);
                    end
                    total_rate = total_rate + rate;
                end
            end
            R_i(i) = total_rate;
            if P_i(i) > 0
                alpha_i(i) = R_i(i) / P_i(i);
                beta_i(i) = 1 / P_i(i);
            else
                alpha_i(i) = 0;
                beta_i(i) = 0;
            end
        end

        % Step 2.2: 固定辅助变量，优化功率 P_i

        % 定义优化变量 x = [P_RF; P_VLC]
        x0 = [P_RF_current; P_VLC_current];

        % 功率上下限
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)];

        % 优化选项
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');

        % 优化求解
        [x_opt, fval] = fmincon(@(x) objective_function_alternate(x, alpha_i, beta_i, n), x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_prev = P_RF_current;
        P_VLC_prev = P_VLC_current;

        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);

        % 更新 SINR 和干扰
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 更新 RF 干扰和 SINR
                    interference_RF(i, j, t) = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF(i, j, t) + noise_RF);

                    % 更新 VLC 干扰和 SINR
                    interference_VLC(i, j, t) = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC(i, j, t) + noise_VLC);
                end
            end
        end

        % Step 2.3: 检查收敛性
        delta_P_RF = max(abs(P_RF_current - P_RF_prev));
        delta_P_VLC = max(abs(P_VLC_current - P_VLC_prev));

        if delta_P_RF < epsilon && delta_P_VLC < epsilon
            break;
        end
    end

    % 保存优化后的功率分配
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;

    % 计算并保存总能耗效率
    EE_total9(t) = sum(R_i) / sum(P_i);

    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, Total EE = %.4f\n', t, iter, EE_total9(t));

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

function obj = objective_function_alternate(x, alpha_i, beta_i, n)
    P_RF_opt = x(1:n);
    P_VLC_opt = x(n+1:end);

    P_i = P_RF_opt + P_VLC_opt;

    obj = sum(beta_i .* P_i - alpha_i);
end
