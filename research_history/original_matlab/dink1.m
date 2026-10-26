%% 参数初始化

n = 5;  % 车辆数量
time_steps = 1000;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 100000;  % RF带宽 (Hz)
B_VLC = 200000; % VLC带宽 (Hz)
N0_RF = 8e-23;   % RF噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC噪声功率 (W)

% 初始化功率分配
P_RF = P_max * ones(n, time_steps);
P_VLC = P_max * ones(n, time_steps);
interference_RF= zeros(1, time_steps);
interference_VLC= zeros(1, time_steps);
% 初始化信道增益矩阵 (假设随机变化，可根据实际模型替换)
% gRF = rand(n, n, time_steps);
% gVLC = rand(n, n, time_steps);

% 初始化SINR矩阵
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% 初始化信道选择矩阵，1表示VLC，0表示RF
channel_choice = zeros(n, n, time_steps);

% 能耗效率记录
EE_total = zeros(1, time_steps);

%% 主仿真过程

for t = 1:time_steps
    % 更新信道增益 (可根据实际模型更新)
    % 此处假设gRF和gVLC已更新

    %% Step 1: 计算SINR并进行信道选择

    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算RF信道的干扰和SINR
                interference_RF(t) = sum(gRF(:, j, t) .* P_RF(:, t)) - gRF(i, j, t) * P_RF(i, t);
                SINR_RF(i, j, t) = gRF(i, j, t) * P_RF(i, t) / (interference_RF(t) + noise_RF);

                % 计算VLC信道的干扰和SINR
                interference_VLC(t) = sum(gVLC(:, j, t) .* P_VLC(:, t)) - gVLC(i, j, t) * P_VLC(i, t);
                SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC(i, t) / (interference_VLC(t) + noise_VLC);

                % 信道选择
                if SINR_VLC(i, j, t) > SINR_RF(i, j, t)
                    channel_choice(i, j, t) = 1;  % 选择VLC
                else
                    channel_choice(i, j, t) = 0;  % 选择RF
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

    % 初始化辅助变量 alpha_i, beta_i
    alpha = ones(n, 1);
    beta = ones(n, 1);

    for iter = 1:max_iter
        % 固定 P，计算 R_i
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
        end

        % 更新辅助变量 alpha_i, beta_i
        for i = 1:n
            if P_i(i) > 0
                alpha(i) = R_i(i) / P_i(i);
                beta(i) = 1 / P_i(i);
            else
                alpha(i) = 0;
                beta(i) = 0;
            end
        end

        % 固定 alpha_i, beta_i，优化 P_i

        % 定义优化变量 x = [P_RF; P_VLC]
        x0 = [P_RF_current; P_VLC_current];
        lb = [P_min * ones(2*n,1)];
        ub = [P_max * ones(2*n,1)];

        % 定义目标函数
        obj_fun = @(x) objective_function_ao(x, gRF(:,:,t), gVLC(:,:,t), channel_choice(:,:,t), B_RF, B_VLC, noise_RF, noise_VLC, alpha, beta, n);

        % 优化选项
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');

        % 优化求解
        [x_opt, fval] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);

        % 更新SINR
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 更新RF信道的干扰和SINR
                    interference_RF(t) = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF(t) + noise_RF);

                    % 更新VLC信道的干扰和SINR
                    interference_VLC(t) = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC(t) + noise_VLC);
                end
            end
        end

        % 检查收敛性
        if norm(x_opt - x0) < epsilon
            break;
        end

        x0 = x_opt;
    end

    % 保存优化后的功率分配
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;

    % 计算并保存总能耗效率
    EE_total(t) = sum(alpha);

    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, Total EE = %.4f\n', t, iter, EE_total(t));

    %% Step 3: 根据信道决策调整功率分配

    for i = 1:n
        if P_VLC(i, t) < 20
            P_VLC(i, t) = 20;  % 确保 VLC 功率不低于 20W
        end
    end

    for i = 1:n
        if sum(channel_choice(i, :, t)) == n - 1  % 该车辆与其他车辆均选择 VLC
            P_RF(i, t) = P_min;  % RF 功率设为最小值
        end
        if sum(channel_choice(i, :, t)) == 0  % 该车辆与其他车辆均选择 RF
            P_VLC(i, t) = 20;  % VLC 功率设为 20W
        end
    end

end

%% 定义目标函数

function obj = objective_function_ao(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, alpha, beta, n)
    P_RF_opt = x(1:n);
    P_VLC_opt = x(n+1:end);

    P_i = P_RF_opt + P_VLC_opt;
    obj = 0;

    for i = 1:n
        % 计算 R_i
        total_rate = 0;
        for j = 1:n
            if i ~= j
                % 计算RF信道的干扰和SINR
                interference_RF = sum(gRF(:, j) .* P_RF_opt) - gRF(i, j) * P_RF_opt(i);
                SINR_RF = gRF(i, j) * P_RF_opt(i) / (interference_RF + noise_RF);

                % 计算VLC信道的干扰和SINR
                interference_VLC = sum(gVLC(:, j) .* P_VLC_opt) - gVLC(i, j) * P_VLC_opt(i);
                SINR_VLC = gVLC(i, j) * P_VLC_opt(i) / (interference_VLC + noise_VLC);

                % 计算速率
                if channel_choice(i, j) == 1
                    rate = B_VLC * log2(1 + SINR_VLC);
                else
                    rate = B_RF * log2(1 + SINR_RF);
                end
                total_rate = total_rate + rate;
            end
        end

        % 计算目标函数
        obj = obj + beta(i) * P_i(i) - alpha(i);
    end
end
