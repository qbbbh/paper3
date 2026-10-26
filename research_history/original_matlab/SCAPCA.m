%% 参数初始化 
n = 5;  % 车辆数量
time_steps = 1001;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 10000000; % RF带宽（单位：Hz）
B_VLC = 20000000; % VLC带宽（单位：Hz）
N0_RF = 8e-23;   % RF噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC噪声功率 (W)
P_VLC_min = 20; % VLC 最小功率
P_RF_min = 0.1; % RF 最小功率
P_RF_max = 30; % RF 最大功率
P_VLC_max=30;
% 初始化功率分配
P_RF = P_max * ones(n, time_steps); % RF功率分配初始化
P_VLC = P_max * ones(n, time_steps); % VLC功率分配初始化

% 初始化信道增益矩阵 (随机初始化)
% gRF = rand(n, n, time_steps); % RF信道增益
% gVLC = rand(n, n, time_steps); % VLC信道增益

% 初始化信道选择矩阵，1表示VLC，0表示RF
channel_choice = zeros(n, n, time_steps);

% 初始化 SINR 矩阵
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);

% 总能效记录
EE_total = zeros(1, time_steps);

%% 主仿真过程

for t = 1:time_steps
    fprintf('时间步 %d 开始\n', t);

    % 更新信道增益 (此处为随机模型，可以替换为实际数据)
    % gRF(:,:,t) 和 gVLC(:,:,t) 已初始化

    % 初始化辅助变量 alpha 和 beta
    alpha_RF = zeros(n, 1);
    beta_RF = ones(n, 1);
    alpha_VLC = zeros(n, 1);
    beta_VLC = ones(n, 1);

    % 初始功率分配
    P_RF_current = P_RF(:, t);
    P_VLC_current = P_VLC(:, t);

    max_iter = 50; % 最大迭代次数
    epsilon = 1e-4; % 收敛条件

    for iter = 1:max_iter
        % Step 1: 计算 SINR
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % RF SINR
                    interference_RF_ij = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF_ij + noise_RF);

                    % VLC SINR
                    interference_VLC_ij = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC_ij + noise_VLC);
                end
            end
        end

        % Step 2: 更新辅助变量 alpha 和 beta
        for i = 1:n
            % RF 通信
            gamma_RF = SINR_RF(i, :, t); % 当前用户 RF SINR (向量)
            alpha_RF(i) = mean(gamma_RF ./ (1 + gamma_RF)); % 使用平均值计算 alpha
            beta_RF(i) = mean(log2(1 + gamma_RF) - (gamma_RF ./ (1 + gamma_RF)) .* log2(gamma_RF));

            % VLC 通信
            gamma_VLC = SINR_VLC(i, :, t); % 当前用户 VLC SINR (向量)
            alpha_VLC(i) = mean(gamma_VLC ./ (1 + gamma_VLC)); % 使用平均值计算 alpha
            beta_VLC(i) = mean(log2(1 + gamma_VLC) - (gamma_VLC ./ (1 + gamma_VLC)) .* log2(gamma_VLC));
        end

        % Step 3: 固定 alpha 和 beta，优化功率分配
        P_RF_current = max(P_RF_min, min(P_RF_max, P_RF_current)); % 保证初始值在合法范围内
        P_VLC_current = max(P_VLC_min, min(P_VLC_max, P_VLC_current)); % 同上
        x0 = [P_RF_current; P_VLC_current]; % 初始功率
        lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)]; % 下限
        ub = [P_max * ones(n, 1); P_max * ones(n, 1)]; % 上限
       % 检查 x0 是否在范围内
       if any(x0 < lb) || any(x0 > ub)
         error('Initial point x0 is out of bounds.');
       end
        % 优化选项
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

        % 调用 fmincon 优化功率分配
        [x_opt, ~] = fmincon(@(x) objective_function_scapca(x, gRF(:,:,t), gVLC(:,:,t), alpha_RF, beta_RF, alpha_VLC, beta_VLC, n), ...
                            x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);

        % 判断收敛性
        if norm(x_opt - x0) < epsilon
            fprintf('时间步 %d: 在第 %d 次迭代收敛\n', t, iter);
            break;
        end
    end

    % 保存功率分配结果
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;

    % 计算总能效
    EE_total(t) = sum(alpha_RF + alpha_VLC);
    fprintf('时间步 %d: 总能效 = %.4f\n', t, EE_total(t));
end

function obj = objective_function_scapca(x, gRF, gVLC, alpha_RF, beta_RF, alpha_VLC, beta_VLC, n)
    % x 包含 [P_RF; P_VLC]
    P_RF = x(1:n);
    P_VLC = x(n+1:end);

    obj = 0;

    for i = 1:n
        % RF 部分
        interference_RF = sum(gRF(:, i) .* P_RF) - gRF(i, i) * P_RF(i);
        if interference_RF + gRF(i, i) * P_RF(i) <= 0
            interference_RF = 1e-6; % 避免非法值
        end
        SINR_RF = gRF(i, i) * P_RF(i) / (interference_RF);
        if SINR_RF < 0 || isnan(SINR_RF) || isinf(SINR_RF)
            SINR_RF = 0; % 防护非法值
        end
        obj = obj + beta_RF(i) * P_RF(i) - alpha_RF(i) * log2(1 + SINR_RF);

        % VLC 部分
        interference_VLC = sum(gVLC(:, i) .* P_VLC) - gVLC(i, i) * P_VLC(i);
        if interference_VLC + gVLC(i, i) * P_VLC(i) <= 0
            interference_VLC = 1e-6; % 避免非法值
        end
        SINR_VLC = gVLC(i, i) * P_VLC(i) / (interference_VLC);
        if SINR_VLC < 0 || isnan(SINR_VLC) || isinf(SINR_VLC)
            SINR_VLC = 0; % 防护非法值
        end
        obj = obj + beta_VLC(i) * P_VLC(i) - alpha_VLC(i) * log2(1 + SINR_VLC);
    end
end
