%% 参数初始化

n = 5;  % 车辆数量
time_steps = 1000;  % 仿真时间步数
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
% 初始化功率分配矩阵，大小为 n x n x time_steps
P_RF = P_max * ones(n, n, time_steps);   % P_RF(i,j,t): 车辆 i 发给车辆 j 的 RF 功率
P_VLC = P_max * ones(n, n, time_steps);  % P_VLC(i,j,t): 车辆 i 发给车辆 j 的 VLC 功率
% 初始化干扰和 SINR 矩阵
% interference_RF = zeros(n, n, time_steps);
% interference_VLC = zeros(n, n, time_steps);
% SINR_RF = zeros(n, n, time_steps);
% SINR_VLC = zeros(n, n, time_steps);

% 初始化信道选择矩阵，1 表示 VLC，0 表示 RF
channel_choice = zeros(n, n, time_steps);
equ_SINR_VLC = zeros(n, n, time_steps);

% 能耗效率记录
% EE_total6 = zeros(1, time_steps);

% 初始化参数 lambda_i_j
lambda = zeros(n, n);

% 信道增益初始化（根据实际情况更新）
% gRF = rand(n, n, time_steps);
% gVLC = rand(n, n, time_steps);

%% 主仿真过程

for t = 1:time_steps
    %% Step 1: 计算 SINR 并进行信道选择

    % for i = 1:n
    %     for j = 1:n
    %         if i ~= j
    %             % 计算 RF 信道的干扰和 SINR
    %             % interference_RF(i, j, t) = sum(gRF(:, j, t) .* P_RF(:, j, t)) - gRF(i, j, t) * P_RF(i, j, t);
    %             % SINR_RF(i, j, t) = gRF(i, j, t) * P_RF(i, j, t) / (interference_RF(i, j, t) + noise_RF);
    %             % 计算 VLC 信道的干扰和 SINR
    %             % interference_VLC(i, j, t) = sum(gVLC(:, j, t) .* P_VLC(:, j, t)) - gVLC(i, j, t) * P_VLC(i, j, t);
    %             % SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC(i, j, t) / (interference_VLC(i, j, t) + noise_VLC);
    % 
    %             % 计算等效 SINR
    %             equ_SINR_VLC(i, j, t) = 2 * ((1 + SINR_VLC(i, j, t) / 3.3)^(B_VLC / B_RF) - 1);
    %             % 信道选择
    %             if equ_SINR_VLC(i, j, t) > SINR_RF(i, j, t)
    %                 channel_choice(i, j, t) = 1;  % 选择 VLC
    %             else
    %                 channel_choice(i, j, t) = 0;  % 选择 RF
    %             end
    %         end
    %     end
    % end

    %% Step 2: 使用广义 Dinkelbach 算法进行功率优化

    max_iter = 50;
    epsilon = 1e-4;

    % 初始功率分配
    P_RF_current = P_RF(:, :, t);
    P_VLC_current = P_VLC(:, :, t);

    % 初始化参数 lambda_i_j
    lambda = zeros(n, n);

    for iter = 1:max_iter
        % 固定 lambda_i_j，优化 P_i_j

        % 定义优化变量 x = [P_RF_vars; P_VLC_vars]
        % 展开为向量，长度为 2 * n * (n-1)
        num_vars = n * (n - 1);
        x0 = [P_RF_current(~eye(n)); P_VLC_current(~eye(n))];

        % 根据 channel_choice 调整功率上下限
        lb = zeros(2 * num_vars, 1);
        ub = zeros(2 * num_vars, 1);

        idx = 1;
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 设置 RF 功率的上下限
                    if channel_choice(i, j, t) == 0  % 使用 RF
                        lb(idx) = P_min;
                        ub(idx) = P_max;
                    else  % 不使用 RF
                        lb(idx) = P_RF_min;
                        ub(idx) = P_RF_max;
                    end
                    idx = idx + 1;
                end
            end
        end

        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 设置 VLC 功率的上下限
                    if channel_choice(i, j, t) == 1  % 使用 VLC
                        lb(idx) = max(P_min, P_VLC_min);
                        ub(idx) = P_max;
                    else  % 不使用 VLC
                        lb(idx) = P_VLC_min;
                        ub(idx) = P_VLC_min;
                    end
                    idx = idx + 1;
                end
            end
        end

        % 优化选项
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

        % 优化求解
        [x_opt, fval] = fmincon(@(x) objective_function_link(x, gRF(:,:,t), gVLC(:,:,t), channel_choice(:,:,t), B_RF, B_VLC, noise_RF, noise_VLC, lambda, n), x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_vars = x_opt(1:num_vars);
        P_VLC_vars = x_opt(num_vars+1:end);

        % 将优化变量重构为矩阵形式
        P_RF_current(~eye(n)) = P_RF_vars;
        P_VLC_current(~eye(n)) = P_VLC_vars;

        % 更新 SINR 和干扰
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 更新 RF 干扰和 SINR
                    interference_RF(i, j, t) = sum(gRF(:, j, t) .* P_RF_current(:, j)) - gRF(i, j, t) * P_RF_current(i, j);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i, j) / (interference_RF(i, j, t) + noise_RF);

                    % 更新 VLC 干扰和 SINR
                    interference_VLC(i, j, t) = sum(gVLC(:, j, t) .* P_VLC_current(:, j)) - gVLC(i, j, t) * P_VLC_current(i, j);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i, j) / (interference_VLC(i, j, t) + noise_VLC);
                end
            end
        end

        % 计算每个链路的速率和功率，以及能耗效率
        EE_link = zeros(n, n);
        numerator = 0;
        for i = 1:n
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j, t) == 1  % VLC
                        SINR = SINR_VLC(i, j, t);
                        rate = B_VLC * log2(1 + SINR);
                        power = P_VLC_current(i, j);
                    else  % RF
                        SINR = SINR_RF(i, j, t);
                        rate = B_RF * log2(1 + SINR);
                        power = P_RF_current(i, j);
                    end
                    if power > 0
                        EE_link(i, j) = rate / power;
                    else
                        EE_link(i, j) = 0;
                    end
                    % 更新 lambda_i_j
                    lambda(i, j) = EE_link(i, j);

                    numerator = numerator + (rate - lambda(i, j) * power);
               
                end
            end
        end

        % 检查收敛性
        if abs(numerator) < epsilon
            break;
        end

    end

    % 保存优化后的功率分配
    P_RF(:,:,  t) = P_RF_current;
    P_VLC(:,:,  t) = P_VLC_current;
    P_RF_opt(:,  t) = P_RF_current(1,  :);
    P_VLC_opt(:,  t) = P_VLC_current(1,  :);

    % 计算并保存总能耗效率
   EE_total6(t) = sum(EE_link);
   EE_total61(t) = sum(EE_link(1,:));
   EE_total62(t) = sum(EE_link(2,:));
   EE_total63(t) = sum(EE_link(3,:));
   EE_total64(t) = sum(EE_link(4,:));
   EE_total65(t) = sum(EE_link(5,:));
    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, Total EE = %.4f\n', t, iter, EE_total6(t));

end





%% 定义目标函数

function obj = objective_function_link(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, lambda, n)
    num_vars = n * (n - 1);
    P_RF_vars = x(1:num_vars);
    P_VLC_vars = x(num_vars+1:end);
alpha = 0.005; 
    % 重构功率矩阵
    P_RF_opt = zeros(n, n);
    P_VLC_opt = zeros(n, n);

    idx = 1;
    for i = 1:n
        for j = 1:n
            if i ~= j
                P_RF_opt(i, j) = P_RF_vars(idx);
                idx = idx + 1;
            end
        end
    end

    for i = 1:n
        for j = 1:n
            if i ~= j
                P_VLC_opt(i, j) = P_VLC_vars(idx - num_vars);
                idx = idx + 1;
            end
        end
    end

    obj = 0;

    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算 RF 信道的干扰和 SINR
                interference_RF = sum(gRF(:, j) .* P_RF_opt(:, j)) - gRF(i, j) * P_RF_opt(i, j);
                SINR_RF = gRF(i, j) * P_RF_opt(i, j) / (interference_RF + noise_RF);

                % 计算 VLC 信道的干扰和 SINR
                interference_VLC = sum(gVLC(:, j) .* P_VLC_opt(:, j)) - gVLC(i, j) * P_VLC_opt(i, j);
                SINR_VLC = gVLC(i, j) * P_VLC_opt(i, j) / (interference_VLC + noise_VLC);

                % 根据信道选择计算速率和功率
                if channel_choice(i, j) == 1  % VLC
                    rate = B_VLC * log2(1 + SINR_VLC);
                    power = P_VLC_opt(i, j);
                else  % RF
                    rate = B_RF * log2(1 + SINR_RF);
                    power = P_RF_opt(i, j);
                end

                % 计算目标函数
               
                 obj = obj - (rate - alpha * lambda(i) * power);
            end
        end
    end
end
