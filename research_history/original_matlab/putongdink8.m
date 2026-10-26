%% 参数初始化

n = 5;  % 车辆数量
time_steps = 100;  % 仅考虑一个时间步
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 100e3;  % RF 带宽 (Hz)
B_VLC = 200e3; % VLC 带宽 (Hz)
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)
P_VLC_min = 0.1;  % VLC 最小功率
P_RF_min = 0.1;  % RF 最小功率

% 初始化功率分配
P_RF = P_max * ones(n, 1);
P_VLC = P_max * ones(n, 1);

% 信道增益初始化（根据实际情况更新）


% 初始化信道选择矩阵，1 表示 VLC，0 表示 RF
channel_choice = randi([0, 1], n, n);
for i = 1:n
    channel_choice(i, i) = 0;  % 车辆不与自己通信
end

% 初始化参数 lambda
lambda = 0;

% Dinkelbach 算法参数
max_iter = 100;
epsilon = 1e-4;

% 初始功率分配
P_RF_current = P_RF;
P_VLC_current = P_VLC;
for t = 1:time_steps
for iter = 1:max_iter
    % 固定 lambda，优化所有车辆的功率分配
    x0 = [P_RF_current; P_VLC_current];
    lb = [P_RF_min * ones(n, 1); P_VLC_min * ones(n, 1)];
    ub = [P_max * ones(n, 1); P_max * ones(n, 1)];
    
    options = optimoptions('fmincon','Display','off','Algorithm','sqp');
    [x_opt, fval] = fmincon(@(x) objective_function_dinkelbach_total(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, lambda, n), x0, [], [], [], [], lb, ub, [], options);
    
    P_RF_current = x_opt(1:n);
    P_VLC_current = x_opt(n+1:end);
    
    % 计算总速率和总功率
    R_total = 0;
    P_total = sum(P_RF_current) + sum(P_VLC_current);
    
    % 计算 SINR 和速率
    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算干扰和 SINR
                interference_RF = sum(gRF(:, j,t) .* P_RF_current) - gRF(i, j,t) * P_RF_current(i);
                SINR_RF = gRF(i, j) * P_RF_current(i) / (interference_RF + noise_RF);

                interference_VLC = sum(gVLC(:, j) .* P_VLC_current) - gVLC(i, j) * P_VLC_current(i);
                SINR_VLC = gVLC(i, j) * P_VLC_current(i) / (interference_VLC + noise_VLC);

                if channel_choice(i, j) == 1  % VLC
                    rate = B_VLC * log2(1 + SINR_VLC);
                else
                    rate = B_RF * log2(1 + SINR_RF);
                end
                R_total = R_total + rate;
            end
        end
    end
    
    % 检查收敛条件
    numerator = R_total - lambda * P_total;
    if abs(numerator) < epsilon
        break;
    end
    
    % 更新 lambda
    if P_total > 0
        lambda = R_total / P_total;
    else
        lambda = 0;
    end
    EEputong=R_total / P_total;
end
EE_putongdink(t)=EEputong;
end
% 输出结果
% fprintf('Converged after %d iterations\n', iter);
% fprintf('Total Rate: %.4f bps\n', R_total);
% fprintf('Total Power: %.4f W\n', P_total);
% fprintf('Energy Efficiency: %.4f bps/W\n', R_total / P_total);
% disp('Optimal P_RF:');
% disp(P_RF_current);
% disp('Optimal P_VLC:');
% disp(P_VLC_current);

%% 定义目标函数

function obj = objective_function_dinkelbach_total(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, lambda, n)
    P_RF_opt = x(1:n);
    P_VLC_opt = x(n+1:end);
    
    R_total = 0;
    P_total = sum(P_RF_opt) + sum(P_VLC_opt);
    
    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算干扰和 SINR
                interference_RF = sum(gRF(:, j) .* P_RF_opt) - gRF(i, j) * P_RF_opt(i);
                SINR_RF = gRF(i, j) * P_RF_opt(i) / (interference_RF + noise_RF);

                interference_VLC = sum(gVLC(:, j) .* P_VLC_opt) - gVLC(i, j) * P_VLC_opt(i);
                SINR_VLC = gVLC(i, j) * P_VLC_opt(i) / (interference_VLC + noise_VLC);

                if channel_choice(i, j) == 1  % VLC
                    rate = B_VLC * log2(1 + SINR_VLC);
                else
                    rate = B_RF * log2(1 + SINR_RF);
                end
                R_total = R_total + rate;
            end
        end
    end
    
    obj = -(R_total - lambda * P_total);
end
