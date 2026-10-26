%% 参数初始化

% 基本参数
n = 5;  % 车辆数量
time_steps = 1000;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 0.1; % 最小功率 (W)
B_RF = 100e3;  % RF带宽 (Hz)
B_VLC = 200e3; % VLC带宽 (Hz)
N0_RF = 8e-23;   % RF噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC噪声功率 (W)

% 初始化功率分配
P_RF = P_max * ones(n, time_steps);
P_VLC = P_max * ones(n, time_steps);

% 初始化信道增益矩阵 (假设随机变化，可根据实际模型替换)
% gRF = rand(n, n, time_steps);
% gVLC = rand(n, n, time_steps);

% 初始化SINR矩阵
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);
interference_RF= zeros(1, time_steps);
interference_VLC= zeros(1, time_steps);
% 初始化信道选择矩阵，1表示VLC，0表示RF
channel_choice = zeros(n, n, time_steps);
equ_SINR_VLC= zeros(n, n, time_steps);
% 能耗效率记录
EE_total = zeros(1, time_steps);

%% 主仿真过程

for t = 1:time_steps
    % 更新信道增益 (可根据实际模型更新)
    % 此处假设gRF和gVLC已更新
    
    %% Step 1: 计算SINR并进行信道选择
    
    total_rate = zeros(n, n);
    for i = 1:n
        for j = 1:n
            if i ~= j
                % 计算RF信道的干扰和SINR
                interference_RF(t) = sum(gRF(:, j, t) .* P_RF(:, t)) - gRF(i, j, t) * P_RF(i, t);
                SINR_RF(i, j, t) = gRF(i, j, t) * P_RF(i, t) / (interference_RF(t) + noise_RF);
                
                % 计算VLC信道的干扰和SINR
                interference_VLC(t) = sum(gVLC(:, j, t) .* P_VLC(:, t)) - gVLC(i, j, t) * P_VLC(i, t);
                SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC(i, t) / (interference_VLC(t) + noise_VLC);
                %计算VLC信噪比等效值
                equ_SINR_VLC(i, j, t)=2*((1+SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF)-1);


                % 信道选择
                if equ_SINR_VLC(i, j, t) >= SINR_RF(i, j, t)
                    channel_choice(i, j, t) = 1;  % 选择VLC
                    total_rate(i, j, t) = B_VLC * log2(1 + SINR_VLC(i, j, t));
                else
                    channel_choice(i, j, t) = 0;  % 选择RF
                    total_rate(i, j, t) = B_RF * log2(1 + SINR_RF(i, j, t));
                end
            end
        end
    end
    
    %% Step 2: 使用Dinkelbach算法进行功率优化
    
    % 初始化Dinkelbach参数
    theta = 0;
    epsilon = 1e-4;
    max_iter = 50;
    
    % 初始功率分配
    P_RF_current = P_RF(:, t);
    P_VLC_current = P_VLC(:, t);
    
    for iter = 1:max_iter
        % 计算总速率和总功耗
        total_power = sum(P_RF_current) + sum(P_VLC_current);
        total_rate_sum = sum(total_rate(:));
        
        % 计算能耗效率
        if total_power > 0
            theta_new = total_rate_sum / total_power;
        else
            theta_new = 0;
        end
        
        % 检查收敛
        if abs(theta_new - theta) < epsilon
            break;
        else
            theta = theta_new;
        end
        
        % 优化目标：最大化 total_rate_sum - theta * total_power
        % 在功率约束下进行优化
        % 使用 fmincon 进行优化
        
        % 优化变量：P_RF_opt, P_VLC_opt
        x0 = [P_RF_current; P_VLC_current];
        lb = P_min * ones(2*n,1);
        ub = P_max * ones(2*n,1);
        
        % 定义目标函数
        obj_fun = @(x) -objective_function(x, gRF(:,:,t), gVLC(:,:,t), channel_choice(:,:,t), B_RF, B_VLC, noise_RF, noise_VLC, theta, n);
        
        % 无约束条件
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        
        % 优化选项
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');
        
        % 优化求解
        [x_opt, fval] = fmincon(obj_fun, x0, A, b, Aeq, beq, lb, ub, [], options);
        
        % 更新功率分配
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);
        
        % 更新SINR和速率
        total_rate = zeros(n, n);
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 更新RF信道的干扰和SINR
                    interference_RF = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF + noise_RF);
                    
                    % 更新VLC信道的干扰和SINR
                    interference_VLC = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC + noise_VLC);
                    
                    % 信道选择保持不变，更新速率
                    if channel_choice(i, j, t) == 1
                        total_rate(i, j) = B_VLC * log2(1 + SINR_VLC(i, j, t));
                    else
                        total_rate(i, j) = B_RF * log2(1 + SINR_RF(i, j, t));
                    end
                end
            end
        end
    end
    
    % 保存优化后的功率分配
    P_RF(:, t) = P_RF_current;
    P_VLC(:, t) = P_VLC_current;
    
    % 计算并保存总能耗效率
    total_power = sum(P_RF_current) + sum(P_VLC_current);
    total_rate_sum = sum(total_rate(:));
    EE_total(t) = total_rate_sum / total_power;
    
    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, EE = %.4f\n', t, iter, EE_total(t));
end

%% 定义目标函数

function obj = objective_function(x, gRF, gVLC, channel_choice, B_RF, B_VLC, noise_RF, noise_VLC, theta, n)
    P_RF_opt = x(1:n);
    P_VLC_opt = x(n+1:end);
    
    total_rate = 0;
    total_power = sum(P_RF_opt) + sum(P_VLC_opt);
    
    for i = 1:n
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
    end
    
    obj = -(total_rate - theta * total_power);
end
