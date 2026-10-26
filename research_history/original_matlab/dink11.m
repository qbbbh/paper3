% 迭代算法优化 EE
% 每辆车逐链路优化功率分配

% 参数设置
n = 5; % 车辆数量
maxIter = 100; % 最大迭代次数
epsilon = 1e-4; % 收敛精度
B_RF = 1e6; % RF带宽（单位：Hz）
B_VLC = 2e6; % VLC带宽（单位：Hz）
timeSteps = 10; % 时间步长

% 噪声功率
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)

% RF 和 VLC 功率限制
P_RF_min = 0.1; P_RF_max = 30; % RF功率上下限
P_VLC_min = 20; P_VLC_max = 30; % VLC功率上下限

% 初始化功率分配和能效
P_RF = zeros(n, n, timeSteps); % RF功率初始化
P_VLC = zeros(n, n, timeSteps); % VLC功率初始化
EE11 = zeros(n, n, timeSteps); % 每条链路的能效

% 信道选择矩阵：1表示VLC，0表示RF
channel_choice = zeros(n, n, timeSteps);

% 主循环：对每个时间步长进行优化
for t = 1:timeSteps
    fprintf('时间步长：%d\n', t);

    % 初始化功率分配
    if t == 1
        P_RF(:, :, t) = P_RF_max * rand(n, n);
        P_VLC(:, :, t) = P_VLC_max * rand(n, n);
        for i = 1:n
            P_RF(i, i, t) = 0; % 自通信功率为 0
            P_VLC(i, i, t) = 0;
        end
    else
        % 从上一步功率分配初始化
        P_RF(:, :, t) = P_RF(:, :, t-1);
        P_VLC(:, :, t) = P_VLC(:, :, t-1);
    end

    % 初始化信道选择
    channel_choice(:, :, t) = randi([0, 1], n, n);
    for i = 1:n
        channel_choice(i, i, t) = 0; % 自通信无意义
    end

    % 迭代优化功率分配和信道选择
    iter = 0;
    gap = inf;
    while iter < maxIter && gap > epsilon
        iter = iter + 1;

        % 记录上一次的功率和 EE
        P_RF_prev = P_RF(:, :, t);
        P_VLC_prev = P_VLC(:, :, t);
        EE_prev = EE11(:, :, t);

        % 更新功率分配
        for i = 1:n
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j, t) == 1 % VLC
                        % 计算干扰和 SINR
                        interference_vlc = noise_VLC;
                        for k = 1:n
                            if k ~= i && k ~= j
                                interference_vlc = interference_vlc + P_VLC(k, j, t) * gVLC(k, j, t);
                            end
                        end
                        SINR_VLC = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                        rate_VLC = B_VLC * log2(1 + SINR_VLC);

                        % 更新 VLC 功率
                        P_VLC(i, j, t) = min(max(rate_VLC / gVLC(i, j, t), P_VLC_min), P_VLC_max);
                        EE11(i, j, t) = rate_VLC / P_VLC(i, j, t);
                    else
                        % 计算干扰和 SINR
                        interference_rf = noise_RF;
                        for k = 1:n
                            if k ~= i && k ~= j
                                interference_rf = interference_rf + P_RF(k, j, t) * gRF(k, j, t);
                            end
                        end
                        SINR_RF = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                        rate_RF = B_RF * log2(1 + SINR_RF);

                        % 更新 RF 功率
                        P_RF(i, j, t) = min(max(rate_RF / gRF(i, j, t), P_RF_min), P_RF_max);
                        EE11(i, j, t) = rate_RF / P_RF(i, j, t);
                    end
                end
            end
        end

        % 更新信道选择
        for i = 1:n
            for j = 1:n
                if i ~= j
                    interference_vlc = noise_VLC;
                    interference_rf = noise_RF;
                    for k = 1:n
                        if k ~= i && k ~= j
                            interference_vlc = interference_vlc + P_VLC(k, j, t) * gVLC(k, j, t);
                            interference_rf = interference_rf + P_RF(k, j, t) * gRF(k, j, t);
                        end
                    end
                    SINR_VLC = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                    SINR_RF = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;

                    if SINR_VLC > SINR_RF
                        channel_choice(i, j, t) = 1; % 选择 VLC
                    else
                        channel_choice(i, j, t) = 0; % 选择 RF
                    end
                end
            end
        end

        % 检查收敛
        gap = max(max(abs(EE11(:, :, t) - EE_prev)));
    end

    fprintf('时间步长 %d: 迭代次数 = %d, 收敛误差 = %.4e\n', t, iter, gap);
end

% 输出优化结果
disp('优化后的功率分配（RF 和 VLC）：');
disp(P_RF);
disp(P_VLC);
disp('优化后的能耗效率（EE）：');
disp(EE11);
