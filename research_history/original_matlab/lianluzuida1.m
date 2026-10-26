% 主程序：基于广义分布式Dinkelbach算法的功率分配优化
%链路最大 结果在EErow和EElianluzuida
% 参数设置
n = 5; % 车辆数量
maxIter = 100; % 最大迭代次数
epsilon = 1e-4; % 收敛精度
B_RF = 1000000; % RF带宽（单位：Hz）
B_VLC = 2000000; % VLC带宽（单位：Hz）
timeSteps = 10; % 时间步长

% 初始化功率分配和能耗效率
P_RF = zeros(n, n, timeSteps); % RF功率分配初始化
P_VLC = zeros(n, n, timeSteps); % VLC功率分配初始化
EE = zeros(n,n, timeSteps); % 每辆车的能耗效率初始化

% 噪声功率
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)

% RF 和 VLC 功率限制
P_RF_min = 0.1; P_RF_max = 30; % RF功率上下限
P_VLC_min = 20; P_VLC_max = 30; % VLC功率上下限

% 主循环：对每个时间步长进行优化
for t = 1:timeSteps
    fprintf('时间步长：%d\n', t);
    iter = 0;
    gap = inf;

    % 初始化通信方式选择矩阵：1表示VLC，0表示RF
    channel_choice = randi([0, 1], n, n); % 初始随机选择
    for i = 1:n
        channel_choice(i, i) = 0; % 自通信置零
    end

    while iter < maxIter && gap > epsilon
        iter = iter + 1;
        EE_prev = EE(:, t);

        % 固定 P，计算 R_i 和 P_i
        R_i = zeros(n, 1);
        P_i = zeros(n, 1);
        for i = 1:n
            total_rate = 0;
            total_power = 0;
            for j = 1:n
                if i ~= j
                    % 计算 SINR
                    if channel_choice(i, j) == 1 % VLC
                        interference_vlc = noise_VLC;
                        for k = 1:n
                            if k ~= i && k ~= j
                                P_VLC_total_k = sum(P_VLC(k, :, t)) - P_VLC(k, j, t);
                                interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                            end
                        end
                        SINR = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                        rate = B_VLC * log2(1 + SINR);
                        total_power = total_power + P_VLC(i, j, t);
                    else % RF
                        interference_rf = noise_RF;
                        for k = 1:n
                            if k ~= i && k ~= j
                                P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                                interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                            end
                        end
                        SINR = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                        rate = B_RF * log2(1 + SINR);
                        total_power = total_power + P_RF(i, j, t);
                    end
                    total_rate = total_rate + rate;
                else 
                    total_rate=0;
                end
            end
            R_i(i) = total_rate;
            P_i(i) = total_power;
        end

        % 更新辅助变量 alpha_i, beta_i
        alpha = zeros(n, 1);
        for i = 1:n
            if P_i(i) > 0
                alpha(i) = R_i(i) / P_i(i);
            else
                alpha(i) = 0;
            end
        end

        % 固定 alpha_i，优化 P_i
        % 定义优化变量 x = [P_RF_vec; P_VLC_vec]
        P_RF_vec = reshape(P_RF(:, :, t), n*n, 1);
        P_VLC_vec = reshape(P_VLC(:, :, t), n*n, 1);
        x0 = [P_RF_vec; P_VLC_vec];

        % 定义功率上下限
        lb = zeros(2*n*n, 1); % 初始化下限
        ub = zeros(2*n*n, 1); % 初始化上限
        for idx = 1:n*n
            [i, j] = ind2sub([n, n], idx);
            if i ~= j
                % VLC功率限制
                lb(n*n + idx) = P_VLC_min;
                ub(n*n + idx) = P_VLC_max;
                
                % RF功率限制
                lb(idx) = P_RF_min;
                ub(idx) = P_RF_max;
            else
                % 自通信功率为零
                ub(idx) = 0;
                ub(n*n + idx) = 0;
            end
        end

        % 调用优化函数
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
        [x_opt, ~, ~] = fmincon(@(x) objective(x, n, t, gRF, gVLC, B_RF, B_VLC, noise_RF, noise_VLC, channel_choice, alpha), ...
                                x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF(:, :, t) = reshape(x_opt(1:n*n), n, n);
        P_VLC(:, :, t) = reshape(x_opt(n*n+1:end), n, n);

        % 更新通信方式选择
        for i = 1:n
            for j = 1:n
                if i ~= j
                    interference_vlc = noise_VLC;
                    interference_rf = noise_RF;
                    for k = 1:n
                        if k ~= i && k ~= j
                            P_VLC_total_k = sum(P_VLC(k, :, t)) - P_VLC(k, j, t);
                            interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                            
                            P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                            interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                        end
                    end
                    SINR_VLC = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                    SINR_RF = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;

                    if SINR_VLC >= SINR_RF
                        channel_choice(i, j) = 1; % VLC
                    else
                        channel_choice(i, j) = 0; % RF
                    end
                end
            end
        end

        % 更新能耗效率 EE
        for i = 1:n
            total_rate = 0;
            total_power = 0;
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j) == 1 % VLC
                        interference_vlc = noise_VLC;
                        for k = 1:n
                            if k ~= i && k ~= j
                                P_VLC_total_k = sum(P_VLC(k, :, t)) - P_VLC(k, j, t);
                                interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                            end
                        end
                        SINR = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                        rate = B_VLC * log2(1 + SINR);
                        total_power = total_power + P_VLC(i, j, t);
                    else % RF
                        interference_rf = noise_RF;
                        for k = 1:n
                            if k ~= i && k ~= j
                                P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                                interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);

                            end
                        end
                        SINR = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                        rate = B_RF * log2(1 + SINR);
                        total_power = total_power + P_RF(i, j, t);
                    end
                    total_rate = total_rate + rate;
                else 
                    total_rate=0;
                end
                EE(i, j,t) = total_rate / total_power;
            
            end
           
        end

        % 检查收敛
        gap = max(abs(EE(i,j, t) - EE_prev));
    end
    EE(1,1,t)=0;
   
end
for t=1:timeSteps
for i=1:n
EErow(i,t)=sum(EE(i,:,t));

end
 EElianluzuida(t)=sum(EErow(:,t));
end
disp('优化后的能耗效率：');
disp(EE);

% 优化目标函数
function obj = objective(x, n, t, gRF, gVLC, B_RF, B_VLC, noise_RF, noise_VLC, channel_choice, alpha)
    P_RF_mat = reshape(x(1:n*n), n, n);
    P_VLC_mat = reshape(x(n*n+1:end), n, n);
    obj = 0;
    for i = 1:n
        total_rate = 0;
        total_power = 0;
        for j = 1:n
            if i ~= j
                if channel_choice(i, j) == 1 % VLC
                    interference_vlc = noise_VLC;
                    for k = 1:n
                        if k ~= i && k ~= j
                            P_VLC_total_k = sum(P_VLC_mat(k, :)) - P_VLC_mat(k, j);
                            interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                        end
                    end
                    SINR = (P_VLC_mat(i, j) * gVLC(i, j, t)) / interference_vlc;
                    rate = B_VLC * log2(1 + SINR);
                    total_power = total_power + P_VLC_mat(i, j);
                else % RF
                    interference_rf = noise_RF;
                    for k = 1:n
                        if k ~= i && k ~= j
                            P_RF_total_k = sum(P_RF_mat(k, :)) - P_RF_mat(k, j);
                            interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                        end
                    end
                    SINR = (P_RF_mat(i, j) * gRF(i, j, t)) / interference_rf;
                    rate = B_RF * log2(1 + SINR);
                    total_power = total_power + P_RF_mat(i, j);
                end
                total_rate = total_rate + rate;
            end
        end
        obj = obj - (total_rate - alpha(i) * total_power);
    end
end
