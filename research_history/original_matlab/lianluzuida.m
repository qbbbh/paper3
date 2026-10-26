% 主程序：基于广义分布式Dinkelbach算法的功率分配优化  链路最大

% 参数设置
n = 5; % 车辆数量
maxIter = 100; % 最大迭代次数
epsilon = 1e-4; % 收敛精度
Pt = 150; % 每辆车的总发射功率（单位：瓦特）
B_RF = 10000000; % RF带宽（单位：Hz）
B_VLC = 20000000; % VLC带宽（单位：Hz）
P_vlc_min = 20; % VLC最小功率（用于照明）

timeSteps = 1000; % 时间步长

% 初始化功率分配和能耗效率
P_RF = ones(n, n, timeSteps) * (Pt / (2 * (n - 1))); % RF功率分配初始化
P_VLC = ones(n, n, timeSteps) * (Pt / (2 * (n - 1))); % VLC功率分配初始化
EE = zeros(n, timeSteps); % 每辆车的能耗效率初始化

% 噪声功率
% sigma2 = 1e-9; % 噪声功率（单位：瓦特）
% B_RF = 100e3;  % RF 带宽 (Hz)
% B_VLC = 200e3; % VLC 带宽 (Hz)
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)

% 主循环：对每个时间步长进行优化

for t = 1:timeSteps

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



    disp(['时间步长：', num2str(t)]);
    iter = 0;
    gap = inf;
    % 初始化通信方式选择矩阵：1表示VLC，0表示RF
    % channel_choice = ones(n, n); % 初始全为VLC
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
                        % 干扰计算
                        interference_vlc =noise_VLC;
                        for k = 1:n
                            if k ~= i && k ~= j
                                % 计算车辆k的VLC总发射功率（除去发送给j的部分）
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
                                % 计算车辆k的RF总发射功率（除去发送给j的部分）
                                P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                                interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                            end
                        end
                        SINR = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                        rate = B_RF * log2(1 + SINR);
                        total_power = total_power + P_RF(i, j, t);
                    end
                    total_rate = total_rate + rate;
                end
            end
            R_i(i) = total_rate;
            P_i(i) = total_power;
        end

        % 更新辅助变量 alpha_i, beta_i
        alpha = zeros(n, 1);
        beta = zeros(n, 1);
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
        % 定义优化变量 x = [P_RF_vec; P_VLC_vec], 大小为 2*n*(n-1) x 1
        P_RF_vec = reshape(P_RF(:, :, t), n*n, 1);
        P_VLC_vec = reshape(P_VLC(:, :, t), n*n, 1);
        x0 = [P_RF_vec; P_VLC_vec];

        % 定义功率上下限
        lb = zeros(2*n*n, 1);
        ub = zeros(2*n*n, 1);
        for idx = 1:n*n
            [i, j] = ind2sub([n, n], idx);
            if i ~= j
                % VLC功率下限为照明所需的最小功率
                lb(n*n + idx) = P_vlc_min;
                ub(n*n + idx) = Pt;
                if channel_choice(i, j) == 1 % VLC
                    ub(idx) = 0; % RF功率置零
                else % RF
                    ub(idx) = Pt; % RF功率上限
                end
            else
                % 对角线元素（自身），功率为零
                ub(idx) = 0;
                ub(n*n + idx) = 0;
            end
        end



        % 定义约束条件
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        % 每辆车的总功率约束
        for i = 1:n
            % 索引计算
            idx_rf = [];
            idx_vlc = [];
            for j = 1:n
                if i ~= j
                    idx_rf = [idx_rf; sub2ind([n, n], i, j)];
                    idx_vlc = [idx_vlc; n*n + sub2ind([n, n], i, j)];
                end
            end
            Aeq_i = zeros(1, 2*n*n);
            Aeq_i(idx_rf) = 1;
            Aeq_i(idx_vlc) = 1;
            Aeq = [Aeq; Aeq_i];
            beq = [beq; Pt];
        end

        % 调用优化函数
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

[x_opt, fval, exitflag] = fmincon(@(x) objective(x, n, t, gRF, gVLC,  P_RF,P_VLC,B_RF, B_VLC, noise_RF, noise_VLC, channel_choice, alpha), ...
    x0, A, b, Aeq, beq, lb, ub, ...
    @(x) rate_constraints(x, n, t, gRF, gVLC, B_RF, B_VLC, noise_RF, noise_VLC, channel_choice), ...
                                   options);

        % 更新功率分配
        P_RF(:, :, t) = reshape(x_opt(1:n*n), n, n);
        P_VLC(:, :, t) = reshape(x_opt(n*n+1:end), n, n);

        % 更新通信方式选择，根据新的SINR
        channel_choice_prev = channel_choice;
        for i = 1:n
            for j = 1:n
                if i ~= j
                    % 计算新的SINR
                    % VLC
                    interference_vlc = noise_VLC;
                    for k = 1:n
                        if k ~= i && k ~= j
                            % 计算车辆k的VLC总发射功率（除去发送给j的部分）
                            P_VLC_total_k = sum(P_VLC(k, :, t)) - P_VLC(k, j, t);
                            interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                        end
                    end
                    SINR_VLC = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                    % RF
                    interference_rf =  noise_RF;
                    for k = 1:n
                        if k ~= i && k ~= j
                            % 计算车辆k的RF总发射功率（除去发送给j的部分）
                            P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                            interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                        end
                    end
                    SINR_RF = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                    % 比较SINR，选择通信方式
                    if SINR_VLC >= SINR_RF
                        channel_choice(i, j) = 1; % VLC
                    else
                        channel_choice(i, j) = 0; % RF
                    end
                end
            end
        end

        % 计算能耗效率 EE
        for i = 1:n
            total_rate = 0;
            total_power = 0;
            for j = 1:n
                if i ~= j
                    if channel_choice(i, j) == 1 % VLC
                        % interference_vlc =  noise_VLC;
                        for k = 1:n
                            if k ~= i && k ~= j
                                % 计算车辆k的VLC总发射功率（除去发送给j的部分）
                                P_VLC_total_k = sum(P_VLC(k, :, t)) - P_VLC(k, j, t);
                                interference_vlc = noise_VLC + P_VLC_total_k * gVLC(k, j, t);
                            end
                        end
                        SINR = (P_VLC(i, j, t) * gVLC(i, j, t)) / interference_vlc;
                        rate = B_VLC * log2(1 + SINR);
                        total_power = total_power + P_VLC(i, j, t);
                    else % RF
                        % interference_rf =  noise_RF;
                        for k = 1:n
                            if k ~= i && k ~= j
                                % 计算车辆k的RF总发射功率（除去发送给j的部分）
                                P_RF_total_k = sum(P_RF(k, :, t)) - P_RF(k, j, t);
                                interference_rf = noise_RF + P_RF_total_k * gRF(k, j, t);
                            end
                        end
                        SINR = (P_RF(i, j, t) * gRF(i, j, t)) / interference_rf;
                        rate = B_RF * log2(1 + SINR);
                        total_power = total_power + P_RF(i, j, t);
                    end
                    total_rate = total_rate + rate;
                end
            end
            EE(i, t) = total_rate / total_power;
        end

        % 计算收敛性
        gap = max(abs(EE(:, t) - EE_prev));
        disp(['迭代次数：', num2str(iter), '，收敛差距：', num2str(gap)]);

        % 判断通信方式是否改变，如果没有改变且收敛差距小于阈值，则退出迭代
        while all(channel_choice == channel_choice_prev, 'all') && gap < epsilon
            break;
        end
    end
end
for t = 1:timeSteps
    EElianluzuida(t)=sum(EE(:,t));
end
% 结果展示
disp('优化后的能耗效率：');
disp(EE);


        % 定义优化目标函数
      function obj = objective(x, n, t, gRF, P_RF,P_VLC,gVLC, B_RF, B_VLC, noise_RF, noise_VLC, channel_choice, alpha)

 
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
            % x 包含 [P_RF_vec; P_VLC_vec]
            P_RF_mat = reshape(x(1:n*n), n, n);
            P_VLC_mat = reshape(x(n*n+1:end), n, n);
            obj = 0;
            for i = 1:n
                total_rate = 0;
                total_power = 0;
                for j = 1:n
                    if i ~= j
                        if channel_choice(i, j) == 1 % VLC
                            % 干扰计算
                            interference_vlc =  noise_VLC;
                            for k = 1:n
                                if k ~= i && k ~= j
                                    % 计算车辆k的VLC总发射功率（除去发送给j的部分）
                                    P_VLC_total_k = sum(P_VLC_mat(k, :)) - P_VLC_mat(k, j);
                                    interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                                end
                            end
                            SINR = (P_VLC_mat(i, j) * gVLC(i, j, t)) / interference_vlc;
                            rate = B_VLC * log2(1 + SINR);
                            total_power = total_power + P_VLC_mat(i, j);
                        else % RF
                            interference_rf =  noise_RF;
                            for k = 1:n
                                if k ~= i && k ~= j
                                    % 计算车辆k的RF总发射功率（除去发送给j的部分）
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
                obj = obj - (total_rate - alpha(i) * total_power); % 最小化负的目标函数等于最大化目标函数
            end
      end
      % 额外添加速率约束条件
function [c, ceq] = rate_constraints(x, n, t, gRF, gVLC, B_RF, B_VLC, noise_RF, noise_VLC, channel_choice)
    % x 包含 [P_RF_vec; P_VLC_vec]
    P_RF_mat = reshape(x(1:n*n), n, n);
    P_VLC_mat = reshape(x(n*n+1:end), n, n);
    
    % 初始化不等式约束，确保每个链路的速率不小于0.5 Mbps
    min_rate = 0.5 * 1e6; % 0.5 Mbps 转换为 bps
    c = []; % 不等式约束 c <= 0
    
    for i = 1:n
        for j = 1:n
            if i ~= j
                % 根据通道选择计算速率
                if channel_choice(i, j) == 1 % VLC
                    interference_vlc = noise_VLC;
                    for k = 1:n
                        if k ~= i && k ~= j
                            % 计算车辆k的VLC总发射功率（除去发送给j的部分）
                            P_VLC_total_k = sum(P_VLC_mat(k, :)) - P_VLC_mat(k, j);
                            interference_vlc = interference_vlc + P_VLC_total_k * gVLC(k, j, t);
                        end
                    end
                    SINR = (P_VLC_mat(i, j) * gVLC(i, j, t)) / interference_vlc;
                    rate = B_VLC * log2(1 + SINR);
                else % RF
                    interference_rf = noise_RF;
                    for k = 1:n
                        if k ~= i && k ~= j
                            % 计算车辆k的RF总发射功率（除去发送给j的部分）
                            P_RF_total_k = sum(P_RF_mat(k, :)) - P_RF_mat(k, j);
                            interference_rf = interference_rf + P_RF_total_k * gRF(k, j, t);
                        end
                    end
                    SINR = (P_RF_mat(i, j) * gRF(i, j, t)) / interference_rf;
                    rate = B_RF * log2(1 + SINR);
                end
                
                % 添加不等式约束，确保速率大于等于0.5 Mbps
                c = [c; min_rate - rate];
            end
        end
    end
    
    ceq = []; % 无等式约束
end