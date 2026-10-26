%% 参数初始化  12SCA论文 Energy-Minimization Task Offloading and Resource Allocation for Mobile Edge Computing in NOMAHeterogeneous Networks 
% 1+gamma算法 辅助变量
%先运行xindaozengyi结果在EE_total3和eerow3        EE_3(t)=EE_total3(t)+EE_total31(t); 
n = 5;  % 车辆数量
time_steps = 500;  % 仿真时间步数
P_max = 30;  % 最大功率 (W)
P_min = 1; % 最小功率 (W)
B_RF = 1000000; % RF带宽（单位：Hz）
B_VLC = 2000000; % VLC带宽（单位：Hz）
N0_RF = 8e-23;   % RF噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC噪声功率 (W)
P_VLC_min = 20;
P_RF_min = 1;

% 初始化功率分配
P_RF = P_max * ones(n, time_steps);
P_VLC = P_max * ones(n, time_steps);

% 初始化信道增益矩阵 (假设随机变化，可根据实际模型替换)

% 初始化信道选择矩阵，1表示VLC，0表示RF
channel_choice = zeros(n, n, time_steps);

% 初始化 SINR 矩阵
SINR_RF = zeros(n, n, time_steps);
SINR_VLC = zeros(n, n, time_steps);
equ_SINR_VLC = zeros(n, n, time_steps);

% 能耗效率记录
EE_total3 = zeros(1, time_steps);

%% 主仿真过程

for t = 2:time_steps
    % 更新信道增益 (可根据实际模型更新)
    % 此处假设gRF和gVLC已更新

    %% Step 1: 计算SINR并进行信道选择

    for i = 1:n
        for j = 1:n
            % if i ~= j
                % 计算RF信道的干扰和SINR
                interference_RF(i,j) = sum(gRF(:, j, t) .* P_RF(:, t)) - gRF(i, j, t) * P_RF(i, t);
                SINR_RF(i, j, t) = gRF(i, j, t) * P_RF(i, t) / (interference_RF(i,j) + noise_RF);

                % 计算VLC信道的干扰和SINR
                interference_VLC(i,j) = sum(gVLC(:, j, t) .* P_VLC(:, t)) - gVLC(i, j, t) * P_VLC(i, t);
                SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC(i, t) / (interference_VLC(i,j) + noise_VLC);

                % 计算VLC等效SINR
                equ_SINR_VLC(i, j, t) = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);

                % 信道选择
                if equ_SINR_VLC(i, j, t) > SINR_RF(i, j, t)
                    channel_choice(i, j, t) = 1;  % 选择VLC
                else
                    channel_choice(i, j, t) = 0;  % 选择RF
                end
                interference_RF1(i,j) = sum(gRF1(:, j, t) .* P_RF(:, t)) - gRF1(i, j, t) * P_RF(i, t);
                SINR_RF1(i, j, t) = gRF1(i, j, t) * P_RF(i, t) / (interference_RF1(i,j) + noise_RF);

                % 计算VLC信道的干扰和SINR
                interference_VLC1(i,j) = sum(gVLC1(:, j, t) .* P_VLC(:, t)) - gVLC1(i, j, t) * P_VLC(i, t);
                SINR_VLC1(i, j, t) = gVLC1(i, j, t) * P_VLC(i, t) / (interference_VLC1(i,j) + noise_VLC);

                % 计算VLC等效SINR
                equ_SINR_VLC1(i, j, t) = 2 * ((1 + SINR_VLC1(i, j, t)/3.3)^(B_VLC/B_RF) - 1);

                % 信道选择
                if equ_SINR_VLC1(i, j, t) > SINR_RF1(i, j, t)
                    channel_choice1(i, j, t) = 1;  % 选择VLC
                else
                    channel_choice1(i, j, t) = 0;  % 选择RF
                end
            % end
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
                % if i ~= j
                    if channel_choice(i, j, t) == 1  % VLC
                        SINR = SINR_VLC(i, j, t);
                        if i ~= j
                        SINR1 = SINR_VLC1(i, j, t);

                        % rate = B_VLC * log2(1 + SINR);
                        if SINR~= 0
                alpha(i,j) = SINR/(1+SINR);
                beta(i,j) = log2(1 + SINR) - (SINR ./ (1 + SINR)) .* log2(SINR);
               
                ee3(i,j,t)=(B_VLC*(beta(i,j) + (alpha(i,j) * log2(1 + SINR))))/P_i(j);
                        end
                       if SINR1~= 0  
                alpha1(i,j) = mean(SINR1/(1+SINR1));
                beta1(i,j) = mean(log2(1 + SINR1) - (SINR1 ./ (1 + SINR1)) .* log2(SINR1));
                ee31(i,j,t)=(B_VLC*(beta1(i,j)  + (alpha1(i,j) * log2(1 + SINR1))))/P_i(j);
                      end
                        end
                    end
                   
                if channel_choice(i, j, t) == 0. % RF
                SINR = SINR_RF(i, j, t);
                if SINR~= 0
                alpha(i,j) = mean(SINR/(1+SINR));
                beta(i,j) = mean(log2(1 + SINR) - (SINR ./ (1 + SINR)) .* log2(SINR)); 
                ee3(i,j,t)=(B_RF*(beta(i,j)  + (alpha(i,j) * log2(1 + SINR))))/P_i(j);
                end
                end

                if channel_choice(i, j, t) == 0 && i ~= j
                SINR1 = SINR_RF1(i, j, t);          
                if SINR1~= 0  
                alpha1(i,j) = mean(SINR1/(1+SINR1));
                beta1(i,j) = mean(log2(1 + SINR1) - (SINR1 ./ (1 + SINR1)) .* log2(SINR1));
                ee31(i,j,t)=(B_RF*(beta1(i,j)  + (alpha1(i,j) * log2(1 + SINR1))))/P_i(j);
                end
                end
            end
        end
                    % total_rate = total_rate + rate;
                 
                % end
           
            % R_i(i) = total_rate;



        % 固定 alpha_i, beta_i，优化 P_i

        % 定义优化变量 x = [P_RF; P_VLC]
        x0 = [P_RF_current; P_VLC_current];

        % 根据 channel_choice 调整功率上下限
        lb = zeros(2*n, 1);
        ub = zeros(2*n, 1);

        for i = 1:n
            % 判断车辆 i 是否需要 RF 通信
            use_RF = any(channel_choice(i, :, t) == 0 & (1:n) ~= i);
            % 判断车辆 i 是否需要 VLC 通信
            use_VLC = any(channel_choice(i, :, t) == 1 & (1:n) ~= i);

            % 设置 RF 功率上下限
            if use_RF
                lb(i) = P_min;
                ub(i) = P_max;
            else
                lb(i) = P_RF_min;
                ub(i) = P_RF_min;
            end

            % 设置 VLC 功率上下限
            if use_VLC
                lb(n + i) = max(P_VLC_min, P_min);
                ub(n + i) = P_max;
            else
                lb(n + i) = P_VLC_min;
                ub(n + i) = P_VLC_min;
            end
        end

        % 优化选项
        options = optimoptions('fmincon','Display','off','Algorithm','sqp');

        % 优化求解
        [x_opt, fval] = fmincon(@(x) objective_function_ao(x, gRF(:,:,t), gVLC(:,:,t), channel_choice(:,:,t), B_RF, B_VLC, noise_RF, noise_VLC, alpha, beta, n), x0, [], [], [], [], lb, ub, [], options);

        % 更新功率分配
        P_RF_current = x_opt(1:n);
        P_VLC_current = x_opt(n+1:end);

        % 更新SINR
        for i = 1:n
            for j = 1:n
                % if i ~= j
                    % 更新RF信道的干扰和SINR
                    interference_RF_ij = sum(gRF(:, j, t) .* P_RF_current) - gRF(i, j, t) * P_RF_current(i);
                    SINR_RF(i, j, t) = gRF(i, j, t) * P_RF_current(i) / (interference_RF_ij + noise_RF);

                    % 更新VLC信道的干扰和SINR
                    interference_VLC_ij = sum(gVLC(:, j, t) .* P_VLC_current) - gVLC(i, j, t) * P_VLC_current(i);
                    SINR_VLC(i, j, t) = gVLC(i, j, t) * P_VLC_current(i) / (interference_VLC_ij + noise_VLC);

                    % 计算VLC等效SINR
                    equ_SINR_VLC(i, j, t) = 2 * ((1 + SINR_VLC(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                    % 更新RF信道的干扰和SINR
                    interference_RF_ij1 = sum(gRF1(:, j, t) .* P_RF_current) - gRF1(i, j, t) * P_RF_current(i);
                    SINR_RF1(i, j, t) = gRF1(i, j, t) * P_RF_current(i) / (interference_RF_ij1 + noise_RF);

                    % 更新VLC信道的干扰和SINR
                    interference_VLC_ij1 = sum(gVLC1(:, j, t) .* P_VLC_current) - gVLC1(i, j, t) * P_VLC_current(i);
                    SINR_VLC1(i, j, t) = gVLC1(i, j, t) * P_VLC_current(i) / (interference_VLC_ij1 + noise_VLC);

                    % 计算VLC等效SINR
                    equ_SINR_VLC1(i, j, t) = 2 * ((1 + SINR_VLC1(i, j, t)/3.3)^(B_VLC/B_RF) - 1);
                % end
                % end
            end
        end

        % 更新信道选择
        for i = 1:n
            for j = 1:n
                % if i ~= j
                    if equ_SINR_VLC(i, j, t) > SINR_RF(i, j, t)
                        channel_choice(i, j, t) = 1;  % 选择VLC
                    else
                        channel_choice(i, j, t) = 0;  % 选择RF
                    end
                    if i ~= j
                    if equ_SINR_VLC1(i, j, t) > SINR_RF1(i, j, t)
                        channel_choice1(i, j, t) = 1;  % 选择VLC
                    else
                        channel_choice1(i, j, t) = 0;  % 选择RF
                    end
                    end
                % end
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
for i=1:n
  
            % 计算并保存总能耗效率
    EE_row3(i,t) = sum(ee3(i,:,t));
    EE_total3(t)=sum(EE_row3(:,t)); 
    EE_row31(i,t) = sum(ee31(i,:,t));
    EE_total31(t)=sum(EE_row31(:,t)); 
    EE_3(t)=EE_total3(t)+EE_total31(t); 
end
    IterationCount3(t) = iter;
    % 输出当前时间步的信息
    fprintf('Time step %d: Converged after %d iterations, Total EE = %.4f\n', t, iter, EE_total3(t));
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
            % if i ~= j
                % 计算RF信道的干扰和SINR
                interference_RF_ij = sum(gRF(:, j) .* P_RF_opt) - gRF(i, j) * P_RF_opt(i);
                SINR_RF = gRF(i, j) * P_RF_opt(i) / (interference_RF_ij + noise_RF);

                % 计算VLC信道的干扰和SINR
                interference_VLC_ij = sum(gVLC(:, j) .* P_VLC_opt) - gVLC(i, j) * P_VLC_opt(i);
                SINR_VLC = gVLC(i, j) * P_VLC_opt(i) / (interference_VLC_ij + noise_VLC);

                % 计算速率
                if channel_choice(i, j) == 1
                    rate = B_VLC * log2(1 + SINR_VLC);
                     obj = obj + beta(i) * P_i(i) - alpha(i) * log2(1 + SINR_VLC);
                else
                    rate = B_RF * log2(1 + SINR_RF);
                    obj = obj + beta(i) * P_i(i) - alpha(i) * log2(1 + SINR_RF);
                end
                total_rate = total_rate + rate;
            % end
        end

        % 计算目标函数
        % obj = obj + beta(i) * P_i(i) - alpha(i);
       
    end
end
