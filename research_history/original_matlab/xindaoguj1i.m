% 参数设置 
num_vehicles = 4; % 车辆数量
time_steps = 1001; % 仿真时间步长
dt = 0.1; % 时间步长
B_RF = 10000000; % RF带宽（单位：Hz）
B_VLC = 20000000; % VLC带宽（单位：Hz）
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_RF = N0_RF * B_RF;   % RF 噪声功率 (W)
noise_VLC = N0_VLC * B_VLC; % VLC 噪声功率 (W)
P_tx=25;
% 修改后的状态转移矩阵 A，考虑车辆的简单线性运动模型
A = [1 dt; 0 1]; % 位置和速度
H = [1 0]; % 观测矩阵，只观测位置

% 修改后的过程噪声和测量噪声协方差矩阵
Q = [0.001 0; 0 0.001]; % 过程噪声协方差，减小过程噪声
R = 0.05; % 测量噪声协方差，增加对测量噪声的考量

% 初始化状态和协方差矩阵
x = zeros(2, num_vehicles, num_vehicles); % 初始状态，包括位置和速度
P = eye(2) * 0.1; % 初始协方差矩阵，设置较小的初始不确定性

% 存储估计结果
x_estimated = zeros(2, time_steps, num_vehicles, num_vehicles);
h_estimated = zeros(num_vehicles, num_vehicles, time_steps); % 存储信道增益估计

% 提取有用的观测信道增益矩阵 h_actual
h_actual = zeros(num_vehicles, num_vehicles, time_steps);
for i = 1:num_vehicles
    for j = 1:num_vehicles
        if i ~= j
            h_actual(i, j, :) = gVLC(i, j, :); % 提取非对角线元素，表示每对车辆之间的信道增益
        end
    end
end

% 卡尔曼滤波迭代估计
for t = 1:time_steps
    for i = 1:num_vehicles
        for j = 1:num_vehicles
            if i ~= j % 只对不同车辆之间的信道增益进行估计
                % 预测步骤
                x(:, i, j) = A * x(:, i, j); % 状态预测
                P = A * P * A' + Q; % 误差协方差预测

                % 观测到的信道增益
                y = h_actual(i, j, t);

                % 计算卡尔曼增益
                K = P * H' / (H * P * H' + R);

                % 更新状态和误差协方差
                x(:, i, j) = x(:, i, j) + K * (y - H * x(:, i, j));
                P = (eye(2) - K * H) * P;

                % 存储估计结果
                x_estimated(:, t, i, j) = x(:, i, j);
                h_estimated(i, j, t) = H * x(:, i, j); % 信道增益的估计值

        % 实际信道增益的 SINR 计算
        signal_power_actual(i,j,t) = P_tx * h_actual(i,j,t); % 目标信号功率
        interference_power_actual(i,j,t) = sum(P_tx * h_actual(i,:,t)) - signal_power_actual(i,j,t); % 干扰功率
        SINR_actual(i,j,t) = signal_power_actual(i,j,t) / (noise_VLC + interference_power_actual(i,j,t));

        % 估计信道增益的 SINR 计算
        signal_power_estimated(i,j,t) = P_tx * h_estimated(i, j, t); % 目标信号功率
        interference_power_estimated(i,j,t) = sum(P_tx * h_estimated(i,:,t)) - signal_power_estimated(i,j,t); % 干扰功率
        SINR_estimated(i,j,t) = signal_power_estimated(i,j,t) / (noise_VLC + interference_power_estimated(i,j,t));

        % signal_power_estimated = P_tx * h_s_estimated(s); % 目标信号功率
        % interference_power_estimated = sum(P_tx * h_s_estimated) - signal_power_estimated; % 干扰功率
        % SINR_estimated(s, t) = signal_power_estimated / (noise_power + interference_power_estimated);
            end
        end
    end
end

% 绘制信道增益估计结果
figure;
hold on;
for i = 1:num_vehicles
    for j = 1:num_vehicles
        if i ~= j
            % 绘制实际观测值
            plot(1:time_steps, squeeze(h_actual(i, j, :)), '--', 'DisplayName', ['Vehicle ' num2str(i) ' to ' num2str(j) ' Actual']);
            % 绘制估计值
            plot(1:time_steps, squeeze(h_estimated(i, j, :)), 'DisplayName', ['Vehicle ' num2str(i) ' to ' num2str(j) ' Estimated']);
        end
    end
end
xlabel('Time Step');
ylabel('Channel Gain');
legend('show'); % 显示图例
title('VLC Channel Gain Estimation between Vehicles using Kalman Filter');



figure;
hold on;
for i = 1:num_vehicles
    for j = 1:num_vehicles
        if i ~= j
    plot(1:time_steps,squeeze( SINR_actual(i,j, :)), '--', 'DisplayName', ['Vehicle ' num2str(i) ' Actual SINR']);
    plot(1:time_steps, squeeze( SINR_estimated(i,j, :)), 'DisplayName', ['Vehicle ' num2str(i) ' Estimated SINR']);
        end
    end
end
hold off;
