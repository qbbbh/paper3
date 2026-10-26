% 参数设置
num_vehicles = 5; % 车辆数量
time_steps = 100; % 仿真时间步长
A = [1 0.1; 0 1]; % 假设简单的运动模型，位置和速度
H = [1 0]; % 观测矩阵
Q = [0.01 0; 0 0.01]; % 过程噪声协方差矩阵
R = 0.1; % 测量噪声协方差

% 初始化状态和协方差矩阵
x = zeros(2, num_vehicles); % 初始状态，包括位置和速度
P = eye(2); % 初始协方差矩阵

% 存储估计结果
x_estimates = zeros(2, time_steps, num_vehicles);
h_estimates = zeros(time_steps, num_vehicles); % 存储信道增益估计

% 模拟真实信道增益数据（作为观测值）
h_actual = gVLC; % 假设真实信道增益有随机变化

% 卡尔曼滤波迭代估计
for t = 1:time_steps
    for v = 1:num_vehicles
        % 预测步骤
        x(:, v) = A * x(:, v); % 状态预测
        P = A * P * A' + Q; % 误差协方差预测

        % 观测到的信道增益
        y = h_actual(t, v);

        % 计算卡尔曼增益
        K = P * H' / (H * P * H' + R);

        % 更新状态和误差协方差
        x(:, v) = x(:, v) + K * (y - H * x(:, v));
        P = (eye(2) - K * H) * P;

        % 存储估计结果
        x_estimates(:, t, v) = x(:, v);
        h_estimates(t, v) = H * x(:, v); % 信道增益的估计值
    end
end

% 绘制信道增益估计结果
figure;
hold on;
for v = 1:num_vehicles
    plot(1:time_steps, h_actual(:, v), '--', 'DisplayName', ['Vehicle ' num2str(v) ' Actual']);
    plot(1:time_steps, h_estimates(:, v), 'DisplayName', ['Vehicle ' num2str(v) ' Estimated']);
end
xlabel('Time Step');
ylabel('Channel Gain');
legend;
title('VLC/RF Channel Gain Estimation using Kalman Filter');
hold off;
