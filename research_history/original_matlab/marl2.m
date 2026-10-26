% === 参数初始化 ===
num_vehicles = 5;        % Number of vehicles
num_timesteps = 1000;    % Total simulation time (ms)
B_RF = 1e6;              % RF bandwidth (Hz)
B_VLC = 2e6;             % VLC bandwidth (Hz)
N0_RF = 8e-23;           % RF noise power spectral density (W/Hz)
N0_VLC = 1e-21;          % VLC noise power spectral density (W/Hz)
noise_RF = N0_RF * B_RF; % RF noise power (W)
noise_VLC = N0_VLC * B_VLC; % VLC noise power (W)
P_max_RF = 20;           % Maximum RF power (W)
P_max_VLC = 30;          % Maximum VLC power (W)
P_min_RF = 0.1;          % Minimum RF power (W)
P_min_VLC = 20;          % Minimum VLC power (W)
R_min_RF = 1e3;          % Minimum RF rate (bps)
R_min_VLC = 1e3;         % Minimum VLC rate (bps)

% Q-learning 参数
num_actions = 2; % RF or VLC (0: RF, 1: VLC)
Q_table = zeros(num_vehicles, num_actions); % Q 表初始化
alpha = 0.1; % 学习率
gamma = 0.9; % 折扣因子
epsilon = 0.1; % ε-greedy 策略中的探索率

% === 加载信道增益矩阵 (gRF 和 gVLC) ===
% load('channel_gain_RF.mat');  % RF channel gains (5x5x1000)
% load('channel_gain_VLC.mat'); % VLC channel gains (5x5x1000)

% 初始化变量
P_RF = ones(num_vehicles, num_vehicles) * P_max_RF / 2; % RF 功率初始值
P_VLC = ones(num_vehicles, num_vehicles) * P_max_VLC / 2; % VLC 功率初始值
SINR_matrix = zeros(num_vehicles, num_timesteps); % SINR 矩阵
EE_history = zeros(num_timesteps, 1); % 总 EE
avg_SINR_history = zeros(1, num_timesteps); % 平均 SINR
avg_BER_history = zeros(1, num_timesteps); % 平均 BER



% === 主仿真循环 ===
for t = 1:num_timesteps
    % 更新当前时刻的信道增益
    g_RF_current = gRF(:, :, t); % RF 信道增益
    g_VLC_current = gVLC(:, :, t); % VLC 信道增益

    for i = 1:num_vehicles
        % === ε-greedy 策略选择动作 ===
        if rand < epsilon
            % 探索：随机选择动作
            action = randi([1, 2]);
        else
            % 利用：选择当前 Q 值最大的动作
            [~, action] = max(Q_table(i, :));
        end

        % === 干扰计算和 SINR ===
        if action == 1 % RF 信道
            interference_RF = sum(g_RF_current(:, i) .* P_RF(:, i)) - g_RF_current(i, i) * P_RF(i, i);
            interference_RF = max(interference_RF, 1e-9); % 防止分母为零
            SINR_RF = g_RF_current(i, i) * P_RF(i, i) / (interference_RF + noise_RF);
            SINR_current = SINR_RF;
       elseif action == 2 % VLC 信道
            interference_VLC = sum(g_VLC_current(:, i) .* P_VLC(:, i)) - g_VLC_current(i, i) * P_VLC(i, i);
            interference_VLC = max(interference_VLC, 1e-9); % 防止分母为零
            SINR_VLC = g_VLC_current(i, i) * P_VLC(i, i) / (interference_VLC + noise_VLC);
            SINR_current = SINR_VLC;
        end

        % === 计算等效 SINR ===
        equ_SINR_VLC = 2 * ((1 + SINR_VLC / 3.3)^(B_VLC / B_RF) - 1);

        % === 奖励计算 ===
        % 奖励与 SINR 成正比，同时对干扰施加惩罚
        reward = SINR_current - log10(1 + interference_RF + interference_VLC);

        % === Q-learning 更新 ===
        max_next_Q = max(Q_table(i, :)); % 下一个状态的最大 Q 值
        Q_table(i, action) = Q_table(i, action ) + ...
                                 alpha * (reward + gamma * max_next_Q - Q_table(i, action ));

        % === 信道选择更新 ===
        if action == 0
            channel_choice_new(i, :) = 0; % 选择 RF
        else
            channel_choice_new(i, :) = 1; % 选择 VLC
        end

        % 记录 SINR
        SINR_matrix(i, t) = SINR_current;
    end

    % === 记录平均 SINR 和 BER ===
    avg_SINR_history(t) = mean(SINR_matrix(:, t)); % 平均 SINR
    avg_BER_history(t) = qfunc(sqrt(2 * avg_SINR_history(t))); % 平均 BER
end

disp('Q-learning 和等效 SINR 仿真完成！');

% === 结果展示 ===
% 绘制 SINR 曲线
figure;
plot(1:num_timesteps, avg_SINR_history, 'b-', 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Average SINR (dB)');
title('Average SINR over Time');
grid on;

% 绘制 BER 曲线
figure;
plot(1:num_timesteps, avg_BER_history, 'r-', 'LineWidth', 2);
xlabel('Time Step (ms)');
ylabel('Average BER');
title('Average Bit Error Rate over Time');
grid on;

% 绘制 Q 值收敛
figure;
plot(1:num_vehicles, Q_table(:, 1), 'b-', 'LineWidth', 2); hold on;
plot(1:num_vehicles, Q_table(:, 2), 'r--', 'LineWidth', 2);
xlabel('Vehicle Index');
ylabel('Q Value');
title('Q Value Convergence');
legend('RF', 'VLC');
grid on;
    % % Calculate SINR for each vehicle based on channel selection
    % for i = 1:num_vehicles
    %     if sum(x_RF(i, :)) > 0 % If RF is selected
    %         SINR_RF_i = sum(P_RF(i, :) .* g_RF_current(i, :) ./ noise_RF .* x_RF(i, :));
    %         SINR_matrix(i, t) = SINR_RF_i; % Store SINR for RF
    %     elseif sum(x_VLC(i, :)) > 0 % If VLC is selected
    %         SINR_VLC_i = sum(P_VLC(i, :) .* g_VLC_current(i, :) ./ noise_VLC .* x_VLC(i, :));
    %         SINR_matrix(i, t) = SINR_VLC_i; % Store SINR for VLC
    %     end
    % end
    % 
    % % Calculate average SINR across all vehicles at time t
    % avg_SINR_history(t) = mean(SINR_matrix(:, t));
    % 
    % % Calculate average BER based on average SINR
    % avg_BER_history(t) = erfc(sqrt(log(2 * avg_SINR_history(t)))); % Average BER at time tend
