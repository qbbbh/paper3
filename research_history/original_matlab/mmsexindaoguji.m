% 参数设置 保存结果在h_mmse和y_observed
num_vehicles = 5; % 车辆数量
time_steps = 1001; % 仿真时间步长
P_tx = 1; % 假设每辆车的传输功率为 1（可以根据实际场景调整）
B_VLC = 2e6; % VLC带宽（单位：Hz）
N0_RF = 8e-23;   % RF 噪声功率谱密度 (W/Hz)
N0_VLC = 1e-21;  % VLC 噪声功率谱密度 (W/Hz)
noise_power =N0_VLC*B_VLC; % 噪声功率
% noise_power =0.0000001; % 噪声功率
% 假设的信道方差（可以根据实际场景调整）
% gVLCLOS13_mean=mean(gVLCLOS13);
% gVLCLOS13_d=gVLCLOS13-gVLCLOS13_mean;
% gVLCLOS13_sqd=gVLCLOS13_d.^2;
% gVLCLOS13_sumsqd=sum(gVLCLOS13_sqd);
% gVLCLOS13_v=gVLCLOS13_sumsqd/(length(gVLCLOS13)-1);
gVLCLOS12_var=var(gVLCLOS13);gVLCLOS13_var=var(gVLCLOS13);gVLCLOS14_var=var(gVLCLOS14);gVLCLOS15_var=var(gVLCLOS15);
gVLCLOS23_var=var(gVLCLOS23);gVLCLOS24_var=var(gVLCLOS24);gVLCLOS25_var=var(gVLCLOS25);
gVLCLOS34_var=var(gVLCLOS34);gVLCLOS35_var=var(gVLCLOS35);
gVLCLOS45_var=var(gVLCLOS45);
% gVLCLOS12_mean=mean(gVLCLOS12);
sigma_h2=0.5;
sigma_h2_1 = mean(gVLCLOS12_var+gVLCLOS13_var+gVLCLOS13_var+gVLCLOS14_var); % 车辆1的信道的先验方差
sigma_h2_2 = mean(gVLCLOS12_var+gVLCLOS23_var+gVLCLOS24_var+gVLCLOS25_var); % 车辆1的信道的先验方差
sigma_h2_3 = mean(gVLCLOS13_var+gVLCLOS23_var+gVLCLOS34_var+gVLCLOS35_var); % 车辆1的信道的先验方差
sigma_h2_4 = mean(gVLCLOS14_var+gVLCLOS24_var+gVLCLOS34_var+gVLCLOS45_var); % 车辆1的信道的先验方差
sigma_h2_5 = mean(gVLCLOS15_var+gVLCLOS25_var+gVLCLOS35_var+gVLCLOS45_var); % 车辆1的信道的先验方差
sigma_n2 = noise_power; % 测量噪声的方差


% 存储估计结果
h_estimates_mmse = zeros(num_vehicles, num_vehicles, time_steps); % 存储MMSE信道增益估计
h_actual = zeros(num_vehicles, num_vehicles, time_steps);
% 初始化 SINR 结果存储矩阵
SINR_actual = zeros(num_vehicles,num_vehicles,time_steps);
SINR_mmse = zeros(num_vehicles,num_vehicles, time_steps);
% 初始化状态和协方差矩阵
x_estimates = zeros(1, time_steps, num_vehicles); % 存储卡尔曼滤波器估计的信道增益

h_actual_1=[gVLCLOS12;gVLCLOS13;gVLCLOS14;gVLCLOS15];
h_actual_2=[gVLCLOS21;gVLCLOS23;gVLCLOS24;gVLCLOS25];
h_actual_3=[gVLCLOS31;gVLCLOS32;gVLCLOS34;gVLCLOS35];
h_actual_4=[gVLCLOS41;gVLCLOS42;gVLCLOS43;gVLCLOS45];
h_actual_5=[gVLCLOS51;gVLCLOS52;gVLCLOS53;gVLCLOS54];
%先算车辆1的
% 卡尔曼滤波器参数设置
A = 1; % 简单的动态模型
% H = 1; % 观测矩阵
Q = 0.1; % 过程噪声协方差
R = 0.95; % 测量噪声协方差
P = 0.5; % 初始协方差矩阵

for i=1:4
    
 for t = 2:time_steps
       x_k_1 = h_actual_1(i, t-1); % 假设初始状态为真实信道值
       noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声
       poisson_noise = sqrt(sigma_n2) * (rand(1, 1) - 0.5); % 泊松噪声，模拟光信号的散粒效应
       % random_offset = 0.1 * (rand(1, 1) - 0.5); % 随机偏移，模拟非线性噪声     
       % % 使用对数正态分布生成噪声
       % noise = lognrnd(0, sqrt(sigma_n2), [1, 1, time_steps]) ;
       
       y_observed_1(i,t) = h_actual_1(i,t)+noise(t) ; % 添加噪声到观测信道增益
       y_observed_1(i,t) = max(y_observed_1(i, t), 1e-11); % 将负值截断为较小的正数
       % 观测到的信道增益（带噪声）
       y_1 = y_observed_1(i, t);
        % 卡尔曼预测步骤
        x_pred = A * x_k_1;
        P_pred = A * P * A' + Q;
        H=h_actual_1(i,t);
        % 卡尔曼更新步骤
        K = P_pred * H' / (H * P_pred * H' + R);
        x_k_1 = x_pred + K * (y_1- H * x_pred);
        P = (1 - K * H) * P_pred;

        % 存储卡尔曼滤波后的信道增益估计
        x_estimates_1(t, i) = x_k_1;

% 使用卡尔曼滤波后的信道增益进行MMSE估计
h_estimates_mmse_1(i, t) = (sigma_h2_1 / (sigma_h2_1 + sigma_n2)) * x_estimates_1(t, i);

 end
end
%车辆2的
for i=1:4  
 for t = 2:time_steps
     x_k_2 = h_actual_2(i, t-1); % 假设初始状态为真实信道值
       noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声

       y_observed_2(i,t) = h_actual_2(i,t)+noise(t); % 添加噪声到观测信道增益
       y_observed_2(i,t) = max(y_observed_2(i, t), 1e-11); % 将负值截断为较小的正数
       % 观测到的信道增益（带噪声）
       y_2 = y_observed_2(i, t);
        % 卡尔曼预测步骤
        x_pred = A * x_k_2;
        P_pred = A * P * A' + Q;
H=h_actual_2(i,t);
        % 卡尔曼更新步骤
        K = P_pred * H' / (H * P_pred * H' + R);
        x_k_2 = x_pred + K * (y_2 - H * x_pred);
        P = (1 - K * H) * P_pred;

        % 存储卡尔曼滤波后的信道增益估计
        x_estimates_2(t, i) = x_k_2;

 % 使用卡尔曼滤波后的信道增益进行MMSE估计
h_estimates_mmse_2(i, t) = (sigma_h2_1 / (sigma_h2_1 + sigma_n2)) * x_estimates_2(t, i);
 end
end
% 车辆3的
for i=1:4
    
 for t = 2:time_steps
     x_k_3 = h_actual_3(i, t-1); % 假设初始状态为真实信道值
       noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声
       % % 使用对数正态分布生成噪声
       % noise = lognrnd(0, sqrt(sigma_n2), [1, 1, time_steps]) ;
       y_observed_3(i,t) = h_actual_3(i,t)+noise(t); % 添加噪声到观测信道增益
       y_observed_3(i,t) = max(y_observed_3(i, t), 1e-11); % 将负值截断为较小的正数
       % 观测到的信道增益（带噪声）
       y_3 = y_observed_3(i, t);
               % 卡尔曼预测步骤
        x_pred = A * x_k_3;
        P_pred = A * P * A' + Q;
H=h_actual_3(i,t);
        % 卡尔曼更新步骤
        K = P_pred * H' / (H * P_pred * H' + R);
        x_k_3 = x_pred + K * (y_3 - H * x_pred);
        P = (1 - K * H) * P_pred;

        % 存储卡尔曼滤波后的信道增益估计
        x_estimates_3(t, i) = x_k_3;

 % 使用卡尔曼滤波后的信道增益进行MMSE估计
h_estimates_mmse_3(i, t) = (sigma_h2_1 / (sigma_h2_1 + sigma_n2)) * x_estimates_3(t, i);

 end
end
%车辆4的
for i=1:4
     % x_k_4 = h_actual_4(i, 1); % 假设初始状态为真实信道值
 for t = 2:time_steps
     x_k_4 = h_actual_4(i, t-1); % 假设初始状态为真实信道值
       noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声
       % % 使用对数正态分布生成噪声
       % noise = lognrnd(0, sqrt(sigma_n2), [1, 1, time_steps]) ;
       y_observed_4(i,t) = h_actual_4(i,t)+noise(t); % 添加噪声到观测信道增益
       y_observed_4(i,t) = max(y_observed_4(i, t), 1e-11); % 将负值截断为较小的正数
       % 观测到的信道增益（带噪声）
       y_4 = y_observed_4(i, t);
                      % 卡尔曼预测步骤
        x_pred = A * x_k_4;
        P_pred = A * P * A' + Q;
H=h_actual_4(i,t);
        % 卡尔曼更新步骤
        K = P_pred * H' / (H * P_pred * H' + R);
        x_k_4 = x_pred + K * (y_4 - H * x_pred);
        P = (1 - K * H) * P_pred;

        % 存储卡尔曼滤波后的信道增益估计
        x_estimates_4(t, i) = x_k_4;
 % 使用卡尔曼滤波后的信道增益进行MMSE估计
h_estimates_mmse_4(i, t) = (sigma_h2_1 / (sigma_h2_1 + sigma_n2)) * x_estimates_4(t, i);

 end
end

%车辆5的
for i=1:4
    % x_k_5 = h_actual_5(i, 1); % 假设初始状态为真实信道值
 for t = 2:time_steps
     x_k_5 = h_actual_5(i, t-1); % 假设初始状态为真实信道值
       noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声
       % % 使用对数正态分布生成噪声
       % noise = lognrnd(0, sqrt(sigma_n2), [1, 1, time_steps]) ;
       y_observed_5(i,t) = h_actual_5(i,t)+noise(t); % 添加噪声到观测信道增益
       y_observed_5(i,t) = max(y_observed_5(i, t), 1e-11); % 将负值截断为较小的正数
       % 观测到的信道增益（带噪声）
       y_5 = y_observed_5(i, t);
                             % 卡尔曼预测步骤
        x_pred = A * x_k_5;
        P_pred = A * P * A' + Q;
H=h_actual_5(i,t);
        % 卡尔曼更新步骤
        K = P_pred * H' / (H * P_pred * H' + R);
        x_k_5 = x_pred + K * (y_5 - H * x_pred);
        P = (1 - K * H) * P_pred;

        % 存储卡尔曼滤波后的信道增益估计
        x_estimates_5(t, i) = x_k_5;

 % 使用卡尔曼滤波后的信道增益进行MMSE估计
h_estimates_mmse_5(i, t) = (sigma_h2_1 / (sigma_h2_1 + sigma_n2)) * x_estimates_5(t, i);

  end
 end
% % 提取有用的观测信道增益矩阵 h_actual
% for i = 1:num_vehicles
%     for j = 1:num_vehicles
%         if i ~= j
%             h_actual(i, j, :) = gVLC(i, j, :); % 提取非对角线元素，表示每对车辆之间的信道增益
%             % noise = sqrt(sigma_n2) * randn(1, 1, time_steps); % 生成高斯噪声
%             % 使用对数正态分布生成噪声
%             noise = lognrnd(0, sqrt(sigma_n2), [1, 1, time_steps]) - exp(sigma_n2 / 2);
%             y_observed(i, j, :) = h_actual(i, j, :) + noise; % 添加噪声到观测信道增益
%         end
%     end
% end
% 
% % MMSE 信道估计
% for t = 1:time_steps
%     for i = 1:num_vehicles
%         for j = 1:num_vehicles
%             if i ~= j % 只对不同车辆之间的信道增益进行估计
% 
% 
%         % 观测到的信道增益（带噪声）
%         y = y_observed(i, j, t);
% 
%         % MMSE 估计公式
%         h_estimates_mmse(i, j, t) = (sigma_h2 / (sigma_h2 + sigma_n2)) * y;
% 
%         % % 计算目标车辆与其他车辆的实际信道增益和估计信道增益
%         % h_s_actual(i, j, t) = squeeze(h_actual(i, j, t)); % 目标车辆的实际信道增益
%         % h_s_mmse(i, j, t) = squeeze(h_estimates_mmse(i, j, t)); % 目标车辆的 MMSE 信道增益
% 
%         % 基于实际信道的 SINR 计算
%         signal_power_actual(i, j, t) = P_tx * y_observed(i, j, t); % 目标信号功率
%         interference_power_actual(i, j, t) = sum(P_tx * y_observed(i, :, t)) - signal_power_actual(i, j, t); % 干扰功率
%         SINR_actual(i, j, t) = signal_power_actual(i, j, t) / (noise_power + interference_power_actual(i, j, t));
% 
%         % 基于 MMSE 信道的 SINR 计算
%         signal_power_mmse(i, j, t) = P_tx * h_estimates_mmse(i, j, t); % 目标信号功率
%         interference_power_mmse(i, j, t) = sum(P_tx * h_estimates_mmse(i, j, t)) - signal_power_mmse(i, j, t); % 干扰功率
%         SINR_mmse(i, j, t) = signal_power_mmse(i, j, t) / (noise_power + interference_power_mmse(i, j, t));
% 
% 
%             end
%         end
%     end
% end





% % 绘制 信道增益和SINR 对比图 
% figure;
% hold on;
% for i = 1:num_vehicles
%     for j = 1:num_vehicles
%         if i ~= j % 只对不同车辆之间的信道增益进行绘制
%             % 绘制实际观测的信道增益
%             plot(1:time_steps, squeeze(h_actual(i, j, :)), '--', 'DisplayName', ['Vehicle ' num2str(i) ' to ' num2str(j) ' Actual Gain']);
%             % 绘制MMSE估计的信道增益
%             plot(1:time_steps, squeeze(h_estimates_mmse(i, j, :)), 'DisplayName', ['Vehicle ' num2str(i) ' to ' num2str(j) ' Estimated Gain']);
%         end
%     end
% end
% 
% % 绘制 SINR 的对比，只绘制每个车辆自己的SINR
% for i = 1:num_vehicles
%     % 绘制实际的SINR
%     plot(1:time_steps, squeeze(SINR_actual(i, j,:)), '--', 'DisplayName', ['Vehicle ' num2str(i) ' Actual SINR']);
%     % 绘制MMSE估计后的SINR
%     plot(1:time_steps, squeeze(SINR_mmse(i, j,:)), 'DisplayName', ['Vehicle ' num2str(i) ' MMSE Estimated SINR']);
% end
% 
% xlabel('Time Step');
% ylabel('Value');
% legend('show'); % 显示图例
% title('Comparison of Channel Gains and SINR with Actual and MMSE Estimated Values');
% hold off;
for t=1:time_steps

h_mmse(1,2,t)=h_estimates_mmse_1(1,t);h_mmse(1,3,t)=h_estimates_mmse_1(2,t);
h_mmse(1,5,t)=h_estimates_mmse_1(4,t);
h_mmse(2,1,t)=h_estimates_mmse_2(1,t);h_mmse(2,3,t)=h_estimates_mmse_2(2,t);
h_mmse(2,5,t)=h_estimates_mmse_2(4,t);
h_mmse(3,1,t)=h_estimates_mmse_3(1,t);h_mmse(3,2,t)=h_estimates_mmse_3(2,t);
h_mmse(3,5,t)=h_estimates_mmse_3(4,t);
h_mmse(5,1,t)=h_estimates_mmse_5(1,t);h_mmse(5,2,t)=h_estimates_mmse_5(2,t);
h_mmse(5,3,t)=h_estimates_mmse_5(3,t);

h_mmse(4,1,t)=h_estimates_mmse_4(1,t);h_mmse(4,2,t)=h_estimates_mmse_4(2,t);
h_mmse(4,3,t)=h_estimates_mmse_4(3,t);h_mmse(4,5,t)=h_estimates_mmse_4(4,t);
h_mmse(1,4,t)=h_estimates_mmse_1(3,t);h_mmse(2,4,t)=h_estimates_mmse_2(3,t);
h_mmse(3,4,t)=h_estimates_mmse_3(3,t);h_mmse(5,4,t)=h_estimates_mmse_5(4,t);

y_observed(1,2,t)=y_observed_1(1,t);y_observed(1,3,t)=y_observed_1(2,t);
y_observed(1,5,t)=y_observed_1(4,t);
y_observed(2,1,t)=y_observed_2(1,t);y_observed(2,3,t)=y_observed_2(2,t);
y_observed(2,5,t)=y_observed_2(4,t);
y_observed(3,1,t)=y_observed_3(1,t);y_observed(3,2,t)=y_observed_3(2,t);
y_observed(3,5,t)=y_observed_3(4,t);
y_observed(5,1,t)=y_observed_5(1,t);y_observed(5,2,t)=y_observed_5(2,t);
y_observed(5,3,t)=y_observed_5(3,t);

% y_observed(1,4,t)=y_observed_1(3,t);y_observed(5,4,t)=y_observed_5(4,t);
% y_observed(4,1,t)=y_observed_4(1,t);y_observed(4,2,t)=y_observed_4(2,t);
% y_observed(4,3,t)=y_observed_4(3,t);y_observed(4,5,t)=y_observed_4(4,t);
% y_observed(3,4,t)=y_observed_3(3,t);y_observed(2,4,t)=y_observed_2(3,t);
end


