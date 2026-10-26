% 参数定义
N = 5; % 用户数量
time_steps = 1001; % 时间步数
sigma_epsilon_n = 0.0001; % 噪声CSI的方差
E = 0.005; % 过时CSI的误差边界

% 初始化信道增益
% h_actual_2 = rand(N, time_steps); % 假设真实的信道增益
h_actual_1=[gVLCLOS12;gVLCLOS13;gVLCLOS14;gVLCLOS15];
h_actual_2=[gVLCLOS21;gVLCLOS23;gVLCLOS24;gVLCLOS25];
h_actual_3=[gVLCLOS31;gVLCLOS32;gVLCLOS34;gVLCLOS35];
h_actual_4=[gVLCLOS41;gVLCLOS42;gVLCLOS43;gVLCLOS45];
h_actual_5=[gVLCLOS51;gVLCLOS52;gVLCLOS53;gVLCLOS54];
% 初始化存储变量
h_estimates_noise = zeros(time_steps, N); % 存储噪声CSI的信道增益估计
h_estimates_outdated = zeros(time_steps, N); % 存储过时CSI的信道增益估计
% 论文方法的信道估计-车辆1
for i = 1:N-1
    for t = 2:time_steps
        % 噪声CSI
        noise_1 = sqrt(sigma_epsilon_n) * abs(0.000001*randn(1));
        y_observed_paper_1 = h_actual_1(i, t) + noise_1;
        
        % 过时CSI
        epsilon_1 =abs( E * 0.000001*randn(1)); % 生成过时CSI误差
        h_outdated_paper_1 = h_actual_1(i, t) + epsilon_1;
        
        % 选择噪声CSI和过时CSI中较小的误差
        h_estimate_1 = min(y_observed_paper_1, h_outdated_paper_1);
        
        % 存储信道增益估计
        h_estimates_paper_1(i, t) = h_estimate_1;
    end
end
% 论文方法的信道估计-车辆2
for i = 1:N-1
    for t = 2:time_steps
        % 噪声CSI
        noise_2 = sqrt(sigma_epsilon_n) * abs(0.000001*randn(1));
        y_observed_paper_2 = h_actual_2(i, t) + noise_2;
        
        % 过时CSI
        epsilon_2 =abs( E * 0.000001*randn(1)); % 生成过时CSI误差
        h_outdated_paper_2 = h_actual_2(i, t) + epsilon_2;
        
        % 选择噪声CSI和过时CSI中较小的误差
        h_estimate_2 = min(y_observed_paper_2, h_outdated_paper_2);
        
        % 存储信道增益估计
        h_estimates_paper_2(i, t) = h_estimate_2;
    end
end

% 论文方法的信道估计-车辆3
for i = 1:N-1
    for t = 2:time_steps
        % 噪声CSI
        noise_3 = sqrt(sigma_epsilon_n) * abs(0.000001*randn(1));
        y_observed_paper_3 = h_actual_3(i, t) + noise_3;
        
        % 过时CSI
        epsilon_3 =abs( E * 0.000001*randn(1)); % 生成过时CSI误差
        h_outdated_paper_3 = h_actual_3(i, t) + epsilon_3;
        
        % 选择噪声CSI和过时CSI中较小的误差
        h_estimate_3 = min(y_observed_paper_3, h_outdated_paper_3);
        
        % 存储信道增益估计
        h_estimates_paper_3(i, t) = h_estimate_3;
    end
end
% 论文方法的信道估计-车辆4
for i = 1:N-1
    for t = 2:time_steps
        % 噪声CSI
        noise_4 = sqrt(sigma_epsilon_n) * abs(0.000001*randn(1));
        y_observed_paper_4 = h_actual_4(i, t) + noise_4;
        
        % 过时CSI
        epsilon_4 =abs( E * 0.000001*randn(1)); % 生成过时CSI误差
        h_outdated_paper_4 = h_actual_4(i, t) + epsilon_4;
        
        % 选择噪声CSI和过时CSI中较小的误差
        h_estimate_4 = min(y_observed_paper_4, h_outdated_paper_4);
        
        % 存储信道增益估计
        h_estimates_paper_4(i, t) = h_estimate_4;
    end
end

% 论文方法的信道估计-车辆5
for i = 1:N-1
    for t = 2:time_steps
        % 噪声CSI
        noise_5 = sqrt(sigma_epsilon_n) * abs(0.000001*randn(1));
        y_observed_paper_5 = h_actual_5(i, t) + noise_5;
        
        % 过时CSI
        epsilon_5 =abs( E * 0.000001*randn(1)); % 生成过时CSI误差
        h_outdated_paper_5 = h_actual_3(i, t) + epsilon_5;
        
        % 选择噪声CSI和过时CSI中较小的误差
        h_estimate_5 = min(y_observed_paper_5, h_outdated_paper_5);
        
        % 存储信道增益估计
        h_estimates_paper_5(i, t) = h_estimate_5;
    end
end
% 计算均方误差（MSE）

for t=1:time_steps

h_mmse_paper(1,2,t)=h_estimates_paper_1(1,t);h_mmse_paper(1,3,t)=h_estimates_paper_1(2,t);
h_mmse_paper(1,5,t)=h_estimates_paper_1(4,t);
h_mmse_paper(2,1,t)=h_estimates_paper_2(1,t);h_mmse_paper(2,3,t)=h_estimates_paper_2(2,t);
h_mmse_paper(2,5,t)=h_estimates_paper_2(4,t);
h_mmse_paper(3,1,t)=h_estimates_paper_3(1,t);h_mmse_paper(3,2,t)=h_estimates_paper_3(2,t);
h_mmse_paper(3,5,t)=h_estimates_paper_3(4,t);
h_mmse_paper(5,1,t)=h_estimates_paper_5(1,t);h_mmse_paper(5,2,t)=h_estimates_paper_5(2,t);
h_mmse_paper(5,3,t)=h_estimates_paper_5(3,t);

h_mmse_paper(4,1,t)=h_estimates_paper_4(1,t);h_mmse_paper(4,2,t)=h_estimates_paper_4(2,t);
h_mmse_paper(4,3,t)=h_estimates_paper_4(3,t);h_mmse_paper(4,5,t)=h_estimates_paper_4(4,t);
h_mmse_paper(1,4,t)=h_estimates_paper_1(3,t);h_mmse_paper(2,4,t)=h_estimates_paper_2(3,t);
h_mmse_paper(3,4,t)=h_estimates_paper_3(3,t);h_mmse_paper(5,4,t)=h_estimates_paper_5(4,t);

end
