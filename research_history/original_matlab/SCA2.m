%% 参数初始化
clear;
clc;

n = 5;  % 车辆数量，可动态改变车辆数量
max_iter = 100;  % 最大迭代次数
epsilon = 1e-4;  % 收敛阈值
eta = 0.1;  % 学习率
N0 = 1e-9;  % 噪声功率谱密度

% 初始化信道增益矩阵 h_ij，表示车辆之间的信道增益
% h(i, j) 表示车辆 i 对车辆 j 的信道增益
h_RF = rand(n, n);  % RF信道增益矩阵
h_VLC = rand(n, n);  % VLC信道增益矩阵
B = ones(n, 1) * 10;  % 带宽，假设每个子信道带宽相同

% 初始化功率分配，RF和VLC
P_RF = rand(n, 1) * 10;  % RF信道初始功率分配
P_VLC = rand(n, 1) * 10;  % VLC信道初始功率分配

% 初始化Hessian矩阵为单位矩阵
H_RF = eye(n);  
H_VLC = eye(n); 

%% SCA迭代过程

for t = 1:max_iter
    %% Step 1: 计算RF和VLC信道的信干噪比 (SINR)
    gamma_RF = zeros(n, 1);  % 存储每辆车的RF信道的SINR
    gamma_VLC = zeros(n, 1);  % 存储每辆车的VLC信道的SINR
    
    for i = 1:n
        % RF信道的SINR计算
        interference_RF = sum(h_RF(i, :) .* P_RF) - h_RF(i, i) * P_RF(i);  % 其他车辆对车辆 i 的干扰
        gamma_RF(i) = h_RF(i, i) * P_RF(i) / (N0 + interference_RF);
        
        % VLC信道的SINR计算
        interference_VLC = sum(h_VLC(i, :) .* P_VLC) - h_VLC(i, i) * P_VLC(i);  % 其他车辆对车辆 i 的干扰
        gamma_VLC(i) = h_VLC(i, i) * P_VLC(i) / (N0 + interference_VLC);
    end
    
    %% Step 2: 信道切换 (根据SINR选择RF或VLC信道)
    a = zeros(n, 1);  % 信道选择标志，1表示VLC信道，0表示RF信道
    for i = 1:n
        if gamma_VLC(i) > gamma_RF(i)
            a(i) = 1;  % 选择VLC信道
            P_RF(i) = 0;  % RF信道功率设为0
        else
            a(i) = 0;  % 选择RF信道
            P_VLC(i) = 0;  % VLC信道功率设为0
        end
    end
    
    %% Step 3: 更新功率分配 (使用准牛顿法BFGS)
    
    % RF信道功率分配更新
    grad_EE_RF = gradient_EE(P_RF, h_RF, B, N0);
    P_RF_new = P_RF - eta * inv(H_RF) * grad_EE_RF;
    
    % VLC信道功率分配更新
    grad_EE_VLC = gradient_EE(P_VLC, h_VLC, B, N0);
    P_VLC_new = P_VLC - eta * inv(H_VLC) * grad_EE_VLC;
    
    % 计算 RF 和 VLC 信道的 y_t 和 s_t
    s_t_RF = P_RF_new - P_RF;
    grad_EE_RF_new = gradient_EE(P_RF_new, h_RF, B, N0);
    y_t_RF = grad_EE_RF_new - grad_EE_RF;
    
    s_t_VLC = P_VLC_new - P_VLC;
    grad_EE_VLC_new = gradient_EE(P_VLC_new, h_VLC, B, N0);
    y_t_VLC = grad_EE_VLC_new - grad_EE_VLC;
    
    % 更新 RF 和 VLC 信道的Hessian矩阵 (BFGS更新)
    H_RF = H_RF + (y_t_RF * y_t_RF') / (y_t_RF' * s_t_RF) - (H_RF * s_t_RF * s_t_RF' * H_RF) / (s_t_RF' * H_RF * s_t_RF);
    H_VLC = H_VLC + (y_t_VLC * y_t_VLC') / (y_t_VLC' * s_t_VLC) - (H_VLC * s_t_VLC * s_t_VLC' * H_VLC) / (s_t_VLC' * H_VLC * s_t_VLC);
    
    % 更新功率
    P_RF = P_RF_new;
    P_VLC = P_VLC_new;
    
    %% Step 4: 收敛条件判断
    if norm(s_t_RF) < epsilon && norm(s_t_VLC) < epsilon
        break;
    end
    
end

%% 输出最优功率分配
disp('最优RF信道功率分配:');
disp(P_RF);

disp('最优VLC信道功率分配:');
disp(P_VLC);

%% 辅助函数：计算能量效率的梯度
function grad_EE = gradient_EE(P, h, B, N0)
    n = length(P);
    grad_R = zeros(n, 1);
    for i = 1:n
        grad_R(i) = B(i) * h(i, i) / ((N0 + sum(h(i, :) .* P) - h(i, i) * P(i)) * log(2));
    end
    total_power = sum(P);
    total_rate = sum(grad_R);
    
    grad_EE = grad_R / total_power - total_rate / total_power^2;
end
