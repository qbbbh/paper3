tolerance = 1e-4;  % 收敛阈值
for t_step = 2:time_steps
    % 计算 SINR 和干扰
    gamma_VLC12 = gVLCLOS12(t_step) * P_VLC1(t_step) / (interference_VLC12(t_step) + noise_VLC);

    % 调用函数进行功率更新
    [P_VLC1_new(t_step)] = update_power_bfgs_sca(P_VLC1(t_step), gamma_VLC12, H_VLC, eta, tolerance);

    % 更新功率
    P_VLC1(t_step) = P_VLC1_new(t_step);
end

function [P_new] = update_power_bfgs_sca(P, gamma, H_prev, grad_prev, eta, tolerance)
% 功能: 使用准牛顿法（BFGS）和SCA方法更新信道功率
% 输入:
%   P         - 当前时间步的功率 (标量或向量)
%   gamma     - 当前时间步的SINR (标量或向量)
%   H_prev    - 前一时间步的Hessian矩阵 (标量或矩阵)
%   grad_prev - 前一时间步的梯度 (标量或向量)
%   eta       - 学习率 (标量)
%   tolerance - 收敛阈值 (标量)
% 输出:
%   P_new     - 更新后的功率 (标量或向量)
%   H_new     - 更新后的Hessian矩阵 (标量或矩阵)
%   grad_new  - 当前时间步的梯度 (标量或向量)
    % 计算当前时间步的alpha和beta参数
    alpha = gamma / (1 + gamma);
    beta = log2(1 + gamma) - alpha * log2(gamma);
    % 逼近速率的函数
    R_approx = alpha * log2(gamma) + beta;
    % 计算功率的梯度
    grad_new = R_approx / P;
    % 计算功率的差分
    s = P - P;  % 功率变化
    y = grad_new - grad_prev;  % 梯度变化
    % 防止s或y为0导致NaN
    if abs(s) > 1e-6 && abs(y) > 1e-6
        % BFGS更新Hessian矩阵
        H_new = H_prev + (y * y') / (y' * s) - (H_prev * s * s' * H_prev) / (s' * H_prev * s);
    else
        % 如果更新无效，保持Hessian矩阵不变
        H_new = H_prev;
    end
    % 使用更新后的Hessian来计算功率的变化
    s_new = eta * (H_new \ grad_new);
    P_new = P - s_new;
    % 确保功率在合理范围内 (可以根据实际情况修改)
    P_new = max(0, P_new);  % 保证功率不为负
    % 收敛检查（如果功率变化很小则停止）
    if abs(P_new - P) < tolerance
        disp('收敛，优化结束');
    end
end
