% 参数初始化
P_VLC1(1) = 30; % VLC信道初始功率分配
tolerance = 1e-4;  % 收敛阈值
eta = 0.1;  % 学习率
time_steps = 1000; % 假设有1000个时刻
H_VLC = 1;  % 初始Hessian矩阵为单位标量

% 初始化梯度和历史梯度
grad_VLC1_prev = 0;

for t_step = 1:time_steps
    % 动态计算VLC信道的alpha和beta
    alpha_VLC1(t_step+1) = gamma_VLC12(t_step) / (1 + gamma_VLC12(t_step));
    beta_VLC1(t_step+1) = log2(1 + gamma_VLC12(t_step)) - alpha_VLC1(t_step) * log2(gamma_VLC12(t_step));

    % 逼近速率的函数
    R_approx_VLC1(t_step+1) = alpha_VLC1(t_step) * log2(gamma_VLC12(t_step)) + beta_VLC1(t_step);

    % 计算功率的梯度
    grad_VLC1(t_step+1) = R_approx_VLC1(t_step) / P_VLC1(t_step);

    % 计算功率的差分
    s_VLC1(t_step+1) = P_VLC1(t_step) - P_VLC1(t_step - 1);  % 功率变化
    y_VLC1(t_step+1) = grad_VLC1(t_step) - grad_VLC1_prev;  % 梯度变化

    % 防止s_VLC1或y_VLC1为0导致计算NaN
    if abs(s_VLC1(t_step)) > 1e-6 && abs(y_VLC1(t_step)) > 1e-6
        % BFGS更新Hessian矩阵
        H_VLC = H_VLC + (y_VLC1(t_step) * y_VLC1(t_step)') / (y_VLC1(t_step)' * s_VLC1(t_step)) ...
                - (H_VLC * s_VLC1(t_step) * s_VLC1(t_step)' * H_VLC) / (s_VLC1(t_step)' * H_VLC * s_VLC1(t_step));
    end

    % 使用更新后的Hessian来计算功率的变化
    s_VLC1_new(t_step) = eta * (H_VLC \ grad_VLC1(t_step));
    P_VLC1_new(t_step) = P_VLC1(t_step) - s_VLC1_new(t_step);

    % 动态调整VLC功率
    P_VLC1(t_step) = P_VLC1_new(t_step);  % 更新新的VLC功率值

    % 收敛检查（如果功率变化很小则停止）
    if abs(P_VLC1_new(t_step) - P_VLC1(t_step)) < tolerance
        disp('收敛，优化结束');
        break;
    end

    % 更新上一个梯度
    grad_VLC1_prev = grad_VLC1(t_step);
end
