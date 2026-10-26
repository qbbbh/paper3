% 参数初始化
P_RF1(1) = 20; P_RF2(1) = 20; P_RF3(1) = 20; P_RF4(1) = 20; P_RF5(1) = 20; % RF信道初始功率分配
P_VLC1(1) = 30; P_VLC2(1) = 30; P_VLC3(1) = 30; P_VLC4(1) = 30; P_VLC5(1) = 30; % VLC信道初始功率分配
tolerance = 1e-4;  % 收敛阈值
% 初始化初始梯度和 Hessian 矩阵
grad_RF1_prev = 0; grad_RF2_prev = 0; grad_RF3_prev = 0; grad_RF4_prev = 0; grad_RF5_prev = 0;
grad_VLC1_prev = 0; grad_VLC2_prev = 0; grad_VLC3_prev = 0; grad_VLC4_prev = 0; grad_VLC5_prev = 0;

% 初始化Hessian矩阵为单位矩阵 (每个时刻都初始化)
H_RF = ones(1, time_steps);  % 对每个时刻初始化RF信道Hessian矩阵为1
H_VLC = ones(1, time_steps);  % 对每个时刻初始化VLC信道Hessian矩阵为1

% 初始化 α 和 β 辅助变量
alpha_RF1 = 1; alpha_RF2 = 1; alpha_RF3 = 1; alpha_RF4 = 1; alpha_RF5 = 1;
beta_RF1 = 1; beta_RF2 = 1; beta_RF3 = 1; beta_RF4 = 1; beta_RF5 = 1;
alpha_VLC1 = 1; alpha_VLC2 = 1; alpha_VLC3 = 1; alpha_VLC4 = 1; alpha_VLC5 = 1;
beta_VLC1 = 1; beta_VLC2 = 1; beta_VLC3 = 1; beta_VLC4 = 1; beta_VLC5 = 1;

for t_step = 2:time_steps
          % VLC信道
  
  alpha_VLC(t_step) = gamma_VLC12(t_step) / (1 + gamma_VLC12(t_step));
  beta_VLC(t_step) = log2(1 + gamma_VLC12(t_step)) -alpha_VLC(t_step)*log2(gamma_VLC12(t_step)) ;
  %速率逼近函数
  R_approx(t_step) = alpha_VLC(t_step) * log2(gamma_VLC12(t_step)) + beta_VLC(t_step);
  % 计算功率的梯度
  grad_VLC(t_step) = R_approx(t_step) / P_VLC1(t_step);
  % 计算功率的差分
  s_VLC(t_step) = P_VLC1(t_step) - P_VLC1(t_step-1);  % 功率的变化
  y_VLC(t_step) = grad_VLC(t_step) - grad_VLC(t_step-1);  % 梯度的变化
  % BFGS更新VLC信道Hessian矩阵
  H_VLC(t_step) = H_VLC(t_step) + (y_VLC(t_step) * y_VLC(t_step)') / (y_VLC(t_step)' * s_VLC(t_step)) ...
                - (H_VLC(t_step) * s_VLC(t_step) * s_VLC(t_step)' * H_VLC(t_step)) / (s_VLC(t_step)' * H_VLC(t_step) * s_VLC(t_step));

 
  % 使用更新后的Hessian来计算功率的变化
  s_VLC1_new(t_step) = eta * (H_VLC(t_step) \ grad_VLC(t_step));
  P_VLC1_new(t_step) = P_VLC1(t_step) - s_VLC1_new(t_step);       
  % 动态调整VLC功率
  P_VLC1(t_step) = P_VLC1_new(t_step);  % 直接更新新的VLC功率值
        
  
  % 收敛检查（如果功率变化很小则停止）
  if max(abs(P_VLC1_new(t_step) - P_VLC1(t_step))) < tolerance
  
        break;
  end



end
