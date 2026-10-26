% 定义目标函数 g(x) 和 h(x)
g = @(x) x(1)^2 + x(2)^2;   % 分子函数 g(x)
h = @(x) x(1) + x(2);       % 分母函数 h(x)

% 梯度
grad_g = @(x) [2*x(1); 2*x(2)];  % g(x) 的梯度
grad_h = @(x) [1; 1];            % h(x) 的梯度

% 设置初始解和初始 lambda
x = [1; 1];            % 初始点 x0
lambda = g(x) / h(x);  % 初始 lambda
tol = 1e-6;            % 收敛容差
max_iter = 1000;       % 最大迭代次数

% 梯度下降法的参数设置
alpha = 0.01;          % 梯度下降的学习率
grad_tol = 1e-6;       % 梯度下降法的停止条件
max_grad_iter = 500;   % 梯度下降法的最大迭代次数

% Dinkelbach 算法迭代过程
for iter = 1:max_iter
    % 优化子问题：max g(x) - lambda * h(x)（手动实现梯度下降）
    for grad_iter = 1:max_grad_iter
        % 计算当前目标函数的梯度：grad(g(x) - lambda * h(x))
        grad = grad_g(x) - lambda * grad_h(x);
        
        % 使用梯度下降法更新 x
        x_new = x - alpha * grad;
        
        % 检查梯度下降是否收敛
        if norm(x_new - x) < grad_tol
            break;
        end
        
        % 更新 x
        x = x_new;
    end
    
    % 计算新的 lambda
    lambda_new = g(x) / h(x);
    
    % 检查收敛条件（Dinkelbach 算法的终止条件）
    if abs(g(x) - lambda_new * h(x)) < tol
        disp('收敛');
        break;
    end
    
    % 更新 lambda
    lambda = lambda_new;
end

% 显示最优解
disp('最优解:');
disp(x);
disp('最优 lambda:');
disp(lambda);
disp('迭代次数:');
disp(iter);
