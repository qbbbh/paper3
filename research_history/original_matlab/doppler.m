% 参数定义
v = 10; % 车辆速度 (单位：米/秒)
lambda = 4e-9; % 信号波长 (单位：米)
delta_t = 0.1; % 时间步长 (单位：秒)
num_steps = 100; % 仿真步数

% 初始化状态转移矩阵 A 的存储数组
A_values = zeros(1, num_steps); % 每个时间步 A 的值

% 计算每个时间步的 A 值
for t = 1:num_steps
    % 动态状态转移矩阵 A (反映多普勒效应)
    A = exp(-1j * 2 * pi * v * delta_t / lambda);
    A_values(t) = A;
    
    % 在这里可以插入其他信道估计过程的代码，例如卡尔曼滤波等
end

% 显示每个时间步的 A 值
disp('每个时间步的动态状态转移矩阵 A 值:');
disp(A_values);
