filename = '添加对比算法.xlsx';  % 你的文件名
data = readmatrix(filename);
% 获取数据的行数和列数
[rows, cols] = size(data);

% 创建一个新的矩阵用于存放计算后的平均值
averaged_data = zeros(rows / 10, cols);

% 循环每列数据，按每10个数计算平均数
for col = 1:cols
    for i = 1:rows/10
        start_idx = (i-1)*10 + 1;  % 计算每个小块的开始索引
        end_idx = i*10;            % 计算每个小块的结束索引
        
        % 计算每10个数的平均数
        averaged_data(i, col) = mean(data(start_idx:end_idx, col));
    end
end

% 输出新矩阵大小
disp(size(averaged_data));  % 输出 (100, N)