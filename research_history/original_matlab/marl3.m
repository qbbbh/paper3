% 参数设置 gRF是车对外 gRF1是车队内
numVehicles = 5;        % 车辆数量
numTimeSteps = 1000;    % 总模拟时间（毫秒）
bandwidthRF = 1e6;      % 射频（RF）带宽（赫兹）
bandwidthVLC = 2e6;     % 可见光通信（VLC）带宽（赫兹）
noisePowerSpectralDensityRF = 8e-23;  % RF 噪声功率谱密度（瓦/赫兹）
noisePowerSpectralDensityVLC = 1e-21; % VLC 噪声功率谱密度（瓦/赫兹）
noisePowerRF = noisePowerSpectralDensityRF * bandwidthRF; % RF 噪声功率（瓦）
noisePowerVLC = noisePowerSpectralDensityVLC * bandwidthVLC; % VLC 噪声功率（瓦）
maxPowerRF = 20;        % RF 最大发射功率（瓦）
maxPowerVLC = 30;       % VLC 最大发射功率（瓦）
minPowerRF = 0.1;       % RF 最小发射功率（瓦）
minPowerVLC = 20;       % VLC 最小发射功率（瓦）
minRateRF = 1e3;        % RF 最小速率（比特/秒）
minRateVLC = 1e3;       % VLC 最小速率（比特/秒）

% % 加载信道增益矩阵（g_RF 和 g_VLC），此处假设文件已存在，可根据实际情况修改路径或加载方式
% load('channel_gain_RF.mat');  % RF 信道增益（5x5x1000）
% load('channel_gain_VLC.mat'); % VLC 信道增益（5x5x1000）

% 学习率和折扣因子，可根据实际情况调整参数值
alpha = 0.1;  
gamma = 0.9;  

% 离散化状态的数量，根据实际情况合理设置
numStates = 10;  

% 初始化动作和功率分配
selectionMatrixRF = zeros(numVehicles, numVehicles); % RF 选择矩阵
selectionMatrixVLC = zeros(numVehicles, numVehicles); % VLC 选择矩阵
powerAllocationRF = ones(numVehicles, numVehicles) * maxPowerRF / 2; % 初始 RF 功率
powerAllocationVLC = ones(numVehicles, numVehicles) * maxPowerVLC / 2; % 初始 VLC 功率

% 初始化Q表，初始值可根据实际情况设置，这里简单设为0
Q_table = zeros(numStates, 2, numVehicles); % 2表示两种动作选择（RF或VLC）

% 初始化结果跟踪
energyEfficiencyHistory = zeros(numTimeSteps, 1);         % 跟踪每个时间步的总能量效率
energyEfficiencyPerVehicle = zeros(numTimeSteps, numVehicles); % 每个时间步每辆车的能量效率
iterationCountHistory = zeros(numTimeSteps, 1);    % 每个时间步的迭代次数

for timeStep = 1:numTimeSteps
    % 获取当前时间步的信道增益
    currentGainRF = gRF(:, :, timeStep); % 当前时间步 RF 增益
    currentGainVLC = gVLC(:, :, timeStep); % 当前时间步 VLC 增益
    
    % 初始化收敛准则相关变量
    maxIterations = 50; % 最大迭代次数
    convergenceThreshold = 1e-6; % 收敛阈值
    currentIteration = 0;
    changeDelta = Inf;

    % 迭代优化过程
    while currentIteration < maxIterations && changeDelta > convergenceThreshold
        % 保存上一轮的功率值用于收敛检查
        prevPowerRF = powerAllocationRF;
        prevPowerVLC = powerAllocationVLC;

        % 根据能量效率和约束条件更新动作（选择 RF 或 VLC）
        for vehicleI = 1:numVehicles
            for vehicleJ = 1:numVehicles
                if vehicleI ~= vehicleJ % 避免自身通信
                    % 计算 RF 的干扰和信干噪比（SINR）
                    interferenceRF = sum(currentGainRF(:, vehicleJ).* powerAllocationRF(:, vehicleJ)) - currentGainRF(vehicleI, vehicleJ) * powerAllocationRF(vehicleI, vehicleJ);
                    interferenceRF = max(interferenceRF, 1e-9); % 防止分母为零
                    sinrRF(vehicleI, vehicleJ) = currentGainRF(vehicleI, vehicleJ) * powerAllocationRF(vehicleI, vehicleJ) / (interferenceRF + noisePowerRF);
                    
                    % 计算 VLC 的干扰和信干噪比（SINR）
                    interferenceVLC = sum(currentGainVLC(:, vehicleJ).* powerAllocationVLC(:, vehicleJ)) - currentGainVLC(vehicleI, vehicleJ) * powerAllocationVLC(vehicleI, vehicleJ);
                    interferenceVLC = max(interferenceVLC, 1e-9); % 防止分母为零
                    sinrVLC(vehicleI, vehicleJ) = currentGainVLC(vehicleI, vehicleJ) * powerAllocationVLC(vehicleI, vehicleJ) / (interferenceVLC + noisePowerVLC);
                    
                    % 计算 VLC 等效信干噪比
                    equivalentSinrVLC = 2 * ((1 + sinrVLC(vehicleI, vehicleJ)/3.3)^(bandwidthVLC/bandwidthRF) - 1);
                    
                    % 根据等效信干噪比进行信道选择
                    if equivalentSinrVLC > sinrRF(vehicleI, vehicleJ)
                        selectionMatrixRF(vehicleI, vehicleJ) = 0;
                        selectionMatrixVLC(vehicleI, vehicleJ) = 1; % 选择 VLC
                    else
                        selectionMatrixRF(vehicleI, vehicleJ) = 1;
                        selectionMatrixVLC(vehicleI, vehicleJ) = 0; % 选择 RF
                    end
                end
            end
        end

        % 更新功率分配（添加一定的随机扰动）
        powerAllocationRF = min(maxPowerRF, max(minPowerRF, powerAllocationRF + randn(size(powerAllocationRF)) * 0.1));
        powerAllocationVLC = min(maxPowerVLC, max(minPowerVLC, powerAllocationVLC + randn(size(powerAllocationVLC)) * 0.1));

        % 检查收敛情况
        changeDelta = max(abs(powerAllocationRF(:) - prevPowerRF(:))) + max(abs(powerAllocationVLC(:) - prevPowerVLC(:)));
        currentIteration = currentIteration + 1;
    end

    % 记录当前时间步的迭代次数
    iterationCountHistory(timeStep) = currentIteration;
     % 先计算容量，得到 capacityRF 和 capacityVLC
    [capacityRF, capacityVLC] = calculateCapacity(currentGainRF, currentGainVLC, powerAllocationRF, powerAllocationVLC, noisePowerRF, noisePowerVLC, bandwidthRF, bandwidthVLC, sinrRF, sinrVLC);
    % 计算即时奖励（假设奖励与能量效率相关，调用自定义函数）
    reward = calculateReward(capacityRF, capacityVLC, powerAllocationRF, powerAllocationVLC, minRateRF, minRateVLC);

    % 更新状态（基于信干噪比离散化到numStates范围）
    state = discretizeStates(sinrRF, sinrVLC, numStates);

    % 更新Q表
    for vehicleIndex = 1:numVehicles
        for actionIndex = 1:2 % 两种动作，0表示选择RF，1表示选择VLC，这里简单遍历两种情况
            % 根据Q-learning更新公式更新Q值
            Q_table(state(vehicleIndex), actionIndex, vehicleIndex) = Q_table(state(vehicleIndex), actionIndex, vehicleIndex) +...
                alpha * (reward + gamma * max(Q_table(:, actionIndex, vehicleIndex)) - Q_table(state(vehicleIndex), actionIndex, vehicleIndex));
        end
    end

    % 计算容量和总能量效率，调用自定义函数

    totalEnergyEfficiencyRF = sum(capacityRF(:)./ max(powerAllocationRF(:), minPowerRF).* selectionMatrixRF(:)); % RF 总能量效率
    totalEnergyEfficiencyVLC = sum(capacityVLC(:)./ max(powerAllocationVLC(:), minPowerVLC).* selectionMatrixVLC(:)); % VLC 总能量效率
    energyEfficiencyHistory(timeStep) = totalEnergyEfficiencyRF + totalEnergyEfficiencyVLC;

    % 计算每辆车的能量效率
    for vehicleIndex = 1:numVehicles
        energyEfficiencyPerVehicle(timeStep, vehicleIndex) = sum(capacityRF(vehicleIndex, :)./ max(powerAllocationRF(vehicleIndex, :), minPowerRF).* selectionMatrixRF(vehicleIndex, :)) +...
                                                           sum(capacityVLC(vehicleIndex, :)./ max(powerAllocationVLC(vehicleIndex, :), minPowerVLC).* selectionMatrixVLC(vehicleIndex, :));
    end
end

disp('优化完成。');

% 绘制随时间变化的总能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyHistory, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('总能量效率（比特/秒/瓦）');
title('随时间变化的总能量效率');
grid on;

% 绘制每辆车随时间变化的能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyPerVehicle, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('每辆车的能量效率（比特/秒/瓦）');
title('每辆车随时间变化的能量效率');
legend('车辆 1', '车辆 2', '车辆 3', '车辆 4', '车辆 5');
grid on;
% % 
% % % 绘制每辆车随时间变化的平均误码率曲线（此处 avg_BER_history 变量未在前面定义，需根据实际情况补充数据）
% % figure;
% % plot(1:numTimeSteps, avg_BER_history, 'LineWidth', 2);
% % xlabel('时间步（毫秒）');
% % ylabel('平均误码率');
% % title('每辆车随时间变化的平均误码率');
% % legend('车辆 1', '车辆 2', '车辆 3', '车辆 4', '车辆 5');
% % grid on;
% 
% % 绘制随时间变化的迭代次数曲线
% figure;
% plot(1:numTimeSteps, iterationCountHistory, 'LineWidth', 2);
% xlabel('时间步（毫秒）');
% ylabel('迭代次数');
% title('算法随时间的迭代次数');
% grid on;

% 自定义函数：计算容量
function [capacityRF, capacityVLC] = calculateCapacity(gainRF, gainVLC, powerRF, powerVLC, noiseRF, noiseVLC, bandwidthRF, bandwidthVLC, sinrRF, sinrVLC)
    % RF 容量计算
    capacityRF = bandwidthRF * log2(1 + sinrRF);
    % VLC 容量计算
    capacityVLC = bandwidthVLC * log2(1 + sinrVLC);
end

% 自定义函数：计算奖励
function reward = calculateReward(capacityRF, capacityVLC, powerRF, powerVLC, minRateRF, minRateVLC)
    % 计算 RF 和 VLC 的能量效率
    energyEfficiencyRF = capacityRF./ powerRF;
    energyEfficiencyVLC = capacityVLC./ powerVLC;

    % 应用服务质量（QoS）约束，计算惩罚项
    qosPenaltyRF = max(0, minRateRF - capacityRF); % RF 的 QoS 惩罚
    qosPenaltyVLC = max(0, minRateVLC - capacityVLC); % VLC 的 QoS 惩罚

    % 组合奖励（考虑能量效率和惩罚项）
    reward = sum(energyEfficiencyRF(:)) + sum(energyEfficiencyVLC(:)) - sum(qosPenaltyRF(:)) - sum(qosPenaltyVLC(:));
end

% 自定义函数：基于信干噪比离散化状态
function state = discretizeStates(sinrRF, sinrVLC, numStates)
    % 这里简单将所有车辆的信干噪比合并成一个向量，然后根据范围离散化
    combinedSINR = [sinrRF(:); sinrVLC(:)];
    minSINR = min(combinedSINR);
    maxSINR = max(combinedSINR);
    state = round((combinedSINR - minSINR) / (maxSINR - minSINR) * (numStates - 1)) + 1;
    % 确保状态值在合理范围内
    state = min(numStates, max(1, state));
end