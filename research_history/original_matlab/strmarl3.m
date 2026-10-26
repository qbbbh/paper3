% 参数设置
numVehiclesTeam1 = input('请输入第一车队的车辆数量：'); % 用户输入第一车队车辆数量
numVehiclesTeam2 = input('请输入第二车队的车辆数量：'); % 用户输入第二车队车辆数量
totalVehicles = numVehiclesTeam1 + numVehiclesTeam2; % 总车辆数
numTimeSteps = 500;    % 总模拟时间（毫秒），与原代码时间步数类似
bandwidthRF = 1000000;      % RF 带宽（Hz），沿用原代码带宽设置示例
bandwidthVLC = 2000000;     % VLC 带宽（Hz），沿用原代码带宽设置示例
noisePowerSpectralDensityRF = 8e-23;  % RF 噪声功率谱密度（W/Hz），沿用原代码设置示例
noisePowerSpectralDensityVLC = 1e-21; % VLC 噪声功率谱密度（W/Hz），沿用原代码设置示例
noisePowerRF = noisePowerSpectralDensityRF * bandwidthRF; % RF 噪声功率（W）
noisePowerVLC = noisePowerSpectralDensityVLC * bandwidthVLC; % VLC 噪声功率（W）
maxPowerRF = 30;        % RF 最大发射功率（W），沿用原代码功率设置示例
maxPowerVLC = 30;       % VLC 最大发射功率（W），沿用原代码功率设置示例
minPowerRF = 1;         % RF 最小发射功率（W），沿用原代码功率设置示例
minPowerVLC = 20;       % VLC 最小发射功率（W），沿用原代码功率设置示例
learningRate = 0.1;     % Q-learning学习率，可根据实际调整
discountFactor = 0.9;   % Q-learning折扣因子，可根据实际调整
numStates = 10;          % 离散化状态数量，根据实际情况合理设置

% 初始化动作和功率分配（考虑所有车辆和车队情况）
selectionMatrixRF = zeros(totalVehicles, totalVehicles); % RF 选择矩阵，涵盖所有车辆
selectionMatrixVLC = zeros(totalVehicles, totalVehicles); % VLC 选择矩阵，涵盖所有车辆
powerAllocationRF = ones(totalVehicles, totalVehicles) * maxPowerRF / 2; % 初始 RF 功率，涵盖所有车辆
powerAllocationVLC = ones(totalVehicles, totalVehicles) * maxPowerVLC / 2; % 初始 VLC 功率，涵盖所有车辆

% 初始化Q表，初始值可根据实际情况设置，这里简单设为0
Q_table = zeros(numStates, 2, totalVehicles); % 2表示两种动作选择（RF或VLC），考虑所有车辆

% 初始化结果跟踪（分别记录两个车队的情况）
energyEfficiencyHistoryTeam1RF = zeros(numTimeSteps, 1); % 第一车队RF通信每个时间步的能量效率
energyEfficiencyHistoryTeam1VLC = zeros(numTimeSteps, 1); % 第一车队VLC通信每个时间步的能量效率
energyEfficiencyHistoryTeam2RF = zeros(numTimeSteps, 1); % 第二车队RF通信每个时间步的能量效率
energyEfficiencyHistoryTeam2VLC = zeros(numTimeSteps, 1); % 第二车队VLC通信每个时间步的能量效率
iterationCountHistory = zeros(numTimeSteps, 1);    % 每个时间步的迭代次数

% 标记两个车队的车辆索引范围
team1StartIndex = 1;
team1EndIndex = numVehiclesTeam1;
team2StartIndex = numVehiclesTeam1 + 1;
team2EndIndex = totalVehicles;

for timeStep = 1:numTimeSteps
    % 获取当前时间步的信道增益（这里假设对应不同车队有相应的信道增益矩阵，例如gRF_team1、gVLC_team1等，需根据实际情况准备好数据）
    currentGainRF_team1 = gRF(:, :, timeStep); % 第一车队当前时间步RF增益
    currentGainVLC_team1 = gVLC(:, :, timeStep); % 第一车队当前时间步VLC增益
    currentGainRF_team2 = gRF1(:, :, timeStep); % 第二车队当前时间步RF增益
    currentGainVLC_team2 = gVLC1(:, :, timeStep); % 第二车队当前时间步VLC增益
    
    % 初始化收敛准则相关变量（这里简单沿用原代码类似逻辑，可根据实际调整）
    maxIterations = 50; % 最大迭代次数
    convergenceThreshold = 1e-6; % 收敛阈值
    currentIteration = 0;
    changeDelta = Inf;

    % 迭代优化过程（类似原代码的交替优化逻辑，融入Q-learning的动作选择和Q表更新等）
    while currentIteration < maxIterations && changeDelta > convergenceThreshold
        % 保存上一轮的功率值用于收敛检查
        prevPowerRF = powerAllocationRF;
        prevPowerVLC = powerAllocationVLC;

        % 根据Q-learning策略和能量效率等选择动作（选择 RF 或 VLC，针对所有车辆）
        for vehicleI = 1:totalVehicles
            for vehicleJ = 1:totalVehicles
                if vehicleI ~= vehicleJ % 避免自身通信
                    % 判断车辆所属车队，分别处理不同车队的通信计算
                    if vehicleI <= numVehiclesTeam1 && vehicleJ <= numVehiclesTeam1
                        % 第一车队内通信计算
                        [channelChoice, sinrRF_team1, sinrVLC_team1] = updateChannelAndSINR(bandwidthVLC,bandwidthRF,currentGainRF_team1, currentGainVLC_team1, powerAllocationRF, powerAllocationVLC, noisePowerRF, noisePowerVLC, vehicleI, vehicleJ);
                        selectionMatrixRF(vehicleI, vehicleJ) = 1 - channelChoice; % 0表示RF，1表示VLC
                        selectionMatrixVLC(vehicleI, vehicleJ) = channelChoice;
                    elseif vehicleI > numVehiclesTeam1 && vehicleJ > numVehiclesTeam1
                        % 第二车队内通信计算
                        % 修正索引取值方式，确保索引为正整数，避免出现无效索引问题
                        indexI = vehicleI - numVehiclesTeam1;
                        indexJ = vehicleJ - numVehiclesTeam1;
                        [channelChoice, sinrRF_team2, sinrVLC_team2] = updateChannelAndSINR(bandwidthVLC,bandwidthRF,currentGainRF_team2, currentGainVLC_team2, powerAllocationRF, powerAllocationVLC, noisePowerRF, noisePowerVLC, indexI, indexJ);
                        selectionMatrixRF(vehicleI, vehicleJ) = 1 - channelChoice; % 0表示RF，1表示VLC
                        selectionMatrixVLC(vehicleI, vehicleJ) = channelChoice;
                    end
                    
                    % 根据选择的动作（信道）计算即时奖励（这里简单示例，可根据实际详细定义奖励函数）
                    reward = calculateReward(sinrRF_team1, sinrVLC_team1, sinrRF_team2, sinrVLC_team2, powerAllocationRF(vehicleI, vehicleJ), powerAllocationVLC(vehicleI, vehicleJ), vehicleI, vehicleJ);
                    
                    % 更新状态（基于信干噪比等离散化到numStates范围，这里需自定义离散化函数，暂未详细实现）
                    state = discretizeStates(sinrRF_team1, sinrVLC_team1, sinrRF_team2, sinrVLC_team2, numStates);
                    
                    % 更新Q表（根据Q-learning更新公式）
                    actionIndex = selectionMatrixVLC(vehicleI, vehicleJ); % 0或1表示选择的动作（RF或VLC）
                    Q_table(state, actionIndex, vehicleI) = Q_table(state, actionIndex, vehicleI) +...
                        learningRate * (reward + discountFactor * max(Q_table(:, actionIndex, vehicleI)) - Q_table(state, actionIndex, vehicleI));
                end
            end
        end

        % 更新功率分配（添加一定的随机扰动，类似原代码逻辑，可根据实际调整）
        powerAllocationRF = min(maxPowerRF, max(minPowerRF, powerAllocationRF + randn(size(powerAllocationRF)) * 0.1));
        powerAllocationVLC = min(maxPowerVLC, max(minPowerVLC, powerAllocationVLC + randn(size(powerAllocationVLC)) * 0.1));

        % 检查收敛情况（类似原代码检查功率分配等变化是否小于阈值）
        changeDelta = max(abs(powerAllocationRF(:) - prevPowerRF(:))) + max(abs(powerAllocationVLC(:) - prevPowerVLC(:)));
        currentIteration = currentIteration + 1;
    end

    % 记录当前时间步的迭代次数
    iterationCountHistory(timeStep) = currentIteration;

    % 分别计算两个车队的容量（调用自定义函数，类似原代码中计算容量逻辑）
    [capacityRF_team1, capacityVLC_team1] = calculateCapacity(currentGainRF_team1, currentGainVLC_team1, powerAllocationRF, powerAllocationVLC, noisePowerRF, noisePowerVLC, bandwidthRF, bandwidthVLC, sinrRF_team1, sinrVLC_team1);
    [capacityRF_team2, capacityVLC_team2] = calculateCapacity(currentGainRF_team2, currentGainVLC_team2, powerAllocationRF, powerAllocationVLC, noisePowerRF, noisePowerVLC, bandwidthRF, bandwidthVLC, sinrRF_team2, sinrVLC_team2);
    
    % 计算第一车队RF通信的总能量效率
    totalEnergyEfficiencyTeam1RF = sum(capacityRF_team1(:)./ max(powerAllocationRF(team1StartIndex:team1EndIndex, :), minPowerRF).* selectionMatrixRF(team1StartIndex:team1EndIndex, :));
    energyEfficiencyHistoryTeam1RF(timeStep) = totalEnergyEfficiencyTeam1RF;
    
    % 计算第一车队VLC通信的总能量效率
    totalEnergyEfficiencyTeam1VLC = sum(capacityVLC_team1(:)./ max(powerAllocationVLC(team1StartIndex:team1EndIndex, :), minPowerVLC).* selectionMatrixVLC(team1StartIndex:team1EndIndex, :));
    energyEfficiencyHistoryTeam1VLC(timeStep) = totalEnergyEfficiencyTeam1VLC;
    
    % 计算第二车队RF通信的总能量效率
    totalEnergyEfficiencyTeam2RF = sum(capacityRF_team2(:)./ max(powerAllocationRF(team2StartIndex:team2EndIndex, :), minPowerRF).* selectionMatrixRF(team2StartIndex:team2EndIndex, :));
    energyEfficiencyHistoryTeam2RF(timeStep) = totalEnergyEfficiencyTeam2RF;
    
    % 计算第二车队VLC通信的总能量效率
    totalEnergyEfficiencyTeam2VLC = sum(capacityVLC_team2(:)./ max(powerAllocationVLC(team2StartIndex:team2EndIndex, :), minPowerVLC).* selectionMatrixVLC(team2StartIndex:team2EndIndex, :));
    energyEfficiencyHistoryTeam2VLC(timeStep) = totalEnergyEfficiencyTeam2VLC;
end

disp('优化完成。');

% 绘制随时间变化的第一车队RF通信能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyHistoryTeam1RF, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('第一车队RF通信能量效率（比特/秒/瓦）');
title('第一车队RF通信能量效率随时间变化');
grid on;

% 绘制随时间变化的第一车队VLC通信能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyHistoryTeam1VLC, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('第一车队VLC通信能量效率（比特/秒/瓦）');
title('第一车队VLC通信能量效率随时间变化');
grid on;

% 绘制随时间变化的第二车队RF通信能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyHistoryTeam2RF, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('第二车队RF通信能量效率（比特/秒/瓦）');
title('第二车队RF通信能量效率随时间变化');
grid on;

% 绘制随时间变化的第二车队VLC通信能量效率曲线
figure;
plot(1:numTimeSteps, energyEfficiencyHistoryTeam2VLC, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('第二车队VLC通信能量效率（比特/秒/瓦）');
title('第二车队VLC通信能量效率随时间变化');
grid on;

% 绘制随时间变化的迭代次数曲线
figure;
plot(1:numTimeSteps, iterationCountHistory, 'LineWidth', 2);
xlabel('时间步（毫秒）');
ylabel('迭代次数');
title('算法随时间的迭代次数');
grid on;

% 自定义函数：更新信道选择和计算信干噪比（类似原代码中相关逻辑整合）
function [channelChoice, sinrRF, sinrVLC] = updateChannelAndSINR(bandwidthVLC,bandwidthRF,gainRF, gainVLC, powerRF, powerVLC, noiseRF, noiseVLC, vehicleI, vehicleJ)
    % 计算RF干扰和信干噪比
    bandwidthRF = 1000000;      % RF 带宽（Hz），沿用原代码带宽设置示例
bandwidthVLC = 2000000;     % VLC 带宽（Hz），沿用原代码带宽设置示例
    interferenceRF = sum(gainRF(:, vehicleJ).* powerRF(vehicleI, vehicleJ)) - gainRF(vehicleI, vehicleJ) * powerRF(vehicleI, vehicleJ);
    interferenceRF = max(interferenceRF, 1e-9); % 防止分母为零
    sinrRF = gainRF(vehicleI, vehicleJ) * powerRF(vehicleI, vehicleJ) / (interferenceRF + noiseRF);
    
    % 计算VLC干扰和信干噪比
    interferenceVLC = sum(gainVLC(:, vehicleJ).* powerVLC(vehicleI, vehicleJ)) - gainVLC(vehicleI, vehicleJ) * powerVLC(vehicleI, vehicleJ);
    interferenceVLC = max(interferenceVLC, 1e-9); % 防止分母为零
    sinrVLC = gainVLC(vehicleI, vehicleJ) * powerVLC(vehicleI, vehicleJ) / (interferenceVLC + noiseVLC);
    
    % 计算 VLC 等效信干噪比
    equ_SINR_VLC = 2 * ((1 + sinrVLC/3.3)^(bandwidthVLC/bandwidthRF) - 1);
    
    % 信道选择
    if equ_SINR_VLC > sinrRF
        channelChoice = 1;  % 选择VLC
    else
        channelChoice = 0;  % 选择RF
    end
end

% 自定义函数：计算奖励（示例简单函数，可根据实际详细定义）
function reward = calculateReward(sinrRF_team1, sinrVLC_team1, sinrRF_team2, sinrVLC_team2, powerRF, powerVLC, vehicleI, vehicleJ)
    % 这里简单示例，根据信干噪比等情况给出奖励值，可根据实际详细定义
    reward = sinrRF_team1 + sinrVLC_team1 + sinrRF_team2 + sinrVLC_team2 - powerRF - powerVLC;
end

% 自定义函数：离散化状态（暂未详细实现，需根据实际情况具体定义如何将信干噪比等信息映射到离散状态）
function state = discretizeStates(sinrRF_team1, sinrVLC_team1, sinrRF_team2, sinrVLC_team2, numStates)
    % 示例：简单返回固定状态值，实际需根据具体规则将信干噪比等信息转换为离散状态
    state = 1;
end

% 自定义函数：计算容量（类似原代码中计算容量逻辑）
function [capacityRF, capacityVLC] = calculateCapacity(gainRF, gainVLC, powerRF, powerVLC, noiseRF, noiseVLC, bandwidthRF, bandwidthVLC, sinrRF, sinrVLC)
    % RF 容量计算
    capacityRF = bandwidthRF * log2(1 + sinrRF);
    % VLC 容量计算
    capacityVLC = bandwidthVLC * log2(1 + sinrVLC);
end