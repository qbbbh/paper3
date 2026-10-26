% 初始化参数
clc;
clear;
% 车辆初始设置
num_vehicles = 10; % 总车车辆数量，可随时调整
vehicle_length = 4; % 车辆长度 (米)
inter_vehicle_gap = 6; % 两车之间的纵向间隔 (米)
% 参数设置
lane_width = 3.6; % 每条车道宽度 (米)
lateral_distance = 2; % 两列车之间的横向距离 (米)
% VLC和RF参数
DR = 0.024; % 发射器接收器距离相关系数
Epsilon = 0.0175; % 光功率衰减指数
Xi =0.1585; % VLC增益比例因子
ALOS=18.7; BLOS=20;CLOS=46.8;
fc = 2.4; % RF信道频率 (Hz)

% 确定每列车队的车辆数
num_vehicles_odd = ceil(num_vehicles / 2); % 第一列车队的车辆数（奇数）
num_vehicles_even = num_vehicles - num_vehicles_odd; % 第二列车队的车辆数（偶数）
% 仿真时间设置
sim_time = 50; % 仿真总时间 (秒)
time_step = 0.1; % 时间步长 (秒)
time = 0:time_step:sim_time;

% 初始化速度（每辆车独立）
v_lead_1 = 10; % 第一列车队头车初始速度 (米/秒)
v_lead_2 = 8; % 第二列车队头车初始速度 (米/秒)
% a_lead_1 = 0.5; % 第一列车队头车加速度 (米/秒^2)
% a_lead_2 = 0.3; % 第二列车队头车加速度 (米/秒^2)

% 初始化位置矩阵
pos_x1 = zeros(num_vehicles_odd, length(time)); % 第一列车队纵向位置
pos_x2 = zeros(num_vehicles_even, length(time)); % 第二列车队纵向位置
pos_y1 = (lane_width / 2) * ones(num_vehicles_odd, 1); % 第一列车队横向位置
pos_y2 = (lane_width / 2 + lateral_distance) * ones(num_vehicles_even, 1); % 第二列车队横向位置

% 初始化信道增益（每辆车独立）
gVLC = zeros(num_vehicles/ 2, num_vehicles/ 2, length(time)); % VLC信道增益
gRF = zeros(num_vehicles/ 2, num_vehicles/ 2, length(time)); % RF信道增益

% 速度矩阵
v1 = zeros(num_vehicles_odd, length(time)); % 第一列车队速度
v2 = zeros(num_vehicles_even, length(time)); % 第二列车队速度
v1(1, 1) = v_lead_1; % 第一列车队头车速度
v2(1, 1) = v_lead_2; % 第二列车队头车速度
for t = 2:100
v1(1,t)=10;
v2(1,t)=8;
end
for t = 101:200
v1(1,t)=9;
v2(1,t)=11;
end
for t = 201:350
v1(1,t)=8;
v2(1,t)=9;
end
for t = 351:500
v1(1,t)=8.5;
v2(1,t)=9;
end
% 仿真过程
for t = 2:length(time)
    % 更新头车速度


    % 更新头车位置
    pos_x1(1, t) = pos_x1(1, t-1) + v1(1, t-1) * time_step;
    pos_x2(1, t) = pos_x2(1, t-1) + v2(1, t-1) * time_step;

    % 更新队列中其他车辆的位置
    for n = 1:num_vehicles_odd
        % 第一列车队
        pos_x1(n, t) = pos_x1(1, t) - (n-1) * (vehicle_length + inter_vehicle_gap);
        v1(n, t) = v1(1, t); % 假设每辆车与头车速度相同
    end
    for n = 2:num_vehicles_even
        % 第二列车队
        pos_x2(n, t) = pos_x2(1, t) - (n-1) * (vehicle_length + inter_vehicle_gap);
        v2(n, t) = v2(1, t); % 假设每辆车与头车速度相同
    end

    % 计算信道增益
    for i = 1:num_vehicles_odd
        for j = 1:num_vehicles_even
            % if i ~= j % 排除自己和自己之间的通信
                % 计算角度和距离
                angle = atan2d(pos_y2(j) - pos_y1(i), pos_x2(j, t) - pos_x1(i, t)); % 计算角度
                distance = sqrt((pos_x2(j, t) - pos_x1(i, t))^2 + (pos_y2(j) - pos_y1(i))^2); % 计算距离
                if i~=j
                % 计算车队内的角度和距离
                angle1 = atan2d(pos_y1(j) - pos_y1(i), pos_x1(j, t) - pos_x1(i, t)); % 计算车队内角度
                distance1 = sqrt((pos_x1(j, t) - pos_x1(i, t))^2 + (pos_y1(j) - pos_y1(i))^2); % 计算车队内距离
                end
                % 保存角度和距离
                angles(i, j, t) = angle;
                distances(i, j, t) = distance;
          
                % 计算VLC增益
                if abs(angle) <= 60
                    gVLC(i, j, t) = ((DR * (cosd(angle))^(1/Epsilon)) / (Xi * distance))^2;
                else
                    gVLC(i, j, t) = 0;
                end
                % 计算VLC增益
                if i~=j
                if abs(angle1) <= 60 && i==j+1
                    gVLC1(i, j, t) = ((DR * (cosd(angle1))^(1/Epsilon)) / (Xi * distance1))^2;
                else
                    gVLC1(i, j, t) = 0;
                end
                end

                % 计算RF增益
                PLRFLOS = ALOS * log10(distance) + BLOS * log10(fc / 5) + CLOS;
                gRF(i, j, t) = 10^(-PLRFLOS / 10);
                if i~=j
                PLRFLOS1 = ALOS * log10(distance1) + BLOS * log10(fc / 5) + CLOS;
                gRF1(i, j, t) = 10^(-PLRFLOS1 / 10);
                end
                
            % end
        end
    end
 
end


