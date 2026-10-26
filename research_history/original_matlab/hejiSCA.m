% 道路参数
L_straight = 100; % 直道长度 (meters)
R_curve_inner = 6.5;   % 内车道弯道半径 (meters)
lane_width = 3.6; % 车道宽度 (meters)
R_curve_outer = R_curve_inner + lane_width; % 外车道弯道半径
Epsilon=0.0175; Xi=0.1585;
ALOS=18.7; BLOS=20;CLOS=46.8;
ANLOS=36.8;
fc=2.4; %载波频率 RF系数 GHz 
DR=0.024;%PD孔径直径
% 时间参数
dt = 0.1; % 时间步长 (seconds)
T = 100;   % 模拟总时间 (seconds)
time_steps = 0:dt:T; % 时间步长序列
% 初始化车辆位置
x1 = -L_straight/2; y1 = -R_curve_inner; % 车辆1的初始位置 (内侧车道)
x2 = -L_straight/2; y2 = -R_curve_inner; % 车辆2
x3 = -L_straight/2; y3 = -R_curve_outer; % 车辆3在外侧车道
x4 = L_straight/2; y4 = R_curve_inner;   % 车辆4
x5 = -L_straight/2; y5 = -R_curve_inner; % 车辆5
% 初始化车辆行驶状态 (1: 下直道, 2: 右弯道, 3: 上直道, 4: 左弯道)
state1 = 1; state2 = 1; state3 = 1; state4 = 3; state5 = 1;
theta1 = 0; theta2 = 0; theta3 = 0; theta4 = 0; theta5 = 0; % 弯道的初始角度
% 模拟车辆运动
for t_idx = 1:num_time_steps
    t = time_steps(t_idx);

    % 初始化车辆速度 m/s
    v1 = 9; v2 = 10; v3 = 12; v4 = 9; v5 = 11;
    if t >= 50
    v1 = 9.5; v2 = 11; v3 = 11; v4 = 10; v5 = 12;
    end

    % 更新每辆车的位置
    [x1, y1, theta1, state1] = update_vehicle_position(x1, y1, theta1, state1, v1, dt, L_straight, R_curve_inner);
    [x2, y2, theta2, state2] = update_vehicle_position(x2, y2, theta2, state2, v2, dt, L_straight, R_curve_inner);
    [x3, y3, theta3, state3] = update_vehicle_position(x3, y3, theta3, state3, v3, dt, L_straight, R_curve_outer);
    [x4, y4, theta4, state4] = update_vehicle_position(x4, y4, theta4, state4, v4, dt, L_straight, R_curve_inner);
    [x5, y5, theta5, state5] = update_vehicle_position(x5, y5, theta5, state5, v5, dt, L_straight, R_curve_inner);

    % 记录车辆位置
    vehicle_positions1(t_idx, :) = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5];
    
    % 计算距离和角度矩阵
    [distances(t_idx,:), angles(t_idx,:)] = calculate_distances_angles(vehicle_positions(t_idx, :));

    % 计算VLC和RF信道增益
    for pair_idx = 1:10
        angle = angles(t_idx, pair_idx);
        distance = distances(t_idx, pair_idx);
        if abs(angle) <= 60
            gVLC(t_idx, pair_idx) = ((DR * (cosd(angle))^(1/Epsilon)) / (Xi * distance))^2;
        else
            gVLC(t_idx, pair_idx) = 0;
        end
        PL_RF = ALOS * log10(distance) + BLOS * log10(fc/5) + CLOS;
        gRF(t_idx, pair_idx) = 10^(-PL_RF / 10);
    end
end

% 功能函数：计算车辆之间的距离和角度
function [distances, angles] = calculate_distances_angles(positions)
    distances = zeros(1, 10);
    angles = zeros(1, 10);
    pairs = [1 2; 1 3; 1 4; 1 5; 2 3; 2 4; 2 5; 3 4; 3 5; 4 5];
    for idx = 1:10
        p1 = pairs(idx, 1); p2 = pairs(idx, 2);
        x1 = positions(2*p1-1); y1 = positions(2*p1);
        x2 = positions(2*p2-1); y2 = positions(2*p2);
        distances(idx) = sqrt((x1 - x2)^2 + (y1 - y2)^2);
        angles(idx) = atan2d(y2 - y1, x2 - x1);
    end
end
% 函数：更新车辆位置
function [x, y, theta, state] = update_vehicle_position(x, y, theta, state, v, dt, L, R)
    % 根据车辆的状态和当前位置更新其位置
    d = v * dt; % 每个时间步长内行驶的距离
    angular_speed = d / R; % 角速度

    if state == 1
        % 车辆在下直道上
        x = x + d;
        if x >= L/2
            % 进入右弯道
            x = L/2;
            theta = -pi/2; % 弯道的初始角度
            state = 2;
        end
    elseif state == 2
        % 车辆在右弯道上，使用极坐标更新位置
        theta = theta + angular_speed;
        x = L/2 + R * cos(theta);
        y =  R * sin(theta);
        if theta >= pi/2
            % 进入上直道
            theta = pi/2;
            y = R;
            state = 3;
        end
    elseif state == 3
        % 车辆在上直道上
        x = x - d;
        if x <= -L/2
            % 进入左弯道
            x = -L/2;
            theta = pi/2;
            state = 4;
        end
    elseif state == 4
        % 车辆在左弯道上，使用极坐标更新位置
        theta = theta + angular_speed;
        x = -L/2 + R * cos(theta);
        y =  R * sin(theta);
        if theta >= 3*pi/2
            % 进入下直道
            theta = 0;
            y = -R;
            state = 1;
        end
    end
end