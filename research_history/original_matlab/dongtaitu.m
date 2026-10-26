% 田径场的参数
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

% 初始化图形
figure;
hold on;
axis equal;
xlim([-L_straight/2 - R_curve_outer - 20, L_straight/2 + R_curve_outer + 20]);
ylim([-R_curve_outer - 20, R_curve_outer + 20]);
grid on;
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('Real-time Vehicle Positions on Track');
% 绘制跑道
% 绘制内侧车道的直道部分
plot([-L_straight/2, L_straight/2], [-R_curve_inner, -R_curve_inner], 'k', 'LineWidth', 2); % 下直道
plot([-L_straight/2, L_straight/2], [R_curve_inner, R_curve_inner], 'k', 'LineWidth', 2);   % 上直道
% 绘制外侧车道的直道部分
plot([-L_straight/2, L_straight/2], [-R_curve_outer, -R_curve_outer], 'k', 'LineWidth', 2); % 外侧下直道
plot([-L_straight/2, L_straight/2], [R_curve_outer, R_curve_outer], 'k', 'LineWidth', 2);   % 外侧上直道
% 绘制内侧车道的弯道
% 绘制左右弯道
theta1 = linspace(-pi, 0, 100); % 上下弯道的角度
theta2 = linspace(-pi, 0, 100); % 上下弯道的角度
x_left = -L_straight/2 + R_curve_inner * sin(theta1);
y_left = R_curve_inner * cos(theta1);
plot(x_left, y_left, 'k', 'LineWidth', 2);
% 右弯道
x_right = L_straight/2 - R_curve_inner * sin(theta2);
y_right = R_curve_inner * cos(theta2);
plot(x_right, -y_right, 'k', 'LineWidth', 2);
% 绘制外侧车道的弯道
theta1_outer = linspace(-pi, 0, 100); % 外侧弯道的角度
x_left_outer = -L_straight/2 + R_curve_outer * sin(theta1_outer);
y_left_outer = R_curve_outer * cos(theta1_outer);
plot(x_left_outer, y_left_outer, 'k', 'LineWidth', 2);
theta2_outer = linspace(-pi, 0, 100); % 外侧弯道的角度
x_right_outer = L_straight/2 - R_curve_outer * sin(theta2_outer);
y_right_outer =  R_curve_outer * cos(theta2_outer);
plot(x_right_outer, y_right_outer, 'k', 'LineWidth', 2);
% 创建车辆位置图例
h1 = plot(x1, y1, 'ro', 'MarkerSize', 10, 'DisplayName', 'Vehicle 1');
h2 = plot(x2, y2, 'go', 'MarkerSize', 10, 'DisplayName', 'Vehicle 2');
h3 = plot(x3, y3, 'bo', 'MarkerSize', 10, 'DisplayName', 'Vehicle 3'); % 3号车在外侧车道
h4 = plot(x4, y4, 'co', 'MarkerSize', 10, 'DisplayName', 'Vehicle 4');
h5 = plot(x5, y5, 'mo', 'MarkerSize', 10, 'DisplayName', 'Vehicle 5');
legend([h1 h2 h3 h4 h5]); % 仅显示车辆的图例
% 初始化用于记录位置的矩阵
vehicle_positions = zeros(length(time_steps), 10); % 存储五辆车的x和y位置
% 初始化GIF保存设置
filename = 'vehicle_motion.gif';
% 模拟车辆运动
for t_idx = 1:length(time_steps)
    t = time_steps(t_idx);
    % 车辆速度 (m/s)
v1 = 8;  % 车辆1速度
v2 = 10; % 车辆2速度
v3 = 12; % 车辆3速度（在外侧车道上行驶）
v4 = 9;  % 车辆4速度
v5 = 11; % 车辆5速度
  if t >= 50
        v1 = 9;  % 车辆1速度
        v2 = 11; % 车辆2速度
        v3 = 11; % 车辆3速度（在外侧车道上行驶）
        v4 = 10; % 车辆4速度
        v5 = 12; % 车辆5速度
  end
    % 更新车辆1的位置
    [x1, y1, theta1, state1] = update_vehicle_position(x1, y1, theta1, state1, v1, dt, L_straight, R_curve_inner);
    set(h1, 'XData', x1, 'YData', y1); % 更新车辆1位置
    
    % 更新车辆2的位置
    [x2, y2, theta2, state2] = update_vehicle_position(x2, y2, theta2, state2, v2, dt, L_straight, R_curve_inner);
    set(h2, 'XData', x2, 'YData', y2); % 更新车辆2位置
    
    % 更新车辆3的位置 (外侧车道)
    [x3, y3, theta3, state3] = update_vehicle_position(x3, y3, theta3, state3, v3, dt, L_straight, R_curve_outer);
    set(h3, 'XData', x3, 'YData', y3); % 更新车辆3位置
    
    % 更新车辆4的位置
    [x4, y4, theta4, state4] = update_vehicle_position(x4, y4, theta4, state4, v4, dt, L_straight, R_curve_inner);
    set(h4, 'XData', x4, 'YData', y4); % 更新车辆4位置
    
    % 更新车辆5的位置
    [x5, y5, theta5, state5] = update_vehicle_position(x5, y5, theta5, state5, v5, dt, L_straight, R_curve_inner);
    set(h5, 'XData', x5, 'YData', y5); % 更新车辆5位置

    % 记录每个时间步长的车辆位置
    vehicle_positions(t_idx, :) = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5];
     % 捕获当前图像帧
    frame = getframe(gcf);
    im = frame2im(frame); % 将帧转换为图像
    [imind, cm] = rgb2ind(im, 256); % 将RGB图像转换为索引图像

    % 将图像保存为GIF文件
    if t == 0
        % 初始化GIF文件
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', dt);
    else
        % 追加帧到GIF文件
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
    % 刷新图像
    pause(0.05);
end
% 初始化矩阵，用于存储各车辆之间的距离和视场角
num_time_steps = size(vehicle_positions, 1);
distance12 = zeros(num_time_steps, 1);distance13 = zeros(num_time_steps, 1);
distance14 = zeros(num_time_steps, 1);distance15 = zeros(num_time_steps, 1);
distance23 = zeros(num_time_steps, 1);distance24 = zeros(num_time_steps, 1);
distance25 = zeros(num_time_steps, 1);distance34 = zeros(num_time_steps, 1);
distance35 = zeros(num_time_steps, 1);distance45 = zeros(num_time_steps, 1);

angle12 = zeros(num_time_steps, 1);angle13 = zeros(num_time_steps, 1);
angle14 = zeros(num_time_steps, 1);angle15 = zeros(num_time_steps, 1);
angle23 = zeros(num_time_steps, 1);angle24 = zeros(num_time_steps, 1);
angle25 = zeros(num_time_steps, 1);angle34 = zeros(num_time_steps, 1);
angle35 = zeros(num_time_steps, 1);angle45 = zeros(num_time_steps, 1);

for t_idx = 1:num_time_steps
    % 提取每个车辆在时间 t_idx 的坐标
    x1 = vehicle_positions(t_idx, 1);    y1 = vehicle_positions(t_idx, 2);   
    x2 = vehicle_positions(t_idx, 3);    y2 = vehicle_positions(t_idx, 4);    
    x3 = vehicle_positions(t_idx, 5);    y3 = vehicle_positions(t_idx, 6);
    x4 = vehicle_positions(t_idx, 7);    y4 = vehicle_positions(t_idx, 8);    
    x5 = vehicle_positions(t_idx, 9);    y5 = vehicle_positions(t_idx, 10);
    
    % 计算各车辆之间的距离
    distance12(t_idx) = sqrt((x1 - x2)^2 + (y1 - y2)^2);
    distance13(t_idx) = sqrt((x1 - x3)^2 + (y1 - y3)^2);
    distance14(t_idx) = sqrt((x1 - x4)^2 + (y1 - y4)^2);
    distance15(t_idx) = sqrt((x1 - x5)^2 + (y1 - y5)^2);
    distance23(t_idx) = sqrt((x2 - x3)^2 + (y2 - y3)^2);
    distance24(t_idx) = sqrt((x2 - x4)^2 + (y2 - y4)^2);
    distance25(t_idx) = sqrt((x2 - x5)^2 + (y2 - y5)^2);
    distance34(t_idx) = sqrt((x3 - x4)^2 + (y3 - y4)^2);
    distance35(t_idx) = sqrt((x3 - x5)^2 + (y3 - y5)^2);
    distance45(t_idx) = sqrt((x4 - x5)^2 + (y4 - y5)^2);
    
    % 计算各车辆之间的视场角（以车辆1为参考，例如1-2，1-3，1-4）
    % 视场角定义为两个向量之间的夹角，这里计算基于坐标差
    angle12(t_idx) = atan2d(y2 - y1, x2 - x1);    angle13(t_idx) = atan2d(y3 - y1, x3 - x1);
    angle14(t_idx) = atan2d(y4 - y1, x4 - x1);    angle15(t_idx) = atan2d(y5 - y1, x5 - x1);
    angle23(t_idx) = atan2d(y3 - y2, x3 - x2);    angle24(t_idx) = atan2d(y4 - y2, x4 - x2);
    angle25(t_idx) = atan2d(y5 - y2, x5 - x2);    angle34(t_idx) = atan2d(y4 - y3, x4 - x3);
    angle35(t_idx) = atan2d(y5 - y3, x5 - x3);    angle45(t_idx) = atan2d(y5 - y4, x5 - x4);
end

for t_idx = 1:num_time_steps
if abs(angle12(t_idx))<=60
gVLCLOS12(t_idx)=((DR*(cosd(angle12(t_idx)))^(1/Epsilon))/(Xi* distance12(t_idx)))^2;
else gVLCLOS12(t_idx)=0;
end
PLRFLOS12(t_idx)=ALOS*log10(distance12(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF12(t_idx)=10^(-PLRFLOS12(t_idx)/10);

if abs(angle13(t_idx))<=60
gVLCLOS13(t_idx)=((DR*(cosd(angle13(t_idx)))^(1/Epsilon))/(Xi* distance13(t_idx)))^2;
else gVLCLOS13(t_idx)=0;
end
PLRFLOS13(t_idx)=ALOS*log10(distance13(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF13(t_idx)=10^(-PLRFLOS13(t_idx)/10);

if abs(angle14(t_idx))<=60
gVLCLOS14(t_idx)=((DR*(cosd(angle14(t_idx)))^(1/Epsilon))/(Xi* distance14(t_idx)))^2;
else gVLCLOS14(t_idx)=0;
end
PLRFLOS14(t_idx)=ALOS*log10(distance14(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF14(t_idx)=10^(-PLRFLOS14(t_idx)/10);

if abs(angle15(t_idx))<=60
gVLCLOS15(t_idx)=((DR*(cosd(angle15(t_idx)))^(1/Epsilon))/(Xi* distance15(t_idx)))^2;
else gVLCLOS15(t_idx)=0;
end
PLRFLOS15(t_idx)=ALOS*log10(distance15(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF15(t_idx)=10^(-PLRFLOS15(t_idx)/10);

if abs(angle23(t_idx))<=60
gVLCLOS23(t_idx)=((DR*(cosd(angle23(t_idx)))^(1/Epsilon))/(Xi* distance23(t_idx)))^2;
else gVLCLOS23(t_idx)=0;
end
PLRFLOS23(t_idx)=ALOS*log10(distance23(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF23(t_idx)=10^(-PLRFLOS23(t_idx)/10);

if abs(angle24(t_idx))<=60
gVLCLOS24(t_idx)=((DR*(cosd(angle24(t_idx)))^(1/Epsilon))/(Xi* distance24(t_idx)))^2;
else gVLCLOS24(t_idx)=0;
end
PLRFLOS24(t_idx)=ALOS*log10(distance24(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF24(t_idx)=10^(-PLRFLOS24(t_idx)/10);

if abs(angle25(t_idx))<=60
gVLCLOS25(t_idx)=((DR*(cosd(angle25(t_idx)))^(1/Epsilon))/(Xi* distance25(t_idx)))^2;
else gVLCLOS25(t_idx)=0;
end
PLRFLOS25(t_idx)=ALOS*log10(distance25(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF25(t_idx)=10^(-PLRFLOS25(t_idx)/10);

if abs(angle34(t_idx))<=60
gVLCLOS43(t_idx)=((DR*(cosd(angle34(t_idx)))^(1/Epsilon))/(Xi* distance34(t_idx)))^2;
else gVLCLOS25(t_idx)=0;
end
PLRFLOS34(t_idx)=ALOS*log10(distance25(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF34(t_idx)=10^(-PLRFLOS34(t_idx)/10);

if abs(angle35(t_idx))<=60
gVLCLOS35(t_idx)=((DR*(cosd(angle35(t_idx)))^(1/Epsilon))/(Xi* distance35(t_idx)))^2;
else gVLCLOS35(t_idx)=0;
end
PLRFLOS35(t_idx)=ALOS*log10(distance35(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF35(t_idx)=10^(-PLRFLOS35(t_idx)/10);

if abs(angle45(t_idx))<=60
gVLCLOS45(t_idx)=((DR*(cosd(angle45(t_idx)))^(1/Epsilon))/(Xi* distance45(t_idx)))^2;
else gVLCLOS45(t_idx)=0;
end
PLRFLOS45(t_idx)=ALOS*log10(distance45(t_idx))+BLOS*log10(fc/5)+CLOS;
gRF45(t_idx)=10^(-PLRFLOS45(t_idx)/10);

end 
%要先旋转一个链路，才能计算信干噪比吧
% 保存距离和视场角数据
% save('vehicle_distances_and_angles.mat', 'distance12', 'distance13', 'distance14', 'distance15', 'distance23', 'distance24', 'distance25', 'distance34', 'distance35', 'distance45', ...
    % 'angle12', 'angle13', 'angle14', 'angle15', 'angle23', 'angle24', 'angle25', 'angle34', 'angle35', 'angle45');

%需要截取部分时刻



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
