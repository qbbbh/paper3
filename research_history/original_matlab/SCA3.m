%% 参数初始化

n = 5;  % 车辆数量

epsilon = 1e-4;  % 收敛阈值
eta = 0.1;  % 学习率
N0_VLC = 1e-21;  % VLC噪声功率谱密度 单位A^2/Hz,噪声：N0_VLC*B_VLC
N0_RF = 8e-23;   % RF噪声功率谱密度 单位A^2/Hz
lambda_reg = 1e-6;  % 正则化参数，用于防止Hessian矩阵奇异
time_steps = 1001;  % 仿真时间步数
B_RF = 100000;  % RF带宽，假设每个子信道带宽相同 参数来源：论文7
B_VLC= 200000;   % VLC带宽  
% 手动定义每对车辆之间的RF和VLC信道增益向量


% 初始化功率分配，RF和VLC
P_RF1 = rand(1, time_steps);P_RF2 = zeros(1, time_steps);P_RF3 = zeros(1, time_steps);
P_RF4 = rand(1, time_steps);P_RF5 = zeros(1, time_steps);% RF信道初始功率分配
P_VLC1 = rand(1, time_steps); P_VLC2 = zeros(1, time_steps);P_VLC3 = zeros(1, time_steps);
P_VLC4 = rand(1, time_steps); P_VLC5 = zeros(1, time_steps);% VLC信道初始功率分配
P_RF1(1) = 30;P_RF2(1) = 30;P_RF3(1) = 30;P_RF4(1) = 30;P_RF5(1) = 30;
P_VLC1(1) = 30;P_VLC2(1) = 30;P_VLC3(1) = 30;P_VLC4(1) = 30;P_VLC5(1) = 30;%信道功率初值
% 初始化Hessian矩阵为单位矩阵，并加入正则化参数
H_RF = eye(n) + lambda_reg * eye(n);  
H_VLC = eye(n) + lambda_reg * eye(n );
% 初始化信道噪声
noise_RF=N0_RF*B_RF;  %RF信道噪声，没变化
noise_VLC=N0_VLC*B_VLC;  %VLC信道噪声，没变化
% 用于存储当前时间步长的SINR值
gamma_RF = zeros(n, n);  % 存储RF信道的SINR
gamma_VLC = zeros(n, n);  % 存储VLC信道的SINR
interference_RF= zeros(n, n);
interference_VLC= zeros(n, n);
% 初始化信道选择矩阵，1表示VLC，0表示RF
channel_choice = zeros(n, n);  
%% 主仿真过程
% 针对每个时间步长进行SCA迭代，处理信道增益的变化
for t_step = 1:time_steps
    total_power_RF(t_step)= P_RF1(t_step)+P_RF2(t_step)+P_RF3(t_step)+P_RF4(t_step)+P_RF5(t_step);  % RF总功耗
    total_power_VLC(t_step)= P_VLC1(t_step)+P_VLC2(t_step)+P_VLC3(t_step)+P_VLC4(t_step)+P_VLC5(t_step); % VLC总功耗
    %PRF和PVLC是需要优化的功率分配
    %% Step 1: 计算RF和VLC信道的信干噪比 (SINR) 对每对车辆进行比较
    %RF 
    % 车辆1到车辆2
    interference_RF12(t_step) = gRF23(t_step)*P_RF3(t_step)+gRF24(t_step)*P_RF4(t_step)+gRF25(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF12(t_step) = gRF12(t_step) * P_RF1(t_step) / (interference_RF12(t_step)+noise_RF);  % RF SINR
    % 车辆1到车辆3
    interference_RF13(t_step) = gRF23(t_step)*P_RF2(t_step)+gRF34(t_step)*P_RF4(t_step)+gRF35(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF13(t_step) = gRF13(t_step) * P_RF1(t_step) / (interference_RF13(t_step)+noise_RF);  % RF SINR
    % 车辆1到车辆4
    interference_RF14(t_step) = gRF24(t_step)*P_RF2(t_step)+gRF34(t_step)*P_RF3(t_step)+gRF45(t_step)*P_RF5(t_step);  % RF干扰，gRF45和gRF54数值相同，这里应该是gRF54(t_step)*PRF5(t_step)，代表从五号车发出的功率，后面也有类似的
    gamma_RF14(t_step) = gRF14(t_step) * P_RF1(t_step) / (interference_RF14(t_step)+noise_RF);  % RF SINR
    % 车辆1到车辆5
    interference_RF15(t_step) = gRF25(t_step)*P_RF2(t_step)+gRF35(t_step)*P_RF3(t_step)+gRF45(t_step)*P_RF4(t_step);  % RF干扰
    gamma_RF15(t_step) = gRF15(t_step) * P_RF1(t_step) / (interference_RF15(t_step)+noise_RF);  % RF SINR   
    
    % 车辆2到车辆1
    interference_RF21(t_step) = gRF13(t_step)*P_RF3(t_step)+gRF14(t_step)*P_RF4(t_step)+gRF15(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF21(t_step) = gRF12(t_step) * P_RF2(t_step) / (interference_RF21(t_step)+noise_RF);  % RF SINR
    % 车辆2到车辆3
    interference_RF23(t_step) = gRF13(t_step)*P_RF1(t_step)+gRF34(t_step)*P_RF4(t_step)+gRF35(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF23(t_step) = gRF23(t_step) * P_RF2(t_step) / (interference_RF23(t_step)+noise_RF);  % RF SINR
    % 车辆2到车辆4
    interference_RF24(t_step) = gRF14(t_step)*P_RF1(t_step)+gRF34(t_step)*P_RF3(t_step)+gRF45(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF24(t_step) = gRF24(t_step) * P_RF2(t_step) / (interference_RF24(t_step)+noise_RF);  % RF SINR
    % 车辆2到车辆5
    interference_RF25(t_step) = gRF15(t_step)*P_RF1(t_step)+gRF35(t_step)*P_RF3(t_step)+gRF45(t_step)*P_RF4(t_step);  % RF干扰
    gamma_RF25(t_step) = gRF25(t_step) * P_RF2(t_step) / (interference_RF25(t_step)+noise_RF);  % RF SINR   

    % 车辆3到车辆1
    interference_RF31(t_step) = gRF12(t_step)*P_RF2(t_step)+gRF14(t_step)*P_RF4(t_step)+gRF15(t_step)*P_RF5(t_step);  % RF干扰,gRF12(t_step)*P_RF2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_RF31(t_step) = gRF13(t_step) * P_RF3(t_step) / (interference_RF31(t_step)+noise_RF);  % RF SINR
    % 车辆3到车辆2
    interference_RF32(t_step) = gRF12(t_step)*P_RF1(t_step)+gRF24(t_step)*P_RF4(t_step)+gRF25(t_step)*P_RF5(t_step);  % RF干扰，gRF12(t_step)*PRF1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_RF32(t_step) = gRF23(t_step) * P_RF3(t_step) / (interference_RF32(t_step)+noise_RF);  % RF SINR
    % 车辆3到车辆4
    interference_RF34(t_step) = gRF14(t_step)*P_RF1(t_step)+gRF24(t_step)*P_RF2(t_step)+gRF45(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF34(t_step) = gRF34(t_step) * P_RF3(t_step) / (interference_RF34(t_step)+noise_RF);  % RF SINR
    % 车辆3到车辆5
    interference_RF35(t_step) = gRF15(t_step)*P_RF1(t_step)+gRF25(t_step)*P_RF2(t_step)+gRF45(t_step)*P_RF4(t_step);  % RF干扰
    gamma_RF35(t_step) = gRF35(t_step) * P_RF3(t_step) / (interference_RF35(t_step)+noise_RF);  % RF SINR   

    % 车辆4到车辆1
    interference_RF41(t_step) = gRF12(t_step)*P_RF2(t_step)+gRF13(t_step)*P_RF3(t_step)+gRF15(t_step)*P_RF5(t_step);  % RF干扰,gRF12(t_step)*P_RF2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_RF41(t_step) = gRF14(t_step) * P_RF4(t_step) / (interference_RF41(t_step)+noise_RF);  % RF SINR
    % 车辆4到车辆2
    interference_RF42(t_step) = gRF12(t_step)*P_RF1(t_step)+gRF23(t_step)*P_RF3(t_step)+gRF25(t_step)*P_RF5(t_step);  % RF干扰，gRF12(t_step)*PRF1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_RF42(t_step) = gRF24(t_step) * P_RF4(t_step) / (interference_RF42(t_step)+noise_RF);  % RF SINR
    % 车辆4到车辆3
    interference_RF43(t_step) = gRF13(t_step)*P_RF1(t_step)+gRF23(t_step)*P_RF2(t_step)+gRF35(t_step)*P_RF5(t_step);  % RF干扰
    gamma_RF43(t_step) = gRF34(t_step) * P_RF4(t_step) / (interference_RF43(t_step)+noise_RF);  % RF SINR
    % 车辆4到车辆5
    interference_RF45(t_step) = gRF15(t_step)*P_RF1(t_step)+gRF25(t_step)*P_RF2(t_step)+gRF35(t_step)*P_RF3(t_step);  % RF干扰
    gamma_RF45(t_step) = gRF45(t_step) * P_RF4(t_step) / (interference_RF45(t_step)+noise_RF);  % RF SINR   

    % 车辆5到车辆1
    interference_RF51(t_step) = gRF12(t_step)*P_RF2(t_step)+gRF13(t_step)*P_RF3(t_step)+gRF14(t_step)*P_RF4(t_step);  % RF干扰,gRF12(t_step)*P_RF2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_RF51(t_step) = gRF15(t_step) * P_RF5(t_step) / (interference_RF51(t_step)+noise_RF);  % RF SINR
    % 车辆5到车辆2
    interference_RF52(t_step) = gRF12(t_step)*P_RF1(t_step)+gRF23(t_step)*P_RF3(t_step)+gRF24(t_step)*P_RF4(t_step);  % RF干扰，gRF12(t_step)*PRF1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_RF52(t_step) = gRF25(t_step) * P_RF5(t_step) / (interference_RF52(t_step)+noise_RF);  % RF SINR
    % 车辆5到车辆3
    interference_RF53(t_step) = gRF13(t_step)*P_RF1(t_step)+gRF23(t_step)*P_RF2(t_step)+gRF34(t_step)*P_RF4(t_step);  % RF干扰
    gamma_RF53(t_step) = gRF35(t_step) * P_RF5(t_step) / (interference_RF53(t_step)+noise_RF);  % RF SINR
    % 车辆5到车辆4
    interference_RF54(t_step) = gRF14(t_step)*P_RF1(t_step)+gRF24(t_step)*P_RF2(t_step)+gRF34(t_step)*P_RF3(t_step);  % RF干扰
    gamma_RF54(t_step) = gRF45(t_step) * P_RF5(t_step) / (interference_RF54(t_step)+noise_RF);  % RF SINR   

    %VLC 
    % 车辆1到车辆2
    interference_VLC12(t_step) = gVLCLOS23(t_step)*P_VLC3(t_step)+gVLCLOS24(t_step)*P_VLC4(t_step)+gVLCLOS25(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC12(t_step) = gVLCLOS12(t_step) * P_VLC1(t_step) / (interference_VLC12(t_step)+noise_VLC);  % VLC SINR
    % 车辆1到车辆3
    interference_VLC13(t_step) = gVLCLOS23(t_step)*P_VLC2(t_step)+gVLCLOS34(t_step)*P_VLC4(t_step)+gVLCLOS35(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC13(t_step) = gVLCLOS13(t_step) * P_VLC1(t_step) / (interference_VLC13(t_step)+noise_VLC);  % VLC SINR
    % 车辆1到车辆4
    interference_VLC14(t_step) = gVLCLOS24(t_step)*P_VLC2(t_step)+gVLCLOS34(t_step)*P_VLC3(t_step)+gVLCLOS45(t_step)*P_VLC5(t_step);  % VLC干扰，gVLCLOS45和gVLCLOS54数值相同，这里应该是gVLCLOS54(t_step)*PVLC5(t_step)，代表从五号车发出的功率，后面也有类似的
    gamma_VLC14(t_step) = gVLCLOS14(t_step) * P_VLC1(t_step) / (interference_VLC14(t_step)+noise_VLC);  % VLC SINR
    % 车辆1到车辆5
    interference_VLC15(t_step) = gVLCLOS25(t_step)*P_VLC2(t_step)+gVLCLOS35(t_step)*P_VLC3(t_step)+gVLCLOS45(t_step)*P_VLC4(t_step);  % VLC干扰
    gamma_VLC15(t_step) = gVLCLOS15(t_step) * P_VLC1(t_step) / (interference_VLC15(t_step)+noise_VLC);  % VLC SINR   
    
    % 车辆2到车辆1
    interference_VLC21(t_step) = gVLCLOS13(t_step)*P_VLC3(t_step)+gVLCLOS14(t_step)*P_VLC4(t_step)+gVLCLOS15(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC21(t_step) = gVLCLOS12(t_step) * P_VLC2(t_step) / (interference_VLC21(t_step)+noise_VLC);  % VLC SINR
    % 车辆2到车辆3
    interference_VLC23(t_step) = gVLCLOS13(t_step)*P_VLC1(t_step)+gVLCLOS34(t_step)*P_VLC4(t_step)+gVLCLOS35(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC23(t_step) = gVLCLOS23(t_step) * P_VLC2(t_step) / (interference_VLC23(t_step)+noise_VLC);  % VLC SINR
    % 车辆2到车辆4
    interference_VLC24(t_step) = gVLCLOS14(t_step)*P_VLC1(t_step)+gVLCLOS34(t_step)*P_VLC3(t_step)+gVLCLOS45(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC24(t_step) = gVLCLOS24(t_step) * P_VLC2(t_step) / (interference_VLC24(t_step)+noise_VLC);  % VLC SINR
    % 车辆2到车辆5
    interference_VLC25(t_step) = gVLCLOS15(t_step)*P_VLC1(t_step)+gVLCLOS35(t_step)*P_VLC3(t_step)+gVLCLOS45(t_step)*P_VLC4(t_step);  % VLC干扰
    gamma_VLC25(t_step) = gVLCLOS25(t_step) * P_VLC2(t_step) / (interference_VLC25(t_step)+noise_VLC);  % VLC SINR   

    % 车辆3到车辆1
    interference_VLC31(t_step) = gVLCLOS12(t_step)*P_VLC2(t_step)+gVLCLOS14(t_step)*P_VLC4(t_step)+gVLCLOS15(t_step)*P_VLC5(t_step);  % VLC干扰,gVLCLOS12(t_step)*P_VLC2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_VLC31(t_step) = gVLCLOS13(t_step) * P_VLC3(t_step) / (interference_VLC31(t_step)+noise_VLC);  % VLC SINR
    % 车辆3到车辆2
    interference_VLC32(t_step) = gVLCLOS12(t_step)*P_VLC1(t_step)+gVLCLOS24(t_step)*P_VLC4(t_step)+gVLCLOS25(t_step)*P_VLC5(t_step);  % VLC干扰，gVLCLOS12(t_step)*PVLC1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_VLC32(t_step) = gVLCLOS23(t_step) * P_VLC3(t_step) / (interference_VLC32(t_step)+noise_VLC);  % VLC SINR
    % 车辆3到车辆4
    interference_VLC34(t_step) = gVLCLOS14(t_step)*P_VLC1(t_step)+gVLCLOS24(t_step)*P_VLC2(t_step)+gVLCLOS45(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC34(t_step) = gVLCLOS34(t_step) * P_VLC3(t_step) / (interference_VLC34(t_step)+noise_VLC);  % VLC SINR
    % 车辆3到车辆5
    interference_VLC35(t_step) = gVLCLOS15(t_step)*P_VLC1(t_step)+gVLCLOS25(t_step)*P_VLC2(t_step)+gVLCLOS45(t_step)*P_VLC4(t_step);  % VLC干扰
    gamma_VLC35(t_step) = gVLCLOS35(t_step) * P_VLC3(t_step) / (interference_VLC35(t_step)+noise_VLC);  % VLC SINR   

    % 车辆4到车辆1
    interference_VLC41(t_step) = gVLCLOS12(t_step)*P_VLC2(t_step)+gVLCLOS13(t_step)*P_VLC3(t_step)+gVLCLOS15(t_step)*P_VLC5(t_step);  % VLC干扰,gVLCLOS12(t_step)*P_VLC2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_VLC41(t_step) = gVLCLOS14(t_step) * P_VLC4(t_step) / (interference_VLC41(t_step)+noise_VLC);  % VLC SINR
    % 车辆4到车辆2
    interference_VLC42(t_step) = gVLCLOS12(t_step)*P_VLC1(t_step)+gVLCLOS23(t_step)*P_VLC3(t_step)+gVLCLOS25(t_step)*P_VLC5(t_step);  % VLC干扰，gVLCLOS12(t_step)*PVLC1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_VLC42(t_step) = gVLCLOS24(t_step) * P_VLC4(t_step) / (interference_VLC42(t_step)+noise_VLC);  % VLC SINR
    % 车辆4到车辆3
    interference_VLC43(t_step) = gVLCLOS13(t_step)*P_VLC1(t_step)+gVLCLOS23(t_step)*P_VLC2(t_step)+gVLCLOS35(t_step)*P_VLC5(t_step);  % VLC干扰
    gamma_VLC43(t_step) = gVLCLOS34(t_step) * P_VLC4(t_step) / (interference_VLC43(t_step)+noise_VLC);  % VLC SINR
    % 车辆4到车辆5
    interference_VLC45(t_step) = gVLCLOS15(t_step)*P_VLC1(t_step)+gVLCLOS25(t_step)*P_VLC2(t_step)+gVLCLOS35(t_step)*P_VLC3(t_step);  % VLC干扰
    gamma_VLC45(t_step) = gVLCLOS45(t_step) * P_VLC4(t_step) / (interference_VLC45(t_step)+noise_VLC);  % VLC SINR   

    % 车辆5到车辆1
    interference_VLC51(t_step) = gVLCLOS12(t_step)*P_VLC2(t_step)+gVLCLOS13(t_step)*P_VLC3(t_step)+gVLCLOS14(t_step)*P_VLC4(t_step);  % VLC干扰,gVLCLOS12(t_step)*P_VLC2(t_step)指的是车辆2发出到车辆1的干扰功率
    gamma_VLC51(t_step) = gVLCLOS15(t_step) * P_VLC5(t_step) / (interference_VLC51(t_step)+noise_VLC);  % VLC SINR
    % 车辆5到车辆2
    interference_VLC52(t_step) = gVLCLOS12(t_step)*P_VLC1(t_step)+gVLCLOS23(t_step)*P_VLC3(t_step)+gVLCLOS24(t_step)*P_VLC4(t_step);  % VLC干扰，gVLCLOS12(t_step)*PVLC1(t_step)指的是车辆1发出到车辆2的干扰功率
    gamma_VLC52(t_step) = gVLCLOS25(t_step) * P_VLC5(t_step) / (interference_VLC52(t_step)+noise_VLC);  % VLC SINR
    % 车辆5到车辆3
    interference_VLC53(t_step) = gVLCLOS13(t_step)*P_VLC1(t_step)+gVLCLOS23(t_step)*P_VLC2(t_step)+gVLCLOS34(t_step)*P_VLC4(t_step);  % VLC干扰
    gamma_VLC53(t_step) = gVLCLOS35(t_step) * P_VLC5(t_step) / (interference_VLC43(t_step)+noise_VLC);  % VLC SINR
    % 车辆5到车辆4
    interference_VLC54(t_step) = gVLCLOS14(t_step)*P_VLC1(t_step)+gVLCLOS24(t_step)*P_VLC2(t_step)+gVLCLOS34(t_step)*P_VLC3(t_step);  % VLC干扰
    gamma_VLC54(t_step) = gVLCLOS45(t_step) * P_VLC5(t_step) / (interference_VLC45(t_step)+noise_VLC);  % VLC SINR   
   
    % 保存到矩阵中
    gamma_RF(1,2,t_step)=gamma_RF12(t_step);gamma_RF(1,3,t_step)=gamma_RF13(t_step);gamma_RF(1,4,t_step)=gamma_RF14(t_step);gamma_RF(1,5,t_step)=gamma_RF15(t_step);
    gamma_RF(2,1,t_step)=gamma_RF21(t_step);gamma_RF(2,3,t_step)=gamma_RF23(t_step);gamma_RF(2,4,t_step)=gamma_RF24(t_step);gamma_RF(2,5,t_step)=gamma_RF25(t_step);
    gamma_RF(3,1,t_step)=gamma_RF31(t_step);gamma_RF(3,2,t_step)=gamma_RF32(t_step);gamma_RF(3,4,t_step)=gamma_RF34(t_step);gamma_RF(3,5,t_step)=gamma_RF35(t_step);
    gamma_RF(4,1,t_step)=gamma_RF41(t_step);gamma_RF(4,2,t_step)=gamma_RF42(t_step);gamma_RF(4,3,t_step)=gamma_RF43(t_step);gamma_RF(4,4,t_step)=gamma_RF45(t_step);
    gamma_RF(5,1,t_step)=gamma_RF51(t_step);gamma_RF(5,2,t_step)=gamma_RF52(t_step);gamma_RF(5,3,t_step)=gamma_RF53(t_step);gamma_RF(5,4,t_step)=gamma_RF54(t_step);
    gamma_VLC(1,2,t_step)=gamma_VLC12(t_step);gamma_VLC(1,3,t_step)=gamma_VLC13(t_step);gamma_VLC(1,4,t_step)=gamma_VLC14(t_step);gamma_VLC(1,5,t_step)=gamma_VLC15(t_step);
    gamma_VLC(2,1,t_step)=gamma_VLC21(t_step);gamma_VLC(2,3,t_step)=gamma_VLC23(t_step);gamma_VLC(2,4,t_step)=gamma_VLC24(t_step);gamma_VLC(2,5,t_step)=gamma_VLC25(t_step);
    gamma_VLC(3,1,t_step)=gamma_VLC31(t_step);gamma_VLC(3,2,t_step)=gamma_VLC32(t_step);gamma_VLC(3,4,t_step)=gamma_VLC34(t_step);gamma_VLC(3,5,t_step)=gamma_VLC35(t_step);
    gamma_VLC(4,1,t_step)=gamma_VLC41(t_step);gamma_VLC(4,2,t_step)=gamma_VLC42(t_step);gamma_VLC(4,3,t_step)=gamma_VLC43(t_step);gamma_VLC(4,5,t_step)=gamma_VLC45(t_step);
    gamma_VLC(5,1,t_step)=gamma_VLC51(t_step);gamma_VLC(5,2,t_step)=gamma_VLC52(t_step);gamma_VLC(5,3,t_step)=gamma_VLC53(t_step);gamma_VLC(5,4,t_step)=gamma_VLC54(t_step);

    interference_RF(1,2,t_step)=interference_RF12(t_step);interference_RF(1,3,t_step)=interference_RF13(t_step);interference_RF(1,4,t_step)=interference_RF14(t_step);interference_RF(1,5,t_step)=interference_RF15(t_step);
    interference_RF(2,1,t_step)=interference_RF21(t_step);interference_RF(2,3,t_step)=interference_RF23(t_step);interference_RF(2,4,t_step)=interference_RF24(t_step);interference_RF(2,5,t_step)=interference_RF25(t_step);
    interference_RF(3,1,t_step)=interference_RF31(t_step);interference_RF(3,2,t_step)=interference_RF32(t_step);interference_RF(3,4,t_step)=interference_RF34(t_step);interference_RF(3,5,t_step)=interference_RF35(t_step);
    interference_RF(4,1,t_step)=interference_RF41(t_step);interference_RF(4,2,t_step)=interference_RF42(t_step);interference_RF(4,3,t_step)=interference_RF43(t_step);interference_RF(4,4,t_step)=interference_RF45(t_step);
    interference_RF(5,1,t_step)=interference_RF51(t_step);interference_RF(5,2,t_step)=interference_RF52(t_step);interference_RF(5,3,t_step)=interference_RF53(t_step);interference_RF(5,4,t_step)=interference_RF54(t_step);
    interference_VLC(1,2,t_step)=interference_VLC12(t_step);interference_VLC(1,3,t_step)=interference_VLC13(t_step);interference_VLC(1,4,t_step)=interference_VLC14(t_step);interference_VLC(1,5,t_step)=interference_VLC15(t_step);
    interference_VLC(2,1,t_step)=interference_VLC21(t_step);interference_VLC(2,3,t_step)=interference_VLC23(t_step);interference_VLC(2,4,t_step)=interference_VLC24(t_step);interference_VLC(2,5,t_step)=interference_VLC25(t_step);
    interference_VLC(3,1,t_step)=interference_VLC31(t_step);interference_VLC(3,2,t_step)=interference_VLC32(t_step);interference_VLC(3,4,t_step)=interference_VLC34(t_step);interference_VLC(3,5,t_step)=interference_VLC35(t_step);
    interference_VLC(4,1,t_step)=interference_VLC41(t_step);interference_VLC(4,2,t_step)=interference_VLC42(t_step);interference_VLC(4,3,t_step)=interference_VLC43(t_step);interference_VLC(4,5,t_step)=interference_VLC45(t_step);
    interference_VLC(5,1,t_step)=interference_VLC51(t_step);interference_VLC(5,2,t_step)=interference_VLC52(t_step);interference_VLC(5,3,t_step)=interference_VLC53(t_step);interference_VLC(5,4,t_step)=interference_VLC54(t_step);


    % 
    % total_power_RF(t_step)= P_RF1(t_step)+P_RF2(t_step)+P_RF3(t_step)+P_RF4(t_step)+P_RF5(t_step);  % RF总功耗
    % total_power_VLC(t_step)= P_VLC1(t_step)+P_VLC2(t_step)+P_VLC3(t_step)+P_VLC4(t_step)+P_VLC5(t_step); % VLC总功耗


    %% Step 2: 信道切换决策
    % 对每对车辆，比较RF和VLC信道的SINR并选择信道
    % for i = 1:n
    %     for j = i:n
    %         % 比较VLC和RF信道的SINR
    %         if gamma_VLC(i, j,t_step) > gamma_RF(i, j,t_step)
    %             channel_choice(i, j,t_step) = 1;  % 选择VLC信道
    %         else
    %             channel_choice(i, j,t_step) = 0;  % 选择RF信道
    %         end
    %         % 计算传输速率（RF和VLC）
    %         total_rate_RF(i,j) = B_RF * log2(1 + gamma_RF(i,j));  % RF信道传输速率
    %         total_rate_VLC(i,j) = B_VLC * log2(1 + gamma_VLC(i,j));  % VLC信道传输速率
    %     end
    % end  
    % 计算能效
    % EE_RF1(t_step) = sum(total_rate_RF(1,:)) / total_power_RF(t_step);  % 车辆1RF信道能效
    % EE_RF2(t_step) = sum(total_rate_RF(2,:)) / total_power_RF(t_step);  % 车辆2RF信道能效
    % EE_RF3(t_step) = sum(total_rate_RF(3,:)) / total_power_RF(t_step);  % 车辆3RF信道能效
    % EE_RF4(t_step) = sum(total_rate_RF(4,:)) / total_power_RF(t_step);  % 车辆4RF信道能效
    % EE_RF5(t_step) = sum(total_rate_RF(5,:)) / total_power_RF(t_step);  % 车辆5RF信道能效，通过功率分配，使得每辆车的RF信道能效达到最大
    % EE_VLC1(t_step) = sum(total_rate_VLC(1,:)) / total_power_VLC(t_step); % 车辆1VLC信道能效
    % EE_VLC2(t_step) = sum(total_rate_VLC(2,:)) / total_power_VLC(t_step); % 车辆2VLC信道能效
    % EE_VLC3(t_step) = sum(total_rate_VLC(3,:)) / total_power_VLC(t_step); % 车辆3VLC信道能效
    % EE_VLC4(t_step) = sum(total_rate_VLC(4,:)) / total_power_VLC(t_step); % 车辆4VLC信道能效
    % EE_VLC5(t_step) = sum(total_rate_VLC(5,:)) / total_power_VLC(t_step); % 车辆5VLC信道能效，通过功率分配，使得每辆车的VLC信道能效达到最大

  
end

