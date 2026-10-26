% MATLAB script for adaptive equalization in a 2x2 MIMO time-varying channel
clear all
clear
% 参数设置
N = 64; % OFDM子载波数量
cp_len = 16; % 循环前缀长度
num_symbols = 100; % OFDM符号数量
M = 16; % 16-QAM的调制阶数
Fs = 1e6; % 采样频率，单位Hz

% 生成随机数据符号
data1 = randi([0 M-1], N*num_symbols, 1);
data2 = randi([0 M-1], N*num_symbols, 1);
qam_symbols1 = qammod(data1, M, 'UnitAveragePower', true);
qam_symbols2 = qammod(data2, M, 'UnitAveragePower', true);

% 数据重塑为OFDM符号
qam_symbols1 = reshape(qam_symbols1, N, num_symbols);
qam_symbols2 = reshape(qam_symbols2, N, num_symbols);

% IFFT和循环前缀添加
ofdm_symbols1 = ifft(qam_symbols1);
ofdm_symbols2 = ifft(qam_symbols2);
ofdm_symbols1 = [ofdm_symbols1(end-cp_len+1:end, :); ofdm_symbols1];
ofdm_symbols2 = [ofdm_symbols2(end-cp_len+1:end, :); ofdm_symbols2];

% 模拟2x2时变MIMO信道（例如衰落信道）
h11 = [0.9 0.5 0.3];
h12 = [0.8 0.4 0.2];
h21 = [0.7 0.3 0.1];
h22 = [0.6 0.2 0.1];

tx_signal1 = reshape(ofdm_symbols1, [], 1);
tx_signal2 = reshape(ofdm_symbols2, [], 1);
rx_signal1 = zeros(length(tx_signal1), 1);
rx_signal2 = zeros(length(tx_signal2), 1);

for t = 1:length(tx_signal1)
    ht11 = h11(mod(t-1, 3) + 1); % 从Tx1到Rx1的信道
    ht12 = h12(mod(t-1, 3) + 1); % 从Tx2到Rx1的信道
    ht21 = h21(mod(t-1, 3) + 1); % 从Tx1到Rx2的信道
    ht22 = h22(mod(t-1, 3) + 1); % 从Tx2到Rx2的信道
    if t > 2
        rx_signal1(t) = tx_signal1(t) * ht11 + tx_signal1(t-1) * h11(mod(t-2, 3) + 1) + tx_signal1(t-2) * h11(mod(t-3, 3) + 1) + ...
                        tx_signal2(t) * ht12 + tx_signal2(t-1) * h12(mod(t-2, 3) + 1) + tx_signal2(t-2) * h12(mod(t-3, 3) + 1);
        rx_signal2(t) = tx_signal1(t) * ht21 + tx_signal1(t-1) * h21(mod(t-2, 3) + 1) + tx_signal1(t-2) * h21(mod(t-3, 3) + 1) + ...
                        tx_signal2(t) * ht22 + tx_signal2(t-1) * h22(mod(t-2, 3) + 1) + tx_signal2(t-2) * h22(mod(t-3, 3) + 1);
    elseif t > 1
        rx_signal1(t) = tx_signal1(t) * ht11 + tx_signal1(t-1) * h11(mod(t-2, 3) + 1) + ...
                        tx_signal2(t) * ht12 + tx_signal2(t-1) * h12(mod(t-2, 3) + 1);
        rx_signal2(t) = tx_signal1(t) * ht21 + tx_signal1(t-1) * h21(mod(t-2, 3) + 1) + ...
                        tx_signal2(t) * ht22 + tx_signal2(t-1) * h22(mod(t-2, 3) + 1);
    else
        rx_signal1(t) = tx_signal1(t) * ht11 + tx_signal2(t) * ht12;
        rx_signal2(t) = tx_signal1(t) * ht21 + tx_signal2(t) * ht22;
    end
end

% 添加噪声
snr = 20; % 信噪比，单位dB
rx_signal1 = awgn(rx_signal1, snr, 'measured');
rx_signal2 = awgn(rx_signal2, snr, 'measured');

% 移除循环前缀并重塑
rx_ofdm_symbols1 = reshape(rx_signal1, N+cp_len, num_symbols);
rx_ofdm_symbols2 = reshape(rx_signal2, N+cp_len, num_symbols);
rx_ofdm_symbols1 = rx_ofdm_symbols1(cp_len+1:end, :);
rx_ofdm_symbols2 = rx_ofdm_symbols2(cp_len+1:end, :);

% FFT
rx_qam_symbols1 = fft(rx_ofdm_symbols1);
rx_qam_symbols2 = fft(rx_ofdm_symbols2);

% 信道估计（假设已知导频符号）
H_est11 = fft([h11 zeros(1, N-length(h11))]);
H_est12 = fft([h12 zeros(1, N-length(h12))]);
H_est21 = fft([h21 zeros(1, N-length(h21))]);
H_est22 = fft([h22 zeros(1, N-length(h22))]);

% 均衡
H_est1 = H_est11 + H_est12;
H_est2 = H_est21 + H_est22;
rx_qam_symbols_eq1 = (rx_qam_symbols1 .* conj(H_est11') + rx_qam_symbols2 .* conj(H_est12')) ./ (abs(H_est11').^2 + abs(H_est12').^2);
rx_qam_symbols_eq2 = (rx_qam_symbols1 .* conj(H_est21') + rx_qam_symbols2 .* conj(H_est22')) ./ (abs(H_est21').^2 + abs(H_est22').^2);

% 解调16-QAM符号
rx_data1 = qamdemod(rx_qam_symbols_eq1(:), M, 'UnitAveragePower', true);
rx_data2 = qamdemod(rx_qam_symbols_eq2(:), M, 'UnitAveragePower', true);

% 计算误码率
[~, ber1] = biterr(data1, rx_data1);
[~, ber2] = biterr(data2, rx_data2);
disp(['Bit Error Rate (Rx1): ', num2str(ber1)]);
disp(['Bit Error Rate (Rx2): ', num2str(ber2)]);

% 绘制结果
figure;
subplot(3,2,1);
plot(1:N, abs(H_est11), 'o-');
title('Estimated Channel Frequency Response H_{11}');
xlabel('Subcarrier Index');
ylabel('Magnitude');

subplot(3,2,2);
plot(1:N, abs(H_est12), 'o-');
title('Estimated Channel Frequency Response H_{12}');
xlabel('Subcarrier Index');
ylabel('Magnitude');

subplot(3,2,3);
plot(1:N, abs(H_est21), 'o-');
title('Estimated Channel Frequency Response H_{21}');
xlabel('Subcarrier Index');
ylabel('Magnitude');

subplot(3,2,4);
plot(1:N, abs(H_est22), 'o-');
title('Estimated Channel Frequency Response H_{22}');
xlabel('Subcarrier Index');
ylabel('Magnitude');


scatterplot(rx_qam_symbols1(:));
title('Received 16-QAM Symbols (Before Equalization)');
grid on;


scatterplot(rx_qam_symbols_eq1(:));
title('Equalized 16-QAM Symbols (Rx1)');
grid on;



scatterplot(rx_qam_symbols2(:));
title('Received 16-QAM Symbols (Before Equalization)');
grid on;


scatterplot(rx_qam_symbols_eq2(:));
title('Equalized 16-QAM Symbols (Rx2)');
grid on;
