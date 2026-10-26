% MATLAB script for adaptive equalization in a time-varying channel

% Parameters
N = 64; % Number of OFDM subcarriers
cp_len = 16; % Length of cyclic prefix
num_symbols = 100; % Number of OFDM symbols
M = 16; % Modulation order for 16-QAM
mu = 0.01; % Step size for LMS algorithm
Fs = 1e6; % Sampling frequency in Hz

% Generate random data symbols
data = randi([0 M-1], N*num_symbols, 1);
qam_symbols = qammod(data, M, 'UnitAveragePower', true);

% Reshape data into OFDM symbols
qam_symbols = reshape(qam_symbols, N, num_symbols);

% IFFT and cyclic prefix addition
ofdm_symbols = ifft(qam_symbols);
ofdm_symbols = [ofdm_symbols(end-cp_len+1:end, :); ofdm_symbols];

% Simulate time-varying channel (e.g., fading channel)
h0 = [0.9 0.5 0.3];
h1 = [0.8 0.4 0.3];
h2 = [0.7 0.3 0.3];
h = [h0; h1; h2];
tx_signal = reshape(ofdm_symbols, [], 1);
rx_signal = zeros(length(tx_signal), 1);

for t = 1:length(tx_signal)
    ht = h(mod(t-1, 3) + 1, :); % Adjusted for 0-based indexing
    rx_signal(t) = tx_signal(t) * ht(1);
    if t > 1
        rx_signal(t) = rx_signal(t) + tx_signal(t-1) * ht(2);
    end
    if t > 2
        rx_signal(t) = rx_signal(t) + tx_signal(t-2) * ht(3);
    end
end

% Add noise
snr = 20; % Signal-to-noise ratio in dB
rx_signal = awgn(rx_signal, snr, 'measured');

% Remove cyclic prefix and reshape
rx_ofdm_symbols = reshape(rx_signal, N+cp_len, num_symbols);
rx_ofdm_symbols = rx_ofdm_symbols(cp_len+1:end, :);

% FFT
rx_qam_symbols = fft(rx_ofdm_symbols);

% Channel estimation using known pilot symbols (assuming perfect knowledge here)
H_est = fft([h0 zeros(1, N-length(h0))]);

% Calculate frequencies corresponding to each subcarrier
freqs = (0:N-1)*(Fs/N);

% Equalization
rx_qam_symbols_eq = rx_qam_symbols ./ H_est.';

% Demodulate 16-QAM symbols
rx_data = qamdemod(rx_qam_symbols_eq(:), M, 'UnitAveragePower', true);

% Calculate bit error rate
[~, ber] = biterr(data, rx_data);
disp(['Bit Error Rate: ', num2str(ber)]);

% Plot results
figure;
plot(freqs, abs(H_est), 'o-');
title('Estimated Channel Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

figure;
plot(freqs, abs(10*(log(H_est))), 'o-');
title('Estimated Channel Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Magnitude(dB)');

scatterplot(rx_qam_symbols(:));
title('Received 16-QAM Symbols (Before Equalization)');
grid on;


scatterplot(rx_qam_symbols_eq(:));
title('Equalized 16-QAM Symbols');
grid on;
