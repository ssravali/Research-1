clear all
clc
close all

load('R:\research1\Main_research1\train2.mat');
train_x = data;
train_y = data_labels;
[m,n,c] = size(train_x)
for i = 1:m
    for j = 1:n
        if
end

subplot(10,1,1)
plot(train_x(1,:,1))
%xlabel("Time")
%ylabel("Amplitude")
title('ch1')
subplot(10,1,2)
plot(train_x(1,:,2))
%xlabel("Time")
%ylabel("Amplitude")
title('ch2')
subplot(10,1,3)
plot(train_x(1,:,3))
%xlabel("Time")
%ylabel("Amplitude")
title('ch3')
subplot(10,1,4)
plot(train_x(1,:,4))
%xlabel("Time")
%ylabel("Amplitude")
title('ch4')
subplot(10,1,5)
plot(train_x(1,:,5))
%xlabel("Time")
%ylabel("Amplitude")
title('ch5')
subplot(10,1,6)
plot(train_x(1,:,6))
%xlabel("Time")
%ylabel("Amplitude")
title('ch6')
subplot(10,1,7)
plot(train_x(1,:,7))
%xlabel("Time")
%ylabel("Amplitude")
title('ch7')
subplot(10,1,8)
plot(train_x(1,:,8))
%xlabel("Time")
%ylabel("Amplitude")
title('ch8')
subplot(10,1,9)
plot(train_x(1,:,9))
%xlabel("Time")
%ylabel("Amplitude")
title('ch9')
subplot(10,1,10)
plot(train_x(1,:,10))
%xlabel("Time")
%ylabel("Amplitude")
title('ch10')