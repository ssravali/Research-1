
clc
clear all
close all

load('R:\research1\Main_research1\train2.mat');
train_x = data;
train_y = data_labels;
[m1,n1] = size(train_x)
size(train_y)
load('R:\research1\Main_research1\test2.mat')
test_x = data;
test_y = data_labels;
[m2,n2]=size(test_x)
size(test_y)


snr1 = 25
sTx1 = zeros(size(train_x));
stx1 = zeros(size(test_x));
% snr2 = 5
% sTx2 = zeros(size(train_x));
% stx2 = zeros(size(test_x));

for i = 1:m1 
    for j = 1:10
        trainx = reshape(train_x(i,:,j),[2800,1]);
        sTx1(i,:,j) = awgn(trainx,snr1);
        %sTx2(i,:,j) = awgn(trainx,snr2);
    end
end
for i = 1:m2
    for j = 1:10
        testx = reshape(test_x(i,:,j),[2800,1]);
        stx1(i,:,j) = awgn(trainx,snr1);
        %stx2(i,:,j) = awgn(trainx,snr2);
    end
end
figure
plot(train_x(1,:,1))

plot(stx1(1,:,1))
hold on
%plot(stx2(1,:,1))
% train_x = [train_x;sTx1;sTx2];
% train_y = [train_y;train_y;train_y];
% test_x = [test_x;stx1;stx2];
% test_y = [test_y;test_y;test_y];
% size(train_x)
% size(test_x)
train_x = [train_x;sTx1;];
train_y = [train_y;train_y;];
test_x = [test_x;stx1;];
test_y = [test_y;test_y;];
size(train_x)
size(test_x)

% 
% figure
% plot(train_x(1,:,1),'r-')
% hold on
% plot(sTx(1,:,1),'b-')
% figure
% plot(test_x(1,:,1),'r-')
% hold on
% plot(stx(1,:,1),'b-')

save('augtrainx2.mat','train_x');
save("augtrainy2.mat",'train_y');
save('augtestx2.mat','test_x');
save('augtesty2.mat','test_y');