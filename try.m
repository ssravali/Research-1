% load('R:\research1\Main_research1\train2.mat');
% train_x = data;
% train_y = data_labels;
% [m,n,c] = size(train_x)
% % for i = 1:m
% %     for j = 1:n
% %     end
% % end
% if all(train_x == 0)
%     print('Yes')
% end
f = 1
E = 1
load(strcat('R:\research1\Main_research1\Database\s',num2str(f),'\S',num2str(f),'_E',num2str(E),'_A1'));
[ml,nl] = size(emg);
[m2,n2] = size(acc);
[m3,n3] = size(restimulus);