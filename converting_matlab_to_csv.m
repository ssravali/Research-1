clc
clear all
close all


j = 1;%colume counter.
for f = 1:10
    if f == 3
        continue;
    end
    f
    nf = 0; %Excersie number
    EMG = [];
    ACC = [];
    Y=[];
    j = 1;%colume counter.
    for E = 1:3 
        E 
        load(strcat('R:\research1\Main_research1\Database2\s',num2str(f),'\S',num2str(f),'_E',num2str(E),'_A1'));
        [ml,nl] = size(emg);
        [m2,n2] = size(acc);
        [m3,n3] = size(restimulus);
        nf = nf + 1;
        k = 1;%excersie number in graph
        n = 0;%length of signal and signal pointer
        for i=1:m3%Bit running through emg
            if(restimulus(i)~=0)
                n = n + 1;
                EMG(j,n,1:8) = emg(i,1:8);
                ACC(j,n,1:3) = acc(i,1:3);
            end
            if((restimulus(i) ~=0) && (restimulus(i+1) == 0))
                n = 0;
                if(restimulus(i) ~= k)
                    nf = nf + 1;
                    k = restimulus(i);
                end
                Y(j,1) = nf;  
                j = j + 1;
            end   
        end
    end
    s = sprintf('EMG2%d.mat',f);
    save(s,'EMG');
    s = sprintf('ACC2%d.mat',f);
    save(s,'ACC');
    s = sprintf('Y2%d.mat',f);
    save(s,'Y');
end

%Save("EMG2.mat",'EMG')
%emg1 = EMG(:,:,1);
%save('EMG2_1','emg1');











