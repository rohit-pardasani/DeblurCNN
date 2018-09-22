% Full Data Create
clc;
clear;
S = 35;
mydir = pwd;
saveFolder = strcat(mydir,'/DeblurData/');
baseFolder = strcat(mydir,'/MyDataset/');
p = 1;
for i=1:480
    baseFile = char(strcat(baseFolder,'P',string(i),'.jpg'));
    blurFile = char(strcat(baseFolder,'Pb',string(i),'.jpg'));
    I0 = double(imread(baseFile))./255;
    Ib = double(imread(blurFile))./255;
    [N,M,C] = size(I0);
    for k = 1:1000    
        rs = randi([1,N-S+1]);
        re = rs + S - 1;
        cs = randi([1,M-S+1]);
        ce = cs + S - 1;
        X = Ib(rs:re,cs:ce,:);
        Y = I0(rs:re,cs:ce,:);
        Z = X - Y;
        saveFileX = strcat(saveFolder,'X',string(p),'.mat');
        %saveFileY = strcat(saveFolder,'Y',string(p),'.mat');
        saveFileZ = strcat(saveFolder,'Z',string(p),'.mat');
        save(saveFileX,'X')
        %save(saveFileY,'Y')
        save(saveFileZ,'Z')
        p = p + 1
    end
end
