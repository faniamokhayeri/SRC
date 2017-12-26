clc
clear all
close all
warning off

nRows=48;
nCols=48;
addpath(genpath('.'));
still_path='Still/';

label_train=[1 3 4 5 6];
data_train = [];
Files=dir([still_path '/*.tif']);
for i=label_train
    FileName=Files(i).name;
    I = imread([still_path FileName]);
    I = imresize(I, [nRows nCols]);
    NT       = double(I);
    I_vec= reshape( NT ,numel(NT),1);
    data_train = [data_train  I_vec];
end

video_label_tests=[1 3 4 5 6 7 9 10 11 12];
data_test = [];
label_test=[];

for video_label_test=video_label_tests
    video_path=['Leaving/C1/' num2str(video_label_test) '/'];
Files=dir([video_path '/*.tif']);
pic_select=randi([1 size(Files,1)],[1 50]);
  for i=pic_select
     FileName=Files(i).name;
     I = imread([video_path FileName]);
     I = imresize(I, [nRows nCols]);
     NT       = double(I);
     I_vec= reshape( NT,numel(NT),1);
     data_test = [data_test  I_vec];
     label_test=[label_test; video_label_test];
  end
end

video_label_Ds=[14 15 16 17 18 19 20 21 22 23];


s=size(video_label_Ds,2);
for k=1:s
    video_path=['Leaving/C1/' num2str(video_label_Ds(k)) '/'];
Files=dir([video_path '/*.tif']);
pic_select=randi([1 size(Files,1)],[1 51]);
data = [];
  for i=pic_select
     FileName=Files(i).name;
     I = imread([video_path FileName]);
     I = imresize(I, [nRows nCols]);
     NT       = double(I);
     I_vec= reshape( NT ,numel(NT),1);
     data = [data  I_vec];
  end
  data_D{k}=data;
end

Y=data_test;
X=data_train;
D=data_D;
label_train=label_train';

save Fania X Y D label_test label_train
