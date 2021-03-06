clear all
close all
clc

d_size = [50*1]; % the number of dictionary atoms should be a multiple of 50 for the chokepoint  database

method = 'SRC'; 
% method = 'ESRC'; 
% method = 'RADL'; 
% method = 'SVDL'; 
% method = 'LGR';  
% method = 'ours_wo_dl'; 
% method = 'ours';

load(['data.mat']); % Pixel-based features


label_train_unique = unique(label_train);
class_num = length(label_train_unique);
fprintf('Loading chokepoint subset (%d subjects)... \n', class_num);

% Normalize input data
X = normc(X);
Y = normc(Y);
D = cellfun(@normc, D, 'UniformOutput', false);

aux_num = size(D, 2);
pic_num = size(D{1}, 2)-1;
d_num = round(d_size / pic_num);
D_aux = cell(1, d_num);
for i = 1:d_num
    D_aux{i} = bsxfun(@minus, D{i}(:,2:51), D{i}(:,1));
end
D_aux = cell2mat(D_aux);

% Prepare training and test data for robust auxiliary dictionary learning
Da = zeros(size(D{1}, 1), aux_num);
Ya = cell(1, aux_num);
j = 1;
% for i = [1:12, 25:42, 13:24, 43:60]
for i = [1:2, 6:8, 3:5, 9:10]
    Da(:,j) = D{i}(:,1);
    Ya{j} = D{i}(:,2:7);
    j = j+1;
end

A_init = D_aux;
Ya = cell2mat(Ya);
label_train_aux = kron(1:aux_num, ones(1, 1))';
label_test_aux = kron(1:aux_num, ones(1, pic_num))';

if strcmp(method, 'ours')
    is_dictionary_learning = 1;
else
    is_dictionary_learning = 0;
end

if is_dictionary_learning
    fprintf('Starting robust auxiliary dictionary learning with d_size = %d...\n', size(D_aux,2));
    Daa = RADL(Ya, Da, A_init, d_size, label_train_aux, label_test_aux);
else
    Daa = D_aux;
end

% Start classification
fprintf('Starting classification by %s...\n', method);

switch method
    case {'ours', 'ours_wo_dl'}
        method = 'RADL';
end

verbose = 1;
mean_x = mean(X,2);
testing_num = size(Y,2);
corr_num = 0;

tic
for j = 1:testing_num;
    y = Y(:,j);
    
    switch method
        case 'SRC'
            lambda = 1e-3;
            x_hat = SolveHomotopy(X, y, 'tolerance', 1e-5, 'lambda', lambda, 'maxiteration', 1000, 'isnonnegative', false);
            residual = zeros(class_num, 1);
            for i = 1:class_num
                tidx = find(label_train == label_train_unique(i))';
                residual(i) = norm(y - X(:, tidx)*x_hat(tidx, 1));
            end
        case 'RSC'
            para.lambda = 1e-4;
            para.mean_D = mean_x;
            [x_hat, w] = robust_sparse_coding(X, y, para);
            for i = 1:class_num
                tidx = find(label_train == label_train_unique(i))';
                residual(i) = norm(w.*(y - X(:, tidx)*x_hat(tidx)));
            end
        case 'ESRC'
            lambda = 1e-3;
            x_hat = SolveHomotopy([X Daa], y, 'tolerance', 1e-5, 'lambda', lambda, 'maxiteration', 1000, 'isnonnegative', false);
            beta = x_hat((1:size(Daa,2))+size(X,2));
            residual = zeros(class_num, 1);
            for i = 1:class_num
                tidx = find(label_train == label_train_unique(i))';
                residual(i) = norm(y - X(:, tidx)*x_hat(tidx) - Daa*beta);
            end
        case 'RADL'
            para.mean_D = mean_x;
            para.lambda = 1e-4;
            [x_hat, w] = robust_sparse_coding([X Daa], y, para);
            beta = x_hat((1:size(Daa,2))+size(X,2));
            residual = zeros(class_num, 1);
            for i = 1:class_num
                tidx = find(label_train == label_train_unique(i))';
                residual(i) = norm(w.*(y - X(:, tidx)*x_hat(tidx) - Daa*beta));
            end
    end
    
    switch method
        case {'SRC', 'ESRC', 'RSC', 'RADL'}
            [foo, id] = min(residual);
    end
  %**************************Perform recognition*********************************
  
    identity = label_train_unique(id);
    is_correct = 0;
    if identity == label_test(j,1);
        label_fania=1;
        corr_num = corr_num + 1;
        is_correct = 1;
    else
        label_fania=0;
    end
    score_fania=1/norm(foo);
    [fpr, tpr, auc, vthrs] = myroc(score_fania, label_fania);
    %p1 = pauroc(fpr , tpr);
    p1 = auroc(fpr , tpr);

    p(j)=p1*5;
    
    switch verbose
        case 1
            fprintf('.');
            if mod(j, 10) == 0, fprintf('\n'); end
        case 2
            if is_correct
                fprintf('%4d ', identity);
            else
                fprintf('%3dx ', identity);
            end
            if mod(j, 50) == 0, fprintf('\n'); end
    end
end

acc = corr_num / testing_num;
fprintf('acc: %.4f; incorr: %d', acc, testing_num - corr_num);
fprintf(' (%.2f secs)\n', toc);









