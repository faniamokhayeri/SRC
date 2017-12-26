function [fpr, tpr, auc, thr] = myroc(scores,lab,nb_thresh)

if nargin < 3 || isempty(nb_thresh)   % consider all scores  
  nb_thresh=[];
end
thresh=sample_scores(scores,nb_thresh);
thresh = sort(unique(thresh),'descend'); 
thresh = [+inf;thresh]; % add pt (0,0).
nb_thresh = length(thresh); 

lab = lab > 0;             %%% Not clear to me
P   = sum( lab);          % # positives.
N   = sum(~lab);          % # negatives.
tpr = zeros(nb_thresh,1);
fpr = zeros(nb_thresh,1);

for i = 1:nb_thresh
    res = (scores >= thresh(i));
    tpr(i) = sum( lab(res));           %%% Not clear to me
    fpr(i) = sum(~lab(res));           %%% Not clear to me
end

if P~=0
    tpr = tpr./P; 
end

if N~=0
    fpr = fpr./N;
end
auc = auroc(fpr,tpr);
thr = thresh;
 
