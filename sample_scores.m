function thresh = sample_scores(scores,nb_thresh)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% thresh = sample_scores(scores,nb_thresh)
%
% Uniform sampling of scores into nb_thresh bins.
%
% Last updated by Wael Khreich: 13 February 2010 - 14:43:13
% (wael.khreich@livia.etsmtl.ca)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(nb_thresh)
  thresh=scores;  % consider all scores.
  return
end
thresh = quantile(scores,linspace(0,1,nb_thresh).');
thresh = sort(thresh,'descend');
thresh = [+inf;thresh];
