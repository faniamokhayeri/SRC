
% This implementation has benefited a lot from the following paper:
% M. Yang, L. Zhang, J. Yang, and D. Zhang, "Robust Sparse Coding for Face Recognition," CVPR, 2011.
% https://sites.google.com/site/mikemengyang/publications

function [x_hat, w] = robust_sparse_coding(D, y, opt)

% Usage
% para.lambda = 1e-4;
% [x_hat, w] = robust_sparse_coding([X Daa], y, para);

if nargin<3
    opt.mean_D = mean(D,2);
    opt.lambda = 1e-4;
    opt.median_e = 0.6;
    opt.max_iter = 5;
else
    if ~isfield(opt,'lambda') || isempty(opt.lambda), opt.lambda = 1e-4; end;

    if ~isfield(opt,'mean_D') || isempty(opt.mean_D), opt.mean_D = mean(D,2); end;

    if ~isfield(opt,'median_e') || isempty(opt.median_e), opt.median_e = 0.6; end;

    if ~isfield(opt,'max_iter') || isempty(opt.max_iter), opt.max_iter = 5; end;
end

mu_delta = 8; % mu * delta

for nit = 1:opt.max_iter
     if nit == 1
         residual = (y-opt.mean_D).^2;
     else
         residual = (y-D*x_hat).^2;
     end
     residual_sort = sort(residual);
     delta = residual_sort(ceil(opt.median_e*length(residual)));
     mu = mu_delta/delta; 
     w = 1./(1+1./exp(-mu*(residual-delta)));

     temp_w = w./max(w);
     index_w = find(temp_w>1e-3);
     % remove the pixels with very small weight
     W_y = w(index_w).*y(index_w);
     %W_D = repmat(w(index_w),[1 size(D,2)]).*D(index_w,:);
     W_D = bsxfun(@times, D(index_w,:), w(index_w));
     dic_l = size(D,2);
    if 1
      x_i               =  zeros(dic_l,1);
      w_i               =  ones(dic_l,1);
      x_hat                 =  ones(dic_l,1);
      Kratio            =  0.01;
      innerit           =  0;
      yupu_pref         =  1000;
      WDWD              =  W_D'*W_D;
      WDWy              =  W_D'*W_y;
      newlambda         =  opt.lambda*norm(W_y);
      while norm(x_hat-x_i,2)/norm(x_hat,2) > 1e-2 && innerit <=50
             x_i          =  x_hat;
             w_l          =  repmat(w_i,[1 dic_l]);
             w_r          =  w_l';
             z            =  (WDWD.*w_r+newlambda*eye(dic_l)) \ WDWy;
             x_hat = w_i.*z; 
             x_sort       =  sort(abs(x_hat));
             yupu         =  abs(x_sort(ceil(Kratio*dic_l)));
             yupu         =  min(yupu/dic_l, yupu_pref);
             yupu_pref    =  yupu;
             w_i          =  sqrt(x_hat.^2+yupu.^2);
             innerit      =  innerit + 1;
     end
     else
        x_hat = SolveHomotopy(W_D, W_y, 'tolerance', 1e-5, 'lambda', 1e-3, 'maxiteration', 1000, 'isnonnegative', false);
    end
end
