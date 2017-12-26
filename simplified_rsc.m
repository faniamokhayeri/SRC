salam
function [x_hat] = simplified_rsc(W_D, W_y)

      lambda = 1e-4;
      dic_l = size(W_D,2);
      x_i               =  zeros(dic_l,1);
      w_i               =  ones(dic_l,1);
      x_hat                 =  ones(dic_l,1);
      Kratio            =  0.01;
      innerit           =  0;
      yupu_pref         =  1000;
      WDWD              =  W_D'*W_D;
      WDWy              =  W_D'*W_y;
      newlambda         =  lambda*norm(W_y);
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
