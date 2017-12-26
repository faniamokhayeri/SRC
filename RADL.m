
% Robust Auxiliary Dictionary Learning
function [A] = RADL(Y, D, A_init, d_size, label_train, label_test)

maxiter = 30;
verbose = 2;
eta = 1;
gamma = sqrt(eta);
obj_rec = zeros(1, maxiter);

lambda = 1e-4;

median_e = 0.6;
mean_D = mean(D, 2);
mu_delta = 8;

[m, n] = size(Y);
class_num = length(unique(label_train));

A = A_init;

if verbose == 2
    obj_pre = 0;
    obj_val = 0;
    %fprintf('---------------------------------------------------------------\n');
    %fprintf('%3s %11s %12s %15s %16s\n', 'it', 'diff', 'obj', 'recon', 'sparsity');
    %fprintf('---------------------------------------------------------------\n');
end
t1 = cputime;

for i = 1:maxiter

    switch verbose
    case 2
        fprintf('iteration %d: sparse coding\n', i);
    case 1
        fprintf('it %d:\n', i);
    end
    X = zeros(size(D,2)+d_size, n);
    Wr = zeros(size(Y));
    Wc = zeros(size(Y));
    for k = 1:n
        Di = zeros(size(D));
        sl = find(label_train == label_test(k));
        Di(:, sl) = D(:, sl);
        if mod(k, 10) == 0, fprintf('.'); end

        y = Y(:,k);
        for j = 1:5
             if i == 1 && j == 1
                 residual_r = (y-mean_D).^2;
                 residual_c = residual_r;
             else
                 if j == 1, x_hat = X(:, k); end
                 residual_r = (y-[D, A]*x_hat).^2;
                 residual_c = (y-[Di, A]*x_hat).^2;
             end
             residual_sort = sort(residual_r);
             delta = residual_sort(ceil(median_e*m));
             mu = mu_delta/delta; 
             w_r = 1./(1+1./exp(-mu*(residual_r-delta)));

             residual_sort = sort(residual_c);
             delta = residual_sort(ceil(median_e*m));
             mu = mu_delta/delta; 
             w_c = 1./(1+1./exp(-mu*(residual_c-delta)));

             temp_wr = w_r./max(w_r);
             index_wr = find(temp_wr > 1e-3);
             % remove the pixels with very small weight
             wri = sqrt(w_r(index_wr));
             Wr_y = wri.*y(index_wr);
             Wr_D = bsxfun(@times, D(index_wr,:), wri);
             Wr_A = bsxfun(@times, A(index_wr,:), wri);

             temp_wc = w_c./max(w_c);
             index_wc = find(temp_wc > 1e-3);
             % remove the pixels with very small weight
             wci = sqrt(w_c(index_wc));
             Wc_y = wci.*y(index_wc);
             Wc_Di = bsxfun(@times, Di(index_wc,:), wci);
             Wc_A = bsxfun(@times, A(index_wc,:), wci);

             x_hat = simplified_rsc([Wr_D, Wr_A; gamma*Wc_Di, gamma*Wc_A], [Wr_y; gamma*Wc_y]);
        end
        X(:,k) = x_hat;
        Wr(:,k) = w_r;
        Wc(:,k) = w_c;
    end
    fprintf('\n');
    X1 = X(1:size(D,2), :);
    X2 = X((1:d_size)+size(D,2),:);

    YD = Y - D*X1;
    YDi = zeros(size(Y));
    for k = 1:n
         sl = find(label_train == label_test(k));
         YDi(:,k) = Y(:,k) - D(:,sl)*X1(sl,k);
    end

    AX2 = A*X2;
    scEr = YD - AX2;
    scEc = YDi - AX2;
    obj_sc = norm(Wr.*scEr, 'fro')^2 + eta*norm(Wc.*scEc, 'fro')^2;
    A_prev = A;

    for k = 1:d_size

        fprintf('o');
        YDA = scEr + A_prev(:,k)*X2(k,:);
        YDiA = scEc + A_prev(:,k)*X2(k,:);
        sX2k  = X2(k,:) .* X2(k,:);

        for j = 1:5

            Er = YDA - A(:,k)*X2(k,:);
            Ec = YDiA - A(:,k)*X2(k,:);

            sEr = sort(Er,1);
            delta = sEr(ceil(median_e*m),:);
            mu = mu_delta ./ delta;
            I = ones(size(Er));
            Wr = I ./ (I + I ./ exp(bsxfun(@times, bsxfun(@minus, Er, delta), -mu)));

            sEc = sort(Ec,1);
            delta = sEc(ceil(median_e*m),:);
            mu = mu_delta ./ delta;
            Wc = I ./ (I + I ./ exp(bsxfun(@times, bsxfun(@minus, Ec, delta), -mu)));

            num = sum(bsxfun(@times, Wr.*YDA, X2(k,:)) + eta*bsxfun(@times, Wc.*YDiA, X2(k,:)), 2);
            den = sum(bsxfun(@times, Wr, sX2k) + eta*bsxfun(@times, Wc, sX2k), 2); 

            idx = find(den > 1e-8);
            asub = num(idx) ./ den(idx);
            A(idx, k) = asub;

            if k < 0
            AX2 = A*X2;
            obj_rc = norm(Wr.*(YD-AX2), 'fro')^2 + eta*norm(Wc.*(YDi-AX2), 'fro')^2;
            fprintf('(%d,%d) %e\n', k, j, obj_rc);
            end
        end
        if mod(k, 6) == 0, fprintf(', '); end
        if mod(k, 60) == 0, fprintf('\n'); end
    end
    if mod(k, 60) ~= 0 && mod(k, 6) == 0
         fprintf('\n');
    end
   
    switch verbose
    case 1
        fprintf('\n');
    case 2
        obj_rc = 0;
        if 0
        AX2 = A*X2;
        Er = YD - AX2;
        Ec = YDi - AX2;

        sEr = sort(Er,1);
        delta = sEr(ceil(median_e*m),:);
        mu = mu_delta ./ delta;
        I = ones(size(Er));
        Wr = I ./ (I + I ./ exp(bsxfun(@times, bsxfun(@minus, Er, delta), -mu)));

        sEc = sort(Ec,1);
        delta = sEc(ceil(median_e*m),:);
        mu = mu_delta ./ delta;
        Wc = I ./ (I + I ./ exp(bsxfun(@times, bsxfun(@minus, Ec, delta), -mu)));
        end

        obj_du = norm(Wr.*Er, 'fro')^2 + eta*norm(Wc.*Ec, 'fro')^2;
        obj_rc = obj_sc + obj_du;
        obj_pre = obj_val;
        obj_sp =  lambda*sum(sum(abs(X), 1));
        obj_val = obj_rc + obj_sp; 
        obj_rec(i) = obj_val;
        %fprintf('%3d: %e, %e, %e, %e\n', i, abs(obj_pre-obj_val), obj_val, obj_rc, obj_sp);
        fprintf('%3d: %e, %e, %e\n', i, obj_val, obj_rc, obj_sp);
    end
end

switch verbose
     case {1, 2}
         fprintf(' (%.2f secs)\n', cputime-t1);
end
