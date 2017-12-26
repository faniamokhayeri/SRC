function pauc = pauroc(fp,tp)
 
% n=size(tp, 1);
% auc=sum((fp(2:n) - fp(1:n-1)).*(tp(2:n)+tp(1:n-1)))./2;

pauc=0;
i=0;
while fp(i+1) < 0.2
    i=i+1;
    pauc=pauc+((fp(i+1)- fp(i))*(tp(i+1)+tp(i)));
end
