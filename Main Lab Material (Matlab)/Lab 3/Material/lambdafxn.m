function [distance,lambda] = lambdafxn(nu,q,sigma2,lambda)
    a = nu/2;
    b = nu*lambda/2;
    p = gamcdf(sigma2,a,1/b);
    distance = abs(p - (1-q));
end