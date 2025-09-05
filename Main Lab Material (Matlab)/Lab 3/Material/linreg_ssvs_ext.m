%% bayesian linear regression with ssvs priors
clear all ; clc
nsim = 10000; burnin = 5000; 
%% Import Gary's growth data
load growth.dat;
%The data set is arranged with data for each country taking up 6 lines
%The following makes it into an N by K matrix
n=72;
rawdat=zeros(n,42);
j=1;
for i=1:n
    rawdat(i,:)= [growth(j,:) growth(j+1,:) growth(j+2,:) ...
            growth(j+3,:) growth(j+4,:) growth(j+5,:)];
    j=j+6;
end

y=rawdat(:,1);
y=(y-mean(y));
% xraw=normalize(rawdat(:,2:42));
xraw=rawdat(:,2:42);
%subtract mean from all regressors as in FLS
mxraw=mean(xraw);
sdxraw=std(xraw);
bigk=size(xraw,2);
for i=1:bigk
    xraw(:,i)= (xraw(:,i) - mxraw(1,i))/sdxraw(1,i);
end

%% Define few things
T = length(y); % no. of observation
X = [ones(T,1) xraw];
k = size(X,2); % no. of regressors
 
   % prior
beta0 = zeros(k,1); iVbeta0 = eye(k)/100;
nu0 = 5;  S0 = .01*(nu0-1);

   % SSVS 
beta = (X'*X)\(X'*y);  
sig2 = sum((y-X*beta).^2)/T;
beta_cov = sig2*((X'*X)\speye(k));
beta_sd = sqrt(diag(beta_cov));

c1 = 0.1*beta_sd; c2=100*beta_sd;
tau1 = c1; % prior variance for tau1
tau2 = c2; % prior variance for tau2

% storage
store_theta = zeros(nsim,k+1);
store_gamma = zeros(nsim,k);

for isim = 1:nsim + burnin
        % sample beta
    Dbeta = (iVbeta0 + X'*X/sig2)\speye(k); % same as inv
    beta_hat = Dbeta*(X'*y/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);

        % sample sig2
    e = y-X*beta;
    sig2 = 1/gamrnd(nu0 + T/2,1/(S0 + e'*e/2));
    
       % sample gamma
    prob = normpdf(beta,0,tau2)./( normpdf(beta,0,tau2) + normpdf(beta,0,tau1));
    unif = rand(k,1);
    gamma = .5*sign(prob-unif) + .5;
    iVbeta0 = diag(1./(gamma.*tau2.^2 + (1-gamma).*tau1.^2));  

        % store the parameters
    if isim > burnin
        isave = isim - burnin;        
        store_theta(isave,:) = [beta' sig2];   
        store_gamma(isave,:) = gamma';    
    end
end

% Selected histogram plots for the beta coefficents
figure
subplot(2,2,1)
histogram(store_theta(:,1))
title('\beta_1')
subplot(2,2,2)
histogram(store_theta(:,3))
title('\beta_3')
subplot(2,2,3)
histogram(store_theta(:,5))
title('\beta_5')
subplot(2,2,4)
histogram(store_theta(:,7))
title('\beta_7')
