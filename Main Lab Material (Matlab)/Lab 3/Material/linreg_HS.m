%% bayesian linear regression with horseshoe (global-local) prior
clear all ; clc
nsim = 20000; burnin = 10000; 
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
xraw=rawdat(:,2:42);
% xraw=normalize(rawdat(:,2:42));
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
Nubeta = ones(k,1);
taubeta = 1;
etabeta = 1;
 
   % prior
iVbeta0 = eye(k)/100;
nu0 = 5;  S0 = .01*(nu0-1);

   % initialise a few things 
beta = (X'*X)\(X'*y);  
sig2 = sum((y-X*beta).^2)/T;


% storage
store_theta = zeros(nsim,k+1);
store_taubeta = zeros(nsim,1);
store_lambdabeta = zeros(nsim,k);

for isim = 1:nsim + burnin
        %% sample beta
    Dbeta = (iVbeta0 + X'*X/sig2)\speye(k); % same as inv
    beta_hat = Dbeta*(X'*y/sig2);
    C = chol(Dbeta,'lower');
    beta = beta_hat + C*randn(k,1);

        %% sample sig2
    e = y-X*beta;
    sig2 = 1/gamrnd(nu0 + T/2,1/(S0 + e'*e/2));
    
    %% sample Lambdabeta
    Lambdabeta = 1./gamrnd(1, 1./( 1./Nubeta + 0.5*beta.^2/taubeta ));
    
    %% sample taubeta
    taubeta = 1/gamrnd( 0.5*(k+1), 1/( 1/etabeta + 0.5*sum(sum(beta.^2./Lambdabeta))  ) );
    
     %% sample Nubeta
    Nubeta = 1./gamrnd(1, 1./(1 + 1./Lambdabeta));
    %% samplel etabeta
    etabeta = 1/gamrnd(1, 1/( 1 + 1/taubeta ));
    iVbeta0 = sparse(1:k,1:k,1./(taubeta*Lambdabeta),k,k);
    

      
        % store the parameters
    if isim > burnin
        isave = isim - burnin;        
        store_theta(isave,:) = [beta' sig2];   
        store_taubeta(isave,:) = taubeta'; 
        store_lambdabeta(isave,:) = Lambdabeta;   
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

% Posterior estimates of beta with their associated 68% credible intervals
Posterior_beta = [mean(store_theta(:,1:k))' quantile(store_theta(:,1:k),.16,1)' quantile(store_theta(:,1:k),.84,1)' ];