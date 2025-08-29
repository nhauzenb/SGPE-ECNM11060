% Time-varying parameter AR model using precision-based methods of Chan and
% Jeliazkov (2009)
clear all; clc;
nloop = 15000;
burnin = 5000;
load dliprod.dat; % load data
p = 2; % no. of lags
y0 = dliprod(1:p,1);
y = dliprod(p+1:end,1);
T = length(y);

%% construct X matrix
X = zeros(T,p); 
tempY = [y0; y];
for i=1:p
    X(:,(i-1)+1:i) = tempY(p-i+1:end-i,:);
end
X = [ones(T,1) X];
q = size(X,2);
Tq = T*q;
bigX = SURform3(X,1);
H = speye(Tq,Tq) - [ [ sparse(q, (T-1)*q); kron(speye(T-1,T-1), speye(q))] sparse(Tq, q)];

%% Priors
% intial condition for beta0
beta0 = zeros(q,1); invVbeta0 = speye(q)*1/10;

% Prior on the sig2 on the measurement equation
nu0 = 5 ; S0 = .04;
newnu0 = T/2 + nu0;

% Prior on the omega2 on the state equation
nu1 = 5 ; S1 = .04;
newnu1 = (T-1)/2 + nu1;

%% Storage
store_beta=zeros(nloop - burnin,Tq);
store_sig2=zeros(nloop - burnin,1);
store_omega2=zeros(nloop - burnin,q);

%% initialize the Markov chain
invomega2 = ones(q,1)*.01;
invSig2 = .01;

%% MCMC starts here
randn('seed',sum(clock*100)); rand('seed',sum(clock*1000));
disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;

for loop=1:nloop
% sample beta
alphabar = H\[beta0;zeros(Tq-q,1)];
invSigma = sparse(1:T,1:T,invSig2*ones(T,1),T,T);
S=blkdiag(invVbeta0,kron(speye(T-1),diag(invomega2)));
invDbeta = bigX'*invSigma*bigX + H'*S*H;
beta = invDbeta\(bigX'*invSigma*y + H'*S*H*alphabar) + chol(invDbeta,'lower')'\randn(Tq,1);
% betabar = reshape(beta,q,T);

%% sample Sig2
err = (y-bigX*beta).^2 ;
newS1 = S0 + sum(err)/2;
invSig2 = gamrnd( newnu0,1/newS1);
Sig2  = 1/invSig2;
    
%% sample omega2
err2 = reshape(H*beta,q,T);
% err2 = betabar(:,2:end) - betabar(:,1:end-1);
newS2 = S1 + sum(err2(:,2:end).^2,2)/2;
invomega2 = gamrnd(newnu1, 1./newS2);
omega2 = 1./invomega2;


if loop>burnin
    i = loop - burnin;
% store the parameters
store_beta(i,:) = beta';
store_sig2(i,:) = Sig2;
store_omega2(i,:) = omega2';
end  

    if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end
    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

post_beta = reshape(median(store_beta)',q,T);
post_beta16 = reshape(quantile(store_beta,.16,1)',q,T);
post_beta84 = reshape(quantile(store_beta,.84,1)',q,T);

figure
subplot(3,1,1)
plot(1:T,post_beta(1,:),1:T,post_beta16(1,:),1:T,post_beta84(1,:))
title('\beta_1')
subplot(3,1,2)
plot(1:T,post_beta(2,:),1:T,post_beta16(2,:),1:T,post_beta84(2,:))
title('\beta_2')
subplot(3,1,3)
plot(1:T,post_beta(3,:),1:T,post_beta16(3,:),1:T,post_beta84(3,:))
title('\beta_3')
legend('Posterior median','16th quantile','84th quantile')
