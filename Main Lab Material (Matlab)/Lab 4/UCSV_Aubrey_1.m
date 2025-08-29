% UCSV model of Stock and Watson (2007) JMCB paper with an exogenous
% variable in the measurement equation
% SV in both the measurement and state equations

clear; clc;
nloop = 15000;
burnin = 5000;

load USinflation.csv
dates = USinflation(:,1);
Y = USinflation(:,2); % CPI inflation rate
U = USinflation(:,4); % Unemployment Rate
T = length(Y);

%% priors
% Prior on the initial condition tau0
tau0bar = 0 ;invVtau = 1/10;
% Prior on the initial condition h
invVh = 1/10; Vh = 1/invVh;
% Prior on the initial condition g
invVg = 1/10; Vg = 1/invVg; 
% Prior on sigh2
nu1 = 5; S1 = .04;
% Prior on sigg2
nu2 = 5; S2 = .04;
% Prior on beta (Unemployment rate)
Vbeta = 10; beta0 = 0;

H = speye(T,T) - sparse(2:T,1:(T-1),ones(1,T-1),T,T);
% initialize the Markov chain
h = log(var(Y)*.8)*ones(T,1);
g = h;
tau0 = 0;
sigh2 = .02;
sigg2 = .02;
beta = 0;
%% initialize for storeage
store_sig = zeros(nloop - burnin,2); 
store_tau = zeros(nloop - burnin,T); 
store_exph = zeros(nloop - burnin,T);
store_expg = zeros(nloop - burnin,T);
store_tau0 = zeros(nloop - burnin,1);
store_beta = zeros(nloop - burnin,1);

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;   
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );
    
for loop = 1:nloop
    %% sample tau   
    alphabar = H\[tau0;zeros(T-1,1)];
    invSigma = sparse(1:T,1:T,exp(-h));
    invOmega = sparse(1:T,1:T,exp(-g));        
    invDtau = H'*invOmega*H + invSigma ;
    Ctau = chol(invDtau,'lower');
    tauhat = invDtau\(invSigma*(Y-U*beta) + H'*invOmega*H*alphabar);
    tau = tauhat + Ctau'\randn(T,1);     
    
    %% sample beta
    ybar = Y-tau;
    Kbeta = 1/Vbeta + U'*invSigma*U;
    beta = Kbeta\(U'*invSigma*ybar + beta0/Vbeta)  + sqrt(1/Kbeta)*randn;
    
    %% sample h
    Ystar = log((Y - tau - U*beta).^2 + .0001 );
    h = SVRW(Ystar,h,sigh2,Vh);
    
     %% sample g
    Tstar = log(( tau - [tau0;tau(1:end-1)]).^2 + .0001 );
    g = SVRW(Tstar,g,sigg2,Vg);
   
    %% sample sigh2
    newS1 = S1 + sum((h(2:end)-h(1:end-1)).^2)/2;
    sigh2 = 1/gamrnd(nu1+(T-1)/2, 1/newS1);    
    
    %% sample sigg2
    newS2 = S2 + sum((g(2:end)-g(1:end-1)).^2)/2;
    sigg2 = 1/gamrnd(nu2+(T-1)/2, 1/newS2);  
    
    %% sample tau0
    Ktau0 = invVtau + exp(-g(1));
    tau0 = Ktau0\(tau(1)/exp(g(1)) + invVtau*tau0bar) + sqrt(1/Ktau0)*randn;
    
    if loop>burnin
        i = loop-burnin;
        store_tau(i,:) = tau';
        store_exph(i,:) = exp(h/2)'; 
        store_expg(i,:) = exp(g/2)'; 
        store_sig(i,:) = [sigh2 sigg2];   
        store_tau0(i,:) = tau0;
        store_beta(i,:) = beta;
    end    
    
    if ( mod( loop, 1000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

post_tau = mean(store_tau)';
post_tau16 = quantile(store_tau,.16)';
post_tau84 = quantile(store_tau,.84)';

figure 
plotx2([post_tau post_tau16 post_tau84 ],dates')
xlim([dates(1) dates(end)])
title('Trend Inflation')
hold on
plot(dates,Y,'k')

figure
subplot(1,2,1)
plotx2([mean(store_exph)' quantile(store_exph,.16,1)' quantile(store_exph,.84,1)' ],dates')
xlim([dates(1) dates(end)])
title('SV Measurement equation')
subplot(1,2,2)
plotx2([mean(store_expg)' quantile(store_expg,.16,1)' quantile(store_expg,.84,1)' ],dates')
xlim([dates(1) dates(end)])
title('SV State equation')

figure
subplot(2,2,1)
histogram(store_sig(:,1))
title('\sigma^2_h')
subplot(2,2,2)
histogram(store_sig(:,2))
title('\sigma^2_g')
subplot(2,2,3)
histogram(store_tau0)
title('\tau_0')
subplot(2,2,4)
histogram(store_beta)
title('\beta')
