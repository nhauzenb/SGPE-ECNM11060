%% univariate stochastic volatility
%% with random walk transition eq

function h = SVRW(Ystar,h,sig2,Vh)
T = length(Ystar);
%% normal mixture
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

%% sample S from a 7-point discrete distribution
temprand = rand(T,1);
q = repmat(pi,T,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,T,1),repmat(sqrtsigi,T,1));
q = q./repmat(sum(q,2),1,7);
S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;

%% sample h
H =  speye(T) - spdiags(ones(T-1,1),-1,T,T);
invSh = spdiags([1/Vh; 1/sig2*ones(T-1,1)],0,T,T);
mu = mi(S)'; invOmega = spdiags(1./sigi(S)',0,T,T);
invDh = H'*invSh*H + invOmega;
Ch = chol(invDh,'lower');
hhat = invDh\(invOmega*(Ystar-mu));
h = hhat + Ch'\randn(T,1);
