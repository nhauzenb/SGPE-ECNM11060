%Exercise 6
%--------------------------------------------------------------------
% This program calculates posterior means and standard deviations for
% theta(1) and theta(2) using both Monte Carlo integration and Gibbs
% sampling
%--------------------------------------------------------------------
%#ok<*NOPTS>

%Specify the number of replications
r=5000;
%For Gibbs sampling, discard r0 burnin replications and specify starting value for theta(2)
r0=100;
th2draw=1;
%Specify the correlation between theta(1) and theta(2)
rho_all=[0.1^10,0.5,0.9,0.99,0.999];

% Loop over rho
Tab_MC=[];
Tab_gibbs=[];
for k=1:length(rho_all)

rho=rho_all(k);
SIG= [ 1 rho; rho 1];

%Initialize MC and Gibbs sums to zero 
thgibbs=zeros(2,1);
thgibbs2=zeros(2,1);
thmc=zeros(2,1);
thmc2=zeros(2,1);
keepgraph=[];
for i = 1:r
    %Monte Carlo draw from bivariate Normal posterior
    thmcdraw=norm_rnd(SIG);
      
    %Gibbs sampling draw from 2 univariate Normal posterior conditionals
    th1mean= rho*th2draw;
    th1var = 1 - rho^2;
    th1draw = th1mean + norm_rnd(th1var);
    th2mean= rho*th1draw;
    th2var = 1 - rho^2;
    th2draw = th2mean + norm_rnd(th2var);
    if i>r0
        %For Gibbs discard r0 burnin draws
        %Don't need to do this for MC, but we do to maintain comparability with Gibbs results
        thmc=thmc+thmcdraw;  
        thmc2=thmc2+thmcdraw.^2;
        thdraw = [th1draw; th2draw];
        thgibbs=thgibbs+thdraw;  
        thgibbs2=thgibbs2+thdraw.^2; 
        keepgraph=[keepgraph; [i-r0 thmcdraw(1,1) th1draw]]; %#ok<AGROW>
    end
end

disp('Number of burn in replications')
r0 
disp('Number of included replications')
r1=r-r0

% Mean measures
% for MC
thmc=thmc./r1;
thmc2=thmc2./r1;
thmcsd=sqrt(thmc2 - thmc.^2);
% for Gibbs
thgibbs=thgibbs./r1;
thgibbs2=thgibbs2./r1;
thgibbssd=sqrt(thgibbs2 - thgibbs.^2);

% Plots for theta 1
subplot(2,3,k)
plot(keepgraph(:,1),keepgraph(:,2),'-', keepgraph(:,1),keepgraph(:,3),'-','LineWidth',0.9)
legend('Monte Carlo','Gibbs sampler')
title(['Figure ',num2str(k),': Draws of \theta_{1} for \rho = ',num2str(rho)])
xlabel('Replication Number')

% Build Table as on page 131
mc=[thmc thmcsd]';
Tab_MC=[Tab_MC;[rho,mc(:)']];
gibbs=[thgibbs thgibbssd]';
Tab_gibbs=[Tab_gibbs;[rho,gibbs(:)']];
end

disp('Posterior means and standard deviations of theta1 and theta2')
disp('Monte Carlo Integration')
disp('rho E(theta1|y) Std(theta1|y) E(theta2|y) Std(theta2|y)')
disp(Tab_MC)
disp('')
disp('Gibbs Sampling')
disp('rho E(theta1|y) Std(theta1|y) E(theta2|y) Std(theta2|y)')
disp(Tab_gibbs)

