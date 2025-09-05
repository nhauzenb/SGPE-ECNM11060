%% Analyze Boston Housing using BART %%
%% Wei Zhang, Purdue University %%
%% Dec. 2020 %%
%Modified by Gary Koop, November 2021%
%The notation follows that of the Chipman et al (2010) paper "Bayesian
%Additive Regression Trees" (Annals of Applied Statistics)
% Load the data
clear
filename = 'BostonHousing.csv';
BostonHousing = table2array(readtable(filename));
%This is a data set where the price of the house is the dependent variable
%and the there are 13 explanatory variables (the variable in the last
%column is the price of the house
dataset = [BostonHousing(:,end),BostonHousing(:,1:end-1)];

[n,xandy] = size(dataset);
p = xandy-1; % number of explanatory variables
m = 5; %Number of trees
%% Set the hyperparameters: alpha, beta,mumu,sigmamu,nu,lambda
alpha = 0.95; %alpha and beta are the prior hyperparameters relating to the depth of the tree. 
beta = 2; % These are set to the default choices discusse din the lecture. 
v = 2; %prior degrees of freedom hyperparameter for error variance
nu = 3; %In lecture slides, I did not include this hyperparameter for simplicity (it is called k in the eq. 4 of the Hill et al paper)
q = 0.9; %This relates to the prior for the error variance, which I did not discuss in the lectures 
% See subsection 2.2.4 of Chipman et al if you are interested in details


%The probabilities of growing/pruning/changing for the M-H candidate generating density
pgrow = 1/3;
pprune = 1/3;
pchange = 1/3;


%% Iterations
iter = 400;
burn = 100;


rng('default')
tic;

   
    y=dataset(:,1);
    x=dataset(:,2:p+1);
    % BART
    [Meta, RMSE, ~] = BART(x, y, alpha, beta, m, v, ...
        nu, q, pgrow, pprune, iter, burn);
  
toc;
%Remember the notation in the Hill et al paper involves "Tree structures"
%labeled T and actual parameter values labelled M. These are the "Tree" and "Mu" above which store MCMC draws (after discarding the burn-in) 
%This code uses Matlab structures (see the Matlab help facility)
%E.g. Tree is a structure which contains groups things (usually vectors or matrices) related to the tree
%Groups within the structure are denoted, e.g., Meta.Tree.Terminal 
% This contains all the M-H draws of the Terminal Nodes of the Trees
%Example of a command which will print out the 212th M-H draw of the
%variables in the splitting  for the each of the m trees
Meta(212).Tree.Splitvar
%Following line prints out 111th draw of the mu's (fitted values for each
%tree)
Meta(111).Outcome.mu
