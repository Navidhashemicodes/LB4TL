clear all
close all
clc

%%%% case 2


RP_dir = pwd;
addpath(genpath( [RP_dir '/../src'] ) );
path = [RP_dir  '/../Test_MATLAB/Georgios_drone_mission'];
cd(path)
[Run_time , net, net0, STL2NN ] = Test_STL();
clc 
disp(['Training process is done and the run time for training was ' num2str(Run_time) ' seconds'])
m = 10000;
%%% m is set to be 4*10^6 in the paper, but it will take a week for you to
%%% run it, so we suggest go for m=10000. The point is to check whether the maximum residual is negative or not;
RM_verifier(net, STL2NN, m)

%%% We can still find the guarantee proposed in paper with m=10000
%%% simulations but we need to use conformal inference which is not
%%% mentioned in the paper. please un-comment the follwing command to get
%%% the guarantee.
% CP_verifier(net, STL2NN, m)

Result_plotter(net, net0)
cd ..
cd ..


