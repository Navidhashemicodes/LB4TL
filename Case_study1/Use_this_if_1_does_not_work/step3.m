cd(path)

load('traininfo.mat')
disp(['Training process is done and the run time for training was ' num2str(Runtime_train) ' seconds'])
m = 10000;
%%% m is set to be 4*10^6 in the paper for verification with risk measure,
%%% but it will take a week for you to run it, so we suggest go for m=10000.
%%% The point is to check whether the maximum residual is negative or not;
load('STL2NN.mat')
%%% The following file will be automatically generated via the .py file called
%%% above, do not change the name of the file.
load('RP_reviewer_Controller_LB4TL_case1.mat')  
RM_verifier(STL2NN, net, m)

%%% We can still find the guarantee proposed in paper with m=10000
%%% simulations but we need to use conformal inference which is not
%%% mentioned in the paper. please un-comment the follwing command 
%%% ( CP_verifier() )  to get the guarantee.
% CP_verifier(net, STL2NN, m)

Result_plotter(net)

cd ..
cd ..