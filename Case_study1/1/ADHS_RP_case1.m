%%%% case 1
clear all
close all
clc
%%%% Assuming that python is installed and recognized by os
py_dir = 'python';
RP_dir = pwd;
addpath(genpath( [RP_dir '/../../src'] ) );
path = [RP_dir  '/../../Test_Pytorch/Georgios_vehicle_navigation_Pytorch'];
cd(path)
comparison_soft()
comparison_hard()
[status, cmdout] = system([ py_dir ' ' RP_dir '/../../Test_Pytorch/Georgios_vehicle_navigation_Pytorch/CPU_Navid_case1_vehicle.py']);
if status == 0
    clc
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


else
    
    disp([" Your MATLAB was not able to call the python file from MATLAB. Please restart with the content of folder ADHS_RP\Case_study1\Use_this_if_1_does_not_work  ."; ...
        " This folder segments this script in different scripts and asks you TO RUN THE PYTHON FILE MANUALLY in your desired python IDE. Then"; ...
        " once the file traininfo.mat is generated from your pthon file, you will run the next scripts. "])

end

cd ..
cd ..