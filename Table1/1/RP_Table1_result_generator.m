%%%% Table
clear all
close all
clc


py_dir = 'python';
RP_dir = pwd;
addpath(genpath( [RP_dir '/../../src'] ) );

value = cell(5,4);

figure()

for i=1:5
    path = [RP_dir  '/../../Test_Pytorch/comparison_vehicle/5inits/init' num2str(i) ];
    cd(path)
    comparison_soft()
    comparison_hard()

    % [status , cmdout] = system([ py_dir ' ' path '/Comparison_CPU_Kurtz_sameinit_vehicle.py']);
    % if ~(status == 0)
    %     cd ..
    %     cd ..
    %     cd ..
    %     cd ..
    %     error([' Your MATLAB was not able to call the python file from MATLAB. Please restart with the content of folder "ADHS_RP\Table1\use_this_if_1_does_not_work"  .' ...
    %         ' This folder segments this script in different scripts and asks you TO RUN THE PYTHON FILE MANUALLY in your desired python IDE. Then' ...
    %         ' once the file training_info.mat is generated from your pthon file, you will run the next scripts. '])
    % end
    % load('training_info.mat')
    % value{i,1} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
    % subplot(5,4,(i-1)*4+1)
    % load(['RP_reviewer_Controller_init' num2str(i) '_Kurtz.mat']) %%% This file will be automatically generated by the above py file.
    % plotting(net)
    % axis equal
    % if i==1
    %     title('Giplin et al.')
    % end
    % clc
    % %%%%
    % system([ py_dir ' ' path '/Comparison_CPU_Abbas_sameinit_vehicle.py']);
    % load('training_info.mat')
    % value{i,2} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
    % subplot(5,4,(i-1)*4+2)
    % load(['RP_reviewer_Controller_init' num2str(i) '_Abbas.mat']) %%% This file will be automatically generated by the above py file.
    % plotting(net)
    % axis equal
    % if i==1
    %     title('Pant et al.')
    % end
    % clc
    % %%%%
    % system([ py_dir ' ' path '/Comparison_CPU_Pavone_sameinit_vehicle.py']);
    % load('training_info.mat')
    % value{i,3} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
    % subplot(5,4,(i-1)*4+3)
    % load(['RP_reviewer_Controller_init' num2str(i) '_STLCG.mat']) %%% This file will be automatically generated by the above py file.
    % plotting(net)
    % axis equal
    % if i==1
    %     title('STLCG')
    % end
    % clc
    % %%%%
    [status , cmdout] = system([ py_dir ' ' path '/Comparison_CPU_Navid_sameinit_vehicle.py']);
    load('training_info.mat')
    value{i,4} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
    subplot(5,4,(i-1)*4+4)
    load(['RP_reviewer_Controller_init' num2str(i) '_LB4TL.mat']) %%% This file will be automatically generated by the above py file.
    plotting(net)
    axis equal
    if i==1
        title('LB4TL')
    end
    clc
end


subject = cell(5,1);
Gilpin_et_al = cell(5,1);
Pant_et_al = cell(5,1);
STLCG = cell(5,1);
LB4TL = cell(5,1);
for i = 1:5

    subject{i} = [num2str(i) '-th guess for theta^0'];
    Gilpin_et_al{i} = value{i,1};
    Pant_et_al{i} = value{i,2};
    STLCG{i} = value{i,3};
    LB4TL{i} = value{i,4};

end


Tbl = table(subject, Gilpin_et_al, Pant_et_al, STLCG, LB4TL);
cd ..
cd ..
cd ..
cd ..
clc
disp(Tbl)
