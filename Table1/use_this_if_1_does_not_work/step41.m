cd(path)
load('training_info.mat')
value{i,3} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
subplot(5,4,(i-1)*4+3)
load(['RP_reviewer_Controller_init' num2str(i) '_STLCG.mat']) %%% This file will be automatically generated by the above py file.
plotting(net)
axis equal
if i==1
    title('STLCG')
end
clc


