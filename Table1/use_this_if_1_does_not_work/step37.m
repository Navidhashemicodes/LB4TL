cd(path)
load('training_info.mat')
value{i,1} = [ 'Training Runtime: ' num2str(Runtime) ' , Terminating min(rob): ' num2str(rho_min) ];
subplot(5,4,(i-1)*4+1)
load(['RP_reviewer_Controller_init' num2str(i) '_Kurtz.mat']) %%% This file will be automatically generated by the above py file.
plotting(net)
axis equal
if i==1
    title('Giplin et al.')
end
clc
