cd(path)
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