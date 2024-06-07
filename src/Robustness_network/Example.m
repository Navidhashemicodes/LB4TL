clear all
close all
clc

%%%%%%%%%%%%%%%%%  STL2NN

T=40;

cell_P1 = str2stl('[1,0]');
cell_P2 = str2stl('[2,0]');
cell_P3 = str2stl('[3,0]');
cell_P4 = str2stl('[4,0]');

% cell_P5 = str2stl('[5,0]');
% cell_P6 = str2stl('[6,0]');
% cell_P7 = str2stl('[7,0]');
% cell_P8 = str2stl('[8,0]');
% 
% cell_P9  = str2stl('[9 ,0]');
% cell_P10 = str2stl('[10,0]');
% cell_P11 = str2stl('[11,0]');
% cell_P12 = str2stl('[12,0]');


cell_U =  or_operation( cell_P1,  or_operation( cell_P2,  or_operation( cell_P3 , cell_P4 )));



% cell_G_2 = and_operation( cell_P5, and_operation( cell_P6, and_operation( cell_P7 , cell_P8 )));
% cell_G_1 = and_operation( cell_P9, and_operation( cell_P10,and_operation( cell_P11, cell_P12)));

% cell_ORD = ordered_operation( cell_G_1 , cell_G_2 , 0 , 40) ; 
cell_SFE = G_operation( cell_U , 0 , T);
% 
% cell_phi  = and_operation( cell_ORD , cell_SFE ) ;

str = cell_SFE.cntn;

[s1,s2] = size(str);
TT = char(regexprep(strjoin(reshape(string(str)', [1,s1*s2]))  , ' ',  ''));


%%% U
predicate_transformation.C(1,:)=[ -1  0  0];
predicate_transformation.d(1,1)=  1  ;

predicate_transformation.C(2,:)=[  1  0  0];
predicate_transformation.d(2,1)= -4;

predicate_transformation.C(3,:)=[  0  1  0];
predicate_transformation.d(3,1)= -5;

predicate_transformation.C(4,:)=[  0 -1  0];
predicate_transformation.d(4,1)=  2;



% %%% G_2
% predicate_transformation.C(5,:)=[  1  0  0];
% predicate_transformation.d(5,1)= -3  ;
% 
% predicate_transformation.C(6,:)=[ -1  0  0];
% predicate_transformation.d(6,1)=  4;
% 
% predicate_transformation.C(7,:)=[  0  1  0];
% predicate_transformation.d(7,1)=  0  ;
% 
% predicate_transformation.C(8,:)=[  0 -1  0];
% predicate_transformation.d(8,1)=  1;
% 
% 
% 
% %%% G_1
% predicate_transformation.C(9,:) =[  1  0  0];
% predicate_transformation.d(9,1) =  -5  ;
% 
% predicate_transformation.C(10,:)=[ -1  0  0];
% predicate_transformation.d(10,1)=   6;
% 
% predicate_transformation.C(11,:)=[  0  1  0];
% predicate_transformation.d(11,1)=  -3  ;
% 
% predicate_transformation.C(12,:)=[  0 -1  0];
% predicate_transformation.d(12,1)=   4;



type = 'hard';

net = Logic_Net(TT, predicate_transformation , T, type);

net = Linear_Nonlinear(net);


%%%    Hardcoded rob
n=3;
Traj = rand(n*(T+1),1);
traj = reshape(Traj, [n, T+1]);
Map = predicate_transformation;

rob =  Map.C(1,:)*traj(:,1)+Map.d(1,1);
for j=2:4
    rob =  max( rob, Map.C(j,:)*traj(:,1)+Map.d(j,1) );
end
    

for i=2:T+1
    rob1 =rob;
    rob =  Map.C(1,:)*traj(:,i)+Map.d(1,1);
    for j=2:4
        rob =  max( rob, Map.C(j,:)*traj(:,i)+Map.d(j,1) );
    end
    rob  = min(rob1, rob);
end


rob_hardcode = rob


b=10;
rob_toolbox = NN(net, Traj, b)

