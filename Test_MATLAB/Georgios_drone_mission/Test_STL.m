function [Run_time , net, net0, STL2NN ] = Test_STL()

Num=30000;


%%%%%%%%%%%%%%%%%  STL2NN


cell_P1 = str2stl('[1,0]');
cell_P2 = str2stl('[2,0]');
cell_P3 = str2stl('[3,0]');
cell_P4 = str2stl('[4,0]');
cell_P5 = str2stl('[5,0]');



cell_P6 = str2stl('[6,0]');
cell_P7 = str2stl('[7,0]');
cell_P8 = str2stl('[8,0]');
cell_P9  = str2stl('[9 ,0]');
cell_P10 = str2stl('[10,0]');
cell_P11 = str2stl('[11,0]');

T = 35;

cell_U =  or_operation( cell_P1,  or_operation( cell_P2,  or_operation( cell_P3 , or_operation( cell_P4 , cell_P5 ) ) ) );
cell_G = and_operation( cell_P6, and_operation( cell_P7, and_operation( cell_P8 , and_operation( cell_P9, and_operation( cell_P10,cell_P11 ) ) ) ) );

cell_F  = F_operation( cell_G , 32 , T) ;
cell_G  = G_operation( cell_U ,  1 , T);

STR  = and_operation( cell_F , cell_G ) ;

cell_phi = STR.cntn;
[s1,s2] = size(cell_phi);
TT = regexprep(strjoin(reshape(string(cell_phi)', [1,s1*s2]))  , ' ',  ''); %%% If you print TT you think it is the STL formula written in terms of '&' and '|'.
char_TT = char(TT);

%%% U
predicate_transformation.C(1,:)=[  1  0  0  0  0  0];
predicate_transformation.d(1,1)=  -0.17  ;

predicate_transformation.C(2,:)=[  0 -1  0  0  0  0];
predicate_transformation.d(2,1)=   0.2   ;

predicate_transformation.C(3,:)=[  0  1  0  0  0  0];
predicate_transformation.d(3,1)=  -0.35;

predicate_transformation.C(4,:)=[  0  0 -1  0  0  0];
predicate_transformation.d(4,1)=   0;

predicate_transformation.C(5,:)=[  0  0  1  0  0  0];
predicate_transformation.d(5,1)=  -1.2;



%%% G
predicate_transformation.C(6,:)=[  1  0  0  0  0  0];
predicate_transformation.d(6,1)=  -0.05;

predicate_transformation.C(7,:)=[ -1  0  0  0  0  0];
predicate_transformation.d(7,1)=   0.1 ;

predicate_transformation.C(8,:)=[  0  1  0  0  0  0];
predicate_transformation.d(8,1)=  -0.5  ;

predicate_transformation.C(9,:)=[  0 -1  0  0  0  0];
predicate_transformation.d(9,1)=   0.58;

predicate_transformation.C(10,:)=[ 0  0  1  0  0  0];
predicate_transformation.d(10,1)= -0.5  ;

predicate_transformation.C(11,:)=[ 0  0 -1  0  0  0];
predicate_transformation.d(11,1)=  0.7;


b=20;   %%% My experimental results shows b should be low to have acceptable gradients

type = 'smooth';
STL2CBF   = Logic_Net(char_TT, predicate_transformation , T , type);

type = 'hard';
STL2NN    = Logic_Net(char_TT, predicate_transformation , T , type);
    
param=cell(1,Num);

dim=[7,10,3];
load('init10_badinit.mat')
leng =length(dim)-1;
len =0;
for i=1:leng
    len =len + (dim(i)+1)*dim(i+1);
end
param{1} = Param;


rng(0)

s0s = [  0.02  0.02   0.05   0.05   0.035 ;...
         0     0.05   0      0.05   0.025 ;...
                       zeros(4,5)         ];       
tic

averageGrad = [];
averageSqGrad = [];

KK = 5;

for iter = 1:2*Num
    
    disp(['iter = ' num2str(iter) ' . '] );
    
    s0 = s0s(:, floor( rand*KK ) +1 )+0.0005*randn(6,1);
    grad2  = Back_prop_STL(s0, param{iter}, dim, T, b, STL2CBF);

    [param{iter+1} , averageGrad , averageSqGrad  ] = adamupdate(param{iter}, -grad2     , averageGrad,averageSqGrad,iter);
        
    
    if mod(iter,50)==0
        smooth = true;
        hard = true;
        Soft = zeros(1,5); Hard = zeros(1,5);
        for i=1:5
            s0 = s0s(:,i);
            [Soft(i), Hard(i)] = STL_obj(s0, param{iter}, dim, T, STL2CBF, b , STL2NN, smooth , hard);
            disp(['Hard and Soft robustness at iteration ' num2str(iter) ' is ' num2str(Hard(i)) ' and ' num2str(Soft(i)) ' for the s0s(:,' num2str(i) ')']);
        end
        if min(Hard)>0.005
            break;
        end
    end
        
end

Run_time=toc;


[W,B,L] =param2net(param{iter}, dim , 'tanh');
net.weights = W;
net.biases = B;
net.layers = L;

[W,B,L] =param2net(param{1}, dim , 'tanh');
net0.weights = W;
net0.biases = B;
net0.layers = L;

end