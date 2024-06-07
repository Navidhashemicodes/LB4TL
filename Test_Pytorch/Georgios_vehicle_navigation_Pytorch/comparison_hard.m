function comparison_hard()


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
cell_P12 = str2stl('[12,0]');


T=40;
scale = 1;

cell_U =  or_operation( cell_P1,  or_operation( cell_P2,  or_operation( cell_P3 , cell_P4 )));
cell_G_2 = and_operation( cell_P5, and_operation( cell_P6, and_operation( cell_P7 , cell_P8 )));
cell_G_1 = and_operation( cell_P9, and_operation( cell_P10,and_operation( cell_P11, cell_P12)));

cell_ORD = ordered_operation( cell_G_1 , cell_G_2 , 0 , T) ; 
cell_SFE = G_operation( cell_U , 0 , T);

STR  = and_operation( cell_ORD , cell_SFE ) ;

cell_phi = STR.cntn;


[s1,s2] = size(cell_phi);
TT = regexprep(strjoin(reshape(string(cell_phi)', [1,s1*s2]))  , ' ',  ''); %%% If you print TT you think it is the STL formula written in terms of '&' and '|'.
char_TT = char(TT);



%%% U
predicate_transformation.C(1,:)=[ -1  0  0];
predicate_transformation.d(1,1)=  scale*1  ;

predicate_transformation.C(2,:)=[  1  0  0];
predicate_transformation.d(2,1)= -scale*4;

predicate_transformation.C(3,:)=[  0  1  0];
predicate_transformation.d(3,1)= -scale*5;

predicate_transformation.C(4,:)=[  0 -1  0];
predicate_transformation.d(4,1)=  scale*2;



%%% G_2
predicate_transformation.C(5,:)=[  1  0  0];
predicate_transformation.d(5,1)= -scale*3  ;

predicate_transformation.C(6,:)=[ -1  0  0];
predicate_transformation.d(6,1)=  scale*4;

predicate_transformation.C(7,:)=[  0  1  0];
predicate_transformation.d(7,1)=  scale*0  ;

predicate_transformation.C(8,:)=[  0 -1  0];
predicate_transformation.d(8,1)=  scale*1;



%%% G_1
predicate_transformation.C(9,:) =[  1  0  0];
predicate_transformation.d(9,1) =  -scale*5  ;

predicate_transformation.C(10,:)=[ -1  0  0];
predicate_transformation.d(10,1)=   scale*6;

predicate_transformation.C(11,:)=[  0  1  0];
predicate_transformation.d(11,1)=  -scale*3  ;

predicate_transformation.C(12,:)=[  0 -1  0];
predicate_transformation.d(12,1)=   scale*4;

tic

type = 'hard';
STL2NN    = Logic_Net(char_TT, predicate_transformation , T , type);
STL2NN_sorted = Linear_Nonlinear(STL2NN);

weights = STL2NN_sorted.weights;

Len = length(STL2NN_sorted.layers);
WL = cell(1,Len+1);
Wn = cell(1, Len);
BL = cell(1, Len+1);
Bn = cell(1, Len);
for i=1:Len
    L = STL2NN_sorted.layers{i};
    ss = 0;
    for j=1:length(L)
        
        if strcmp(L{j}, 'purelin')
            ss = ss+1;
        end
    end
    WL{i} = STL2NN_sorted.weights{i}(1:ss     , :);
    Wn{i} = STL2NN_sorted.weights{i}(ss+1:end , :);
    BL{i} = STL2NN_sorted.biases{i}(1:ss     , :);
    Bn{i} = STL2NN_sorted.biases{i}(ss+1:end , :);
end

for i=1:Len
    WL{i} = full(WL{i});
    Wn{i} = full(Wn{i});
end
WL{Len+1} = full(STL2NN_sorted.weights{Len+1});
save('weights_acc', 'WL', 'Wn')

for i=1:Len
    BL{i} = full(BL{i})';
    Bn{i} = full(Bn{i})';
end
BL{Len+1} = full(STL2NN_sorted.biases{Len+1})';
save('biases_acc', 'BL', 'Bn')

save('STL2NN' , 'STL2NN')

end