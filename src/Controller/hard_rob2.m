function J2_hard = hard_rob2(s0, param, dim, T,  Map , str)


[Weights, Biases,~] = param2net(param, dim, []);

num_layer = length(dim)-2;
%%% Generate the trajectory

Preactivate = cell(num_layer,T);
Postactivate  = cell(num_layer,T);
a   = cell(1,T);
s   = cell(1,T);



Preactivate{1,1}=Weights{1}*[s0;0]+Biases{1};
Postactivate{1,1}=tanh(Preactivate{1,1});
for i=2:num_layer
    Preactivate{i,1}=Weights{i}*Postactivate{i-1,1}+Biases{i};
    Postactivate{i,1}=tanh(Preactivate{i,1});
end
a{1}=Weights{num_layer+1}*Postactivate{num_layer,1}+Biases{num_layer+1};
s{1} = model([s0;a{1}]);

for i=2:T
    Preactivate{1,i}=Weights{1}*[s{i-1};i-1]+Biases{1};
    Postactivate{1,i}=tanh(Preactivate{1,i});
    for j=2:num_layer
        Preactivate{j,i}=Weights{j}*Postactivate{j-1,i}+Biases{j};
        Postactivate{j,i}=tanh(Preactivate{j,i});
    end
    a{i}=Weights{num_layer+1}*Postactivate{num_layer,i}+Biases{num_layer+1};
    s{i} = model([s{i-1};a{i}]);
end

Trajectory  = [s0 , [s{:}]];
traj = reshape(Trajectory, [numel(Trajectory),1]);
[~ , J2_hard] = Robustness(str, Map, traj);
end