function gradient_STL = Back_prop_STL(s0, param, dim, T, b, NS_CBF)


n=size(s0,1);


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


%%% Re-define dynamics and trajectory states based on new Postactivatemple times
%%% We assume the first Postactivatempled state is always s0 and the array Postactivatempled_T
%%% refers to the other Postactivatempled states.


D_f_a = cell(1, T);
D_f_s = cell(1, T);
D_S_step = cell(1, T);
D_J_S = cell(1, T);


model_grad = d_model([s0;a{1}]);
D_f_s{1} = [];  %%% This is wher A_dyn{1} was supposed to be involved but s0 is fixed  
D_f_a{1} = model_grad(:,n+1:end);

D_S_step{1}=0;
D_J_S{1}=zeros(1,n);  %%% This is only an initialization. This value is to be counted 
                      %%% with back propagation please consider this is not with respect 
                      %%% to s0 that is a constant. in fact this is with respect to s{Postactivatempled_T(1)}

for i=2:T
    
    model_grad = d_model([s{i-1};a{i}]);
    D_f_s{i}=model_grad(:,1:n);
    D_f_a{i}=model_grad(:,n+1:end);
    
    der =1;
    for j=num_layer+1:-1:2
        der  = der * Weights{j}*diag(sparse(1-Postactivate{j-1,i}.^2));
    end
    der = der * Weights{1}(:,1:n);
    
    D_S_step{i}=D_f_s{i}+D_f_a{i} * der;
    D_J_S{i}=zeros(1,n);   %%% this is only an initialization, This value is to be counted with back propagation
end


Traj = reshape([s0, [s{:}]] , [n*(T+1) , 1]);
[pre, ~] = NN_middle(NS_CBF, Traj, b);
grad = in_out_der(pre, NS_CBF, b);
E = cell(1, T);
for i=1:T
    E{i}=grad(n*i+1:n*(i+1));
end


%%%%%%% Start of Back Propagation
D_J_S{T} = E{T};
for i=T-1:-1:1
    D_J_S{i} = E{i}+D_J_S{i+1}*D_S_step{i+1};
end


%%%%%% First Layers

D_Postactivate_W = cell(1,num_layer);
D_Postactivate_B = cell(1,num_layer);
D_S_W = cell(1,num_layer+1);
D_S_B = cell(1,num_layer+1);
D_J_Weight = cell(1,num_layer+1);
D_J_Biases = cell(1,num_layer+1);
D_J_W = cell(1,num_layer+1);
D_J_B = cell(1,num_layer+1);
D_J_param = cell(1,num_layer+1);


for param_index=1:num_layer
M = dim(param_index+1);
I=eye(M);
D_Postactivate_W{param_index} = cell(M,T);
D_Postactivate_B{param_index} = cell(M,T);
D_S_W{param_index} = cell(M,T);
D_S_B{param_index} = cell(M,T);

for m=1:M
    if param_index ==1
        D_Postactivate_W{param_index}{m,1}=kron(I(:,m), ((1-(Postactivate{param_index,1}(m,1))^2)*[s0;0])' );  %%% NN is designed for the time steps not the real times in seconds.
    else
        D_Postactivate_W{param_index}{m,1}=kron(I(:,m), ((1-(Postactivate{param_index,1}(m,1))^2)*Postactivate{param_index-1,1})' );
    end
    D_Postactivate_B{param_index}{m,1}=(1-(Postactivate{param_index,1}(m,1))^2)*I(:,m);
    
    der =1;
    for j=num_layer+1:-1:param_index+2
        der  = der * Weights{j}*diag(sparse(1-Postactivate{j-1,1}.^2));
    end
    der = der * Weights{param_index+1};
   
    D_S_W{param_index}{m,1}=D_f_a{1}*der*D_Postactivate_W{param_index}{m,1};
    D_S_B{param_index}{m,1}=D_f_a{1}*der*D_Postactivate_B{param_index}{m,1};
    
    for i=2:T
        if param_index == 1
            D_Postactivate_W{param_index}{m,i}=kron(I(:,m), ((1-(Postactivate{param_index,i}(m,1))^2)*[s{i-1};i-1])' );
        else
            D_Postactivate_W{param_index}{m,i}=kron(I(:,m), ((1-(Postactivate{param_index,i}(m,1))^2)*Postactivate{param_index-1,i})' );
        end
        D_Postactivate_B{param_index}{m,i}=(1-(Postactivate{param_index,i}(m,1))^2)*I(:,m);
        
        der =1;
        for j=num_layer+1:-1:param_index+2
            der  = der * Weights{j}*diag(sparse(1-Postactivate{j-1,i}.^2));
        end
        der = der * Weights{param_index+1};
        
        D_S_W{param_index}{m,i}=D_f_a{i}*der*D_Postactivate_W{param_index}{m,i}+D_f_s{i}*D_S_W{param_index}{m,i-1};
        D_S_B{param_index}{m,i}=D_f_a{i}*der*D_Postactivate_B{param_index}{m,i}+D_f_s{i}*D_S_B{param_index}{m,i-1};
    end
end


D_J_Weight{param_index}=[];
D_J_Biases{param_index}=[];
D_J_W{param_index} = cell(1,M);
D_J_B{param_index} = cell(1,M);
for m=1:M
    D_J_W{param_index}{m}=zeros(1,dim(param_index));
    D_J_B{param_index}{m}=zeros(1,1);
    for i=1:T
        D_J_W{param_index}{m}=D_J_W{param_index}{m}+D_J_S{i}*D_S_W{param_index}{m,i};
        D_J_B{param_index}{m}=D_J_B{param_index}{m}+D_J_S{i}*D_S_B{param_index}{m,i};
    end
    D_J_Weight{param_index}=[D_J_Weight{param_index},D_J_W{param_index}{m}];
    D_J_Biases{param_index}=[D_J_Biases{param_index},D_J_B{param_index}{m}];
end
D_J_param{param_index}=[D_J_Weight{param_index},D_J_Biases{param_index}]; 
end






%%%%% Last Layer
P =dim(end);
I=eye(P);
for p=1:P
    D_S_W{num_layer+1}{p,1}=D_f_a{1}*kron(I(:,p) , Postactivate{num_layer,1}');
    D_S_B{num_layer+1}{p,1}=D_f_a{1}*I(:,p);
    for i=2:T
        D_S_W{num_layer+1}{p,i}=D_f_a{i}*kron(I(:,p) , Postactivate{num_layer,i}')+D_f_s{i}*D_S_W{num_layer+1}{p,i-1};
        D_S_B{num_layer+1}{p,i}=D_f_a{i}*I(:,p)+D_f_s{i}*D_S_B{num_layer+1}{p,i-1};
    end
end


D_J_Weight{num_layer+1}=[];
D_J_Biases{num_layer+1}=[];
for p=1:P
    D_J_W{num_layer+1}{p}=zeros(1,dim(end-1));
    D_J_B{num_layer+1}{p}=zeros(1,1);
    for i=1:T
        D_J_W{num_layer+1}{p}=D_J_W{num_layer+1}{p}+D_J_S{i}*D_S_W{num_layer+1}{p,i};
        D_J_B{num_layer+1}{p}=D_J_B{num_layer+1}{p}+D_J_S{i}*D_S_B{num_layer+1}{p,i};
    end
    D_J_Weight{num_layer+1}=[D_J_Weight{num_layer+1},D_J_W{num_layer+1}{p}];
    D_J_Biases{num_layer+1}=[D_J_Biases{num_layer+1},D_J_B{num_layer+1}{p}];
end

D_J_param{num_layer+1}=[D_J_Weight{num_layer+1},D_J_Biases{num_layer+1}];


gradient_STL = zeros(size(param));
index = 0;
for i=1:num_layer+1
    the_length = size(D_J_param{i} , 2);
    gradient_STL(index+1:index+the_length) = D_J_param{i};
    index = index+the_length;
end

end


