function [Weights, Biases, Layers] = param2net(param, dim, activation)

index=0;
n_layers = length(dim)-1;
Weights = cell(1, n_layers);
Biases = cell(1, n_layers);
Layers = cell(1, n_layers-1);
for i=1: n_layers
    if i==1
        Weights{i}  = reshape(param(1,index+1:index+dim(1)*dim(2)), [dim(1), dim(2)])';
        index = index+(dim(1))*dim(2);
        Biases{i} = param(1,index+1:index+dim(2))';
        index = index+dim(2);
    else
        L = cell(dim(i),1);
        L(:) = {activation};
        Layers{i-1} = L;
        Weights{i} = reshape(param(1,index+1:index+dim(i)*dim(i+1)), [dim(i), dim(i+1)])';
        index = index+dim(i)*dim(i+1);
        Biases{i} = param(1,index+1:index+dim(i+1))';
        index = index+dim(i+1);
    end
end

end