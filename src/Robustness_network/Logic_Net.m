function net = Logic_Net(str, Map , T , type)

if strcmp(type, 'hard')
    [W, L , B]  =  STR2FFNN(str, Map , T);
elseif strcmp(type, 'smooth')
    [W, L , B]  =  STR2FFNN_smooth(str, Map , T);
end

len = length(L);

net.weights = cell(1, len+1);
net.biases = cell(1, len+1);
net.layers = cell(1, len);
for i=1:len+1
    
    
    net.weights{i} = W{i};
    if i<= len
        net.layers{i} = L{i};
    end
    if i==1
        net.biases{1} = B;
    else
        leng = size(W{i},1);
        net.biases{i} = sparse(leng,1);
    end
    
    
end


end