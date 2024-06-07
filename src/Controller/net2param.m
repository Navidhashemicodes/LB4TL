function  param = net2param(net, dim)
leng =length(dim)-1;
len =0;
for i=1:leng
    len =len + (dim(i)+1)*dim(i+1);
end

param = zeros(1, len);

index = 0;
for i=1:leng
    W = net.weights{i};
    B=net.biases{i};
    for j=1:dim(i+1)
        param(index+1:index+dim(i)) = W(j,:);
        index = index + dim(i);
    end
    param(index+1:index+dim(i+1)) = B';
    index = index+dim(i+1);
end
    

end

