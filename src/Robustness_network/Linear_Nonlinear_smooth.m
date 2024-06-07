function Net=Linear_Nonlinear_smooth(net)

Net = net;
len = length(net.layers);

W1 = net.weights{1};
for i = 1:len
    W2 = net.weights{i+1};
    b = net.biases{i};
    L = net.layers{i};
    ss = 0;
    pure_location = [];
    W1_new = W1;
    W2_new = W2;
    b_new = b;
    L_new = L;
    for j = 1:length(L)
        if strcmp(L(j), 'purelin')
           ss = ss+1; 
           pure_location = [pure_location j];
           W1_new(ss,:) = W1(j,:);
           b_new(ss,1) = b(j,1);
           L_new(ss) = L(j);
           W2_new(:,ss) = W2(:,j);
        end
    end
    TT = 1:length(L);
    nonlin_location = setdiff(TT,pure_location);
    index = length(pure_location);
    for j = 1:length(nonlin_location)
        W1_new(index+j,:) = W1(nonlin_location(j), :);
        b_new(index+j,1)  = b(nonlin_location(j),1)  ;
        L_new(index+j)    = L(nonlin_location(j))    ;
        W2_new(:,index+j) = W2(:, nonlin_location(j));
    end

    W1_new2 = W1_new;
    W2_new2 = W2_new;
    b_new2  = b_new;
    L_new2  = L_new;
    soft_location = [];
    ss = index;
    for j = index+1:length(L)
        if strcmp(L_new(j), 'softplus')
           ss = ss+1; 
           soft_location = [soft_location j];
           W1_new2(ss,:) = W1_new(j,:);
           b_new2(ss,1) = b_new(j,1);
           L_new2(ss) = L_new(j);
           W2_new2(:,ss) = W2_new(:,j);
        end
    end
    TT = index+1:length(L);
    swish_location = setdiff(TT, soft_location);
    index2 = index+length(soft_location);
    for j = index+1:length(swish_location)
        W1_new2(index2+j,:) = W1_new(swish_location(j), :);
        b_new2(index2+j,1)  = b_new(swish_location(j),1)  ;
        L_new2(index2+j)    = L_new(swish_location(j))    ;
        W2_new2(:,index2+j) = W2_new(:, swish_location(j));
    end

    Net.weights{i} = W1_new2;
    Net.biases{i} = b_new2;
    Net.layers{i} = L_new2;
    W1 = W2_new2;
end
Net.weights{len+1} = W2_new2;
Net.biases{len+1} = net.biases{len+1};


end
