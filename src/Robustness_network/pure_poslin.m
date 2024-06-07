function [Weights, layers] = pure_poslin(type, num_inputs)

%%%%   This function gets the number of inputs who are going to contribute in an max or min process and returns the NN (weights,layers) who returns the min or the max. Consider the parameter 'type' can be 'max' or 'min' 
%%%%   First of all, I wanna make it clear, why this function is recursive,
%%%%   Assume you have written a function that do this for 2 inputs. In addition we have another function that returns a neural network for 3 inputs. what about 4? for 4 inputs we pair the inputs into piars of 2 and take 
%%%%   the 'min/max' process for each pair. So clearly in the next step we need to process 'min/max' for 2 elements which we had a function before. What about 5? Again we pair all of these 5 inputs and take 'min/max' of each pair.
%%%%   Then the problem will be reduced to take a 'min/max' from 3 elements, which we had a function for this before. We can continue this process for an arbitrary number of inputs and the problem is reducible to find the 'min/max'
%%%%   of 3 or 2. The following lines of codes implements this idea and generates NN for 'min/max' process in a recursive way.



min_mapper1 = [1 0; 1 -1 ];
max_mapper1 = [0 1; 1 -1 ];
min_mapper2 = [1 -1];
max_mapper2 = [1  1];
if num_inputs==2
    [Weights, layers] = Weights_for_2(type);
elseif num_inputs==3
    [Weights, layers] = Weights_for_3(type);
else
    if mod(num_inputs,2)==0
        number = num_inputs/2;
        layers{1} = cell(num_inputs, 1);
        for i=1:num_inputs
            if mod(i,2)==1
                layers{1}(i) = {'purelin'};
            else
                layers{1}(i) = {'poslin'};
            end
        end
        if strcmp(type,'min')
            Weights{1} = sparse(kron(speye(number), min_mapper1));
            Weights{2} = sparse(kron(speye(number), min_mapper2));
        elseif strcmp(type,'max')
            Weights{1} = sparse(kron(speye(number), max_mapper1));
            Weights{2} = sparse(kron(speye(number), max_mapper2));
        end
        [Weights_remainder , layers_remainder]  =  pure_poslin(type, number);
    else
        number = floor(num_inputs/2);
        layers{1} = cell(num_inputs, 1);
        for i=1:num_inputs-1
            if mod(i,2)==1
                layers{1}(i) = {'purelin'};
            else
                layers{1}(i) = {'poslin'};
            end
        end
        layers{1}(num_inputs) = {'purelin'};   %%% I know this is the same in overall but this is more coceptual
        if strcmp(type,'min')
            Weights{1} = sparse(blkdiag(kron(speye(number), min_mapper1) , 1));        
            Weights{2} = sparse(blkdiag(kron(speye(number), min_mapper2) , 1));
        elseif strcmp(type,'max')
            Weights{1} = sparse(blkdiag(kron(speye(number), max_mapper1) , 1));
            Weights{2} = sparse(blkdiag(kron(speye(number), max_mapper2) , 1));
        end
        [Weights_remainder , layers_remainder]  =  pure_poslin(type, number+1);
    end
    for i = 1:length(Weights_remainder)
        if i==1
            Weights{2} = sparse(Weights_remainder{1}*Weights{2});
        else
            Weights{i+1} = sparse(Weights_remainder{i});
        end
        if i< length(Weights_remainder)
            layers{i+1} = layers_remainder{i};
        end
    end
end
end




function [Weights, layers] = Weights_for_2(type)

min_mapper1 = [1 0; 1 -1 ];
max_mapper1 = [0 1; 1 -1 ];
min_mapper2 = [1 -1];
max_mapper2 = [1  1];

layers{1} = cell(2, 1);
layers{1}(1) = {'purelin'};
layers{1}(2) = {'poslin'};
if strcmp(type, 'min')
    Weights{1} = min_mapper1;
    Weights{2} = min_mapper2;
elseif strcmp(type,'max')
    Weights{1} = max_mapper1;
    Weights{2} = max_mapper2;
end
end



function [Weights, layers] = Weights_for_3(type)

min_mapper1 = [1 0; 1 -1 ];
max_mapper1 = [0 1; 1 -1 ];
min_mapper2 = [1 -1];
max_mapper2 = [1  1];

layers{1} = cell(2+1, 1);
layers{1}(1) = {'purelin'};
layers{1}(2) = {'poslin'};
layers{1}(3) = {'purelin'};


if strcmp(type, 'min')
    Weights{1} = blkdiag(min_mapper1 , 1);
    Weights{2} = min_mapper1*blkdiag(min_mapper2,1);
elseif strcmp(type,'max')
    Weights{1} = blkdiag(max_mapper1 , 1);
    Weights{2} = max_mapper1*blkdiag(max_mapper2,1);
end
layers{2} = cell(2,1);
layers{2}(1) = {'purelin'};
layers{2}(2) = {'poslin'};
if strcmp(type, 'min')
    Weights{3} = min_mapper2;
elseif strcmp(type,'max')
    Weights{3} = max_mapper2;
end
end