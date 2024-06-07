function [W, L , B]  =  STR2FFNN_smooth(str, Map , T)


i = 0;
ands = [];
ors = [];
j1 = 0;
number_and = 0;
number_or  = 0;

for j = 1:length(str)   %%% reads the str word by word
    if strcmp(str(j),'(')   %%% this if command makes it certain that we are not inside an open paranthesis. when i=0 then we are ok, but when i~=0 then we are inside an open paranthesis.
        i = i+1;
    elseif strcmp(str(j), ')')
        i = i-1;
    end
    
    if i==0   %%% assume 'i=0' then when we face '&' we pick up the event before '&' and store it in a cell array that collects the events for a conjunction.
        if strcmp(str(j),'&')
            str1 = str(j1+1:j-1);
            j1 = j;
            number_and = number_and+1;
            ands{number_and} = str1; %#ok<AGROW>
        elseif strcmp(str(j), '|')   %%% and once it faces '|' it collects the event before '|' in a cell array that contains events for disjunctions.
            str1 = str(j1+1:j-1);
            j1 = j;
            number_or = number_or+1;
            ors{number_or} = str1; %#ok<AGROW>
        end
    end
end


if isempty(ands) && ~isempty(ors)   %%% include the remainder
    ors{number_or+1} = str(j1+1:end);
    number_or = number_or+1;
    Ws = cell(1,number_or);
    Ls = cell(1,number_or);
    Bs = cell(1, number_or);
    H = sparse(1, number_or);
    [W_add, L_add] = pure_smooth_poslin('max' , number_or); 
    add_L = length(L_add);
    [Ws{1}, Ls{1}, Bs{1}] = STR2FFNN_smooth(unpack(ors{1}), Map , T);
    H(1) = length(Ls{1});
    for i=2:number_or
        [Ws{i}, Ls{i}, Bs{i}] = STR2FFNN_smooth(unpack(ors{i}), Map , T);
        H(i) = length(Ls{i});
    end
    Height = max(H);
    for i=1:number_or
        if H(i)<Height
            for j = H(i)+1:Height
                Ws{i}{j+1} = 1;
                Ls{i}{j} = {'purelin'};
            end
        end
    end
    W = cell(1, Height+1+add_L);
    L = cell(1, Height+add_L);
    B =[];
    for j = 1: Height+1
        W{j} = [];
        L{j} = [];
        for i=1:number_or
            if j==1
                W{j} = sparse([W{j}; Ws{i}{j}]);
                B = [B ; Bs{i}];
            else
                W{j} = sparse(blkdiag(W{j}, Ws{i}{j}));
            end
            if j<=Height
                L{j} = [L{j} ; Ls{i}{j}];
            end
        end
    end
    W{Height+1} = W_add{1}*W{Height+1};
    W(Height+2:Height+1+add_L)= W_add(2:end);
    L(Height+1:Height+add_L) = L_add;  
    if Height==0
        B = W_add{1}*B;
    end
    
elseif ~isempty(ands) && ~isempty(ors)  %%% the logically wrong to have an statement like (a & b | c) it should be ((a&b)|c) or (a&(b|c)). Thus its not possible to have something like (a & b | c) when i=0. 
    
    error('Not valid logical expression')

elseif isempty(ors) && ~isempty(ands)   %%% and we generate the tree like this if all the events are contributing in conjunction
    ands{number_and+1} = str(j1+1:end);
    number_and = number_and+1;
    Ws = cell(1,number_and);
    Ls = cell(1,number_and);
    Bs = cell(1, number_and);
    H = sparse(1, number_and);
    [W_add, L_add] = pure_smooth_poslin('min' , number_and); 
    add_L = length(L_add);
    [Ws{1}, Ls{1} , Bs{1}] = STR2FFNN_smooth(unpack(ands{1}), Map , T);
    H(1) = length(Ls{1});
    for i=2:number_and
        [Ws{i}, Ls{i}, Bs{i}] = STR2FFNN_smooth(unpack(ands{i}), Map , T);
        H(i) = length(Ls{i});
    end
    Height = max(H);
    for i=1:number_and
        if H(i)<Height
            for j = H(i)+1:Height
                Ws{i}{j+1} = 1;
                Ls{i}{j} = {'purelin'};
            end
        end
    end
    W = cell(1, Height+1+add_L);
    L = cell(1, Height+add_L);
    B = [];
    for j = 1: Height+1
        W{j} = [];
        L{j} = [];
        for i=1:number_and
             if j==1
                W{j} = sparse([W{j}; Ws{i}{j}]);
                B = [B ; Bs{i}];
            else
                W{j} = sparse(blkdiag(W{j}, Ws{i}{j}));
            end
            if j<=Height
                L{j} = [L{j} ; Ls{i}{j}];
            end
        end
    end
    W{Height+1} = W_add{1}*W{Height+1};
    W(Height+2:Height+1+add_L)= W_add(2:end);
    L(Height+1:Height+add_L) = L_add;
    if Height==0
        B = W_add{1}*B;
    end
    
else
    Info = str2num(str); %#ok<ST2NM>
    what_predicate = Info(1);
    what_time = Info(2);
    n= size(Map.C,2);
    W = cell(1,1);
    Wei = sparse(1, n*(T+1));
    Wei(1, what_time*n+1:what_time*n+n) = Map.C(what_predicate, :);
    B = Map.d(what_predicate, 1);
    W{1} = Wei;
    L = [];
end


end