function  [winner, value]  = Robustness(str , Map , traj)

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
    the_value = zeros(1,number_or);
    the_winner = cell(1,number_or);
    [the_winner{1} , the_value(1)] = Robustness(unpack(ors{1}), Map, traj);
    value = the_value(1);
    for i=2:number_or
        [the_winner{i} , the_value(i)] = Robustness(unpack(ors{i}), Map, traj);
        value = max(value , the_value(i));
    end
    index = find(the_value == value);
    index = index(1);
    winner = the_winner{index(1)};
        
    
elseif ~isempty(ands) && ~isempty(ors)  %%% the logically wrong to have an statement like (a & b | c) it should be ((a&b)|c) or (a&(b|c)). Thus its not possible to have something like (a & b | c) when i=0. 
    
    error('Not valid logical expression')

elseif isempty(ors) && ~isempty(ands)   %%% and we generate the tree like this if all the events are contributing in conjunction
    ands{number_and+1} = str(j1+1:end);
    number_and = number_and+1;
    the_value = zeros(1,number_and);
    the_winner = cell(1,number_and);
    [the_winner{1} , the_value(1)] = Robustness(unpack(ands{1}), Map, traj);
    value = the_value(1);
    for i=2:number_and
        [the_winner{i} , the_value(i)] = Robustness(unpack(ands{i}), Map, traj);
        value = min(value , the_value(i));
    end
    index = find(the_value == value);
    index = index(1);
    winner = the_winner{index(1)};
else
    winner = str;
    Info = str2num(str); %#ok<ST2NM>
    what_predicate = Info(1);
    what_time = Info(2);
    n= size(Map.C,2);
    state = traj(what_time*n+1:what_time*n+n);
    value  = Map.C(what_predicate, :)*state+Map.d(what_predicate,1);
end


end