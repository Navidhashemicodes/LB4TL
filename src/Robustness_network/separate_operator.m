function Tree = separate_operator(str)

%%% str is a string that is the STL formula written based on 'and--->&' and
%%% 'or--->|'. In this function we read the string and separate its
%%% subformulas and locate them on the tree. For example --->  (event1 & event2 )| event3 will be
%%% separated to event1  ,  event2,  event3  while they are on a tree, event1 and event2 will be siblings and their parents is '&', on the other hand event3 is the sibling of '&' and their parent is '|'.


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
            ands{number_and} = str1;
        elseif strcmp(str(j), '|')   %%% and once it faces '|' it collects the event before '|' in a cell array that contains events for disjunctions.
            str1 = str(j1+1:j-1);
            j1 = j;
            number_or = number_or+1;
            ors{number_or} = str1;
        end
    end
end


if length(ands)==0 && length(ors)~=0   %%% We generate the tree like this if all the events are contributing in disjunction
    Tree = tree();
    Tree = addnode(Tree,0,'or');
    ors{number_or+1} = str(j1+1:end);
    for i = 1:length(ors)
        Tree = addnode(Tree,1,ors{i});
    end
elseif length(ands)~=0 && length(ors)~=0  %%% the logically wrong to have an statement like (a & b | c) it should be ((a&b)|c) or (a&(b|c)). Thus its not possible to have something like (a & b | c) when i=0. 
    
    error('Not valid logical expression')

elseif length(ors)==0 && length(ands)~=0   %%% and we generate the tree like this if all the events are contributing in conjunction
    Tree = tree();
    Tree = addnode(Tree,0,'and');
    ands{number_and+1} = str(j1+1:end);
    for i = 1:length(ands)
        Tree = addnode(Tree,1,ands{i});
    end
else
    Tree = tree();
    Tree = addnode(Tree,0,str);
end
    
end