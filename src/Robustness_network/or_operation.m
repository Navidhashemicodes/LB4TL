function  STR  = or_operation(STR1,STR2)

string1 = STR1.cntn;
string2 = STR2.cntn;
type1 = STR1.type;
type2 = STR2.type;
%%% please consider the string '[i,j]' means the i-th predicate is imposed at time j an is valid. or in another word b_{phi_i}(s_j)>0. take a look at manual. 
%%%%%%%%% stacks the cell of char arrays and applies '|' between the two.

t1_or   = strcmp(type1 , 'or'  );  
t1_and  = strcmp(type1 , 'and' ) || strcmp(type1 , 'leaf');

t2_or   = strcmp(type2 , 'or'  );  
t2_and  = strcmp(type2 , 'and' ) || strcmp(type2 , 'leaf');



[size11,size21] = size(string1);
[size12,size22] = size(string2);
size1 = size11+size12;
size2 = max(size21,size22)+2;
str = cell(size1,size2);
str(:) = {' '};




if     t1_or && t2_or  %%% don't pack the first and  don't pack the second
    
    if size21>=size22
        
        str(1:size11, 2:size2-1) = string1;
        
        str(size11, size2   ) =  {'|'};
        shift =  0.5*(size2-size22);
        
        str(size11+1:end, shift+1:size2-shift) = string2;     
        
    else
         
        str(1:size12, 2:size2-1) = string2;
         
        str(size12, size2   ) =  {'|'};
        shift =  0.5*(size2-size21);
         
        str(size12+1:end, shift+1:size2-shift) = string1;
         
    end
    



elseif t1_or && t2_and  %%% don't pack the first and  pack the second
    
     if size21>=size22
         
        str(1:size11, 2:size2-1) = string1;
         
        str(size11, size2   ) =  {'|'};
        shift =  0.5*(size2-size22);
        str(size11+1,shift) = {'('};
        str(size11+1:end, shift+1:size2-shift) = string2;
        str(size1, size2-shift ) =  {')'};
    else
        str(1,1) = {'('};
        str(1:size12, 2:size2-1) = string2;
        str(size12, size2-1 ) =  {')'};
        str(size12, size2   ) =  {'|'};
        shift =  0.5*(size2-size21);
         
        str(size12+1:end, shift+1:size2-shift) = string1;
        
    end



    
elseif t1_and && t2_or  %%% pack the first and  don't pack the second
    
    
     if size21>=size22
        str(1,1) = {'('};
        str(1:size11, 2:size2-1) = string1;
        str(size11, size2-1 ) =  {')'};
        str(size11, size2   ) =  {'|'};
        shift =  0.5*(size2-size22);
         
        str(size11+1:end, shift+1:size2-shift) = string2;
         
     else
         
        str(1:size12, 2:size2-1) = string2;
         
        str(size12, size2   ) =  {'|'};
        shift =  0.5*(size2-size21);
        str(size12+1,shift) = {'('};
        str(size12+1:end, shift+1:size2-shift) = string1;
        str(size1, size2-shift ) =  {')'};
    end
    
    
    
    
elseif t1_and && t2_and  %%% pack the first and  pack the second
    
    
    if size21>=size22
        str(1,1) = {'('};
        str(1:size11, 2:size2-1) = string1;
        str(size11, size2-1 ) =  {')'};
        str(size11, size2   ) =  {'|'};
        shift =  0.5*(size2-size22);
        str(size11+1,shift) = {'('};
        str(size11+1:end, shift+1:size2-shift) = string2;
        str(size1, size2-shift ) =  {')'};
    else
        str(1,1) = {'('};
        str(1:size12, 2:size2-1) = string2;
        str(size12, size2-1 ) =  {')'};
        str(size12, size2   ) =  {'|'};
        shift =  0.5*(size2-size21);
        str(size12+1,shift) = {'('};
        str(size12+1:end, shift+1:size2-shift) = string1;
        str(size1, size2-shift ) =  {')'};
    end
    



end

STR.cntn = str;
STR.type = 'or';

end