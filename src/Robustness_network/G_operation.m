function STR = G_operation( STR,  init, finish)

%%% The diffierence between temporal operator 'G' and nontemporal operator '&' is we also copy the string and increase the time.
%%% for example G_operation( '[1,0]', 1,3) means  '[1,1]&[1,2]&[1,3]'
%%% please consider the string '[i,j]' means the i-th predicate is imposed at time j an is valid. or in another word b_{phi_i}(s_j)>0. take a look at manual. 
string = STR.cntn;
type = STR.type;



number = finish-init+1;
[size1,size2] = size(string);
str = cell(size1*number,size2+2);
str(:) = {''};
time_index =  floor(0.5*(size2+2+2));



if strcmp(type, 'or') || strcmp(type, 'leaf')  %%% pack when repeat
    
    for i = init:finish
        str1 = cell(size1,size2+2);
        str1(:) = {''};
        str1(:, 2:size2+1) = string;
        index = (i-init)*size1;
        str1(1,1) = {'('};
        str1(size1,size2+2-1) = {')'};
        str1(size1,size2+2  ) = {'&'};
        for j = 1:size1
            str1(j,time_index) = {cell2mat(str1(j,time_index))+i};
        end
        str(index+1:index+size1 , : ) = str1;
    end
    
else   %%% don't pack when repeat
    
    for i = init:finish
        str1 = cell(size1,size2+2);
        str1(:) = {''};
        str1(:, 2:size2+1) = string;
        index = (i-init)*size1;
        
        
        str1(size1,size2+2  ) = {'&'};
        for j = 1:size1
            str1(j,time_index) = {cell2mat(str1(j,time_index))+i};
        end
        str(index+1:index+size1 , : ) = str1;
    end
    
end

str(end,end) = {''};

STR.cntn = str;
STR.type = 'and';


end
