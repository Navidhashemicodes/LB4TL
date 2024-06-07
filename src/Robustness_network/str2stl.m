function  STL  =  str2stl(str)

%%% the STL specification are not characters in this toolbox. They are
%%% cell. Therefore in this code we get the STL specification as char and
%%% translate them in the form of cell. The cell should be of the
%%% introduced type. This is how this toolbox works.

index1 = find(str=='[');
index2 = find(str==',');
index3 = find(str==']');
stl = cell(1,6);
stl(:) = {''};
stl(1,1) = {'['};
stl(1,2) = {str2num(str(index1+1:index2-1))};
stl(1,3) = {','};
stl(1,4) = {str2num(str(index2+1:index3-1))};
stl(1,5) = {']'};


STL.cntn = stl;
STL.type = 'leaf';

end

    
    