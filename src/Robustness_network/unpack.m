function  str2 = unpack(str)

i=0;
update =true;
len = length(str);
for j=1:len
    if strcmp(str(j), '(')
        i=i+1;
    elseif strcmp(str(j), ')')
        i=i-1;
    end
    if j<len
        if i==0
            str2= str;
            update = false;
            break
        end
    end
end

if update
    if ~isempty(str)
        if i==0
            str2 = unpack(str(2:end-1));
        else
            error('something is wrong')
        end
    else
        str2 = '';
    end
end


end