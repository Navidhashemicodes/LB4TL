function STR = Until_operation(STR1, STR2, start, End)


%%%% Until finction like every other function can be written based on 'and' and 'or' operations. This function shows how is this relation
%%%% phi1 U_[a,b] phi2    \exists t2\in[a,b] phi2(t2)==true & G_t\in[0,t2] phi1(t)==true
if start>0
    STR_until = and_operation(  F_operation(   STR2 , start,start  )  , G_operation(   STR1  ,  0,start-1)   );
    for i = start+1:End
        STR_until = or_operation(   STR_until  , and_operation(   F_operation(   STR2, i,i   )  ,  G_operation(   STR1, 0,i-1 )  )  ) ;
    end
    STR = STR_until;
else
    error('Lower time bound for Until operation can not be zero. ')
end
end
