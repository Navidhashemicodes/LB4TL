function cell_ordered = ordered_operation(cell_1, cell_2, start, End )

if End > start
    Cell = and_operation(  F_operation(   cell_1 , start,start  )  , F_operation(   cell_2  ,  start+1, End )   );
    for i = start+1 : End-1
        Cell = or_operation(   Cell  , and_operation(   F_operation(   cell_1 , i , i ) , F_operation(   cell_2 , i+1 , End )  )   );
    end
    cell_ordered = Cell;
else
    error('Lower time bound should be at least one step lower than upper bound time. ')
end
end
