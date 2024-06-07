function s_next  = model(input)

s = input(1:3,1);
a = input(4:5,1);


dt =0.05;
L =1;

a(1,1)=0.5*a(1,1);
a(2,1)=0.5*a(2,1);

v   = 2.5*tanh(a(1,1)) + 2.5;
gam = (pi/4)*tanh(a(2,1))   ;

f1 =  s(1,1)  +  (  L/tan(gam)  )  *   (    sin(  s(3,1) + (v/L)*tan(gam)*dt  )  -    sin(  s(3,1)  )   );
f2 =  s(2,1)  +  (  L/tan(gam)  )  *   (   -cos(  s(3,1) + (v/L)*tan(gam)*dt  )  +    cos(  s(3,1)  )   );
f3 =  s(3,1)  +  (v/L)*tan(gam)*dt  ;

s_next = [ f1; f2; f3 ];


end