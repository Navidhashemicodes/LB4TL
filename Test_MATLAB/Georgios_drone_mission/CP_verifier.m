function CP_verifier(net, STL2NN, m)
n=6;
T = 35;
lb = [0.02;    zeros(5,1)    ];
ub = [0.05; 0.05 ; zeros(4,1)];

rho = zeros(1, m);

parfor i=1:m
    i
    s0 =  lb +rand*(ub-lb);
    a0 = NN( net , [s0;0] , []);
    xx = zeros(6,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(3,T);
    for ij=1:T-1
        a(:,ij)=NN(net, [xx(:,ij);ij], []);
        xx(:,ij+1)=model([xx(:,ij);a(:,ij)]);
    end

    traj = reshape( [s0 xx]  , [n*(T+1) , 1]   );
    rho(1,i) = NN(STL2NN , traj, []);
end


ell = floor(0.9999*(m+1))+1;

Rs = sort(-rho);

R_star = Rs(ell);

p1 = 0.9995;
p2 = 1 - betainc(p1 , ell , m+1-ell);

if R_star < 0 

    disp( [ ' Pr[    Pr[   controller is approved    ]  >  ' num2str(p1*100) '%   ]  >  ' num2str(p2*100) '%  ' ] );
else
    disp(' It seems like the controller does not satisfy Pr[    Pr[   controller is approved    ]  >  99.98   ]  >  99.5');

end