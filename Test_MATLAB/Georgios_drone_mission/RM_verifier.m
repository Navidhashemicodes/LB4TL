function RM_verifier(net, STL2NN, m)
n=6;
T = 35;
lb = [0.02;    zeros(5,1)    ];
ub = [0.05; 0.05 ; zeros(4,1)];

rho = zeros(1, m);

parfor i=1:m
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
Rs = sort(-rho);
R_star = Rs(end);

if R_star<0
    disp(['The maximum found residual is negative and is equal to: ' num2str(R_star) ])
else
    disp(['The maximum found residual is positive and is equal to: ' num2str(R_star) ])
end

end