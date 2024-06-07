function CP_verifier(net, STL2NN, m)

Net.weights = net.weights;
Net.biases = net.biases;
dim = net.dims;
n = dim(1)-1;
for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {'tanh'};
    Net.layers{i} = L ;
end
net = Net;

T = 40;
thet_lb = -3*pi/4;
thet_ub = -4*pi/8;


rho = zeros(1, m);

parfor i=1:m
    i
    s0 =  [ 6 ;  8;  thet_lb+rand*(thet_ub-thet_lb)  ];
    a0 = NN( net , [s0;0] , []);
    xx = zeros(3,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(2,T);
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

end