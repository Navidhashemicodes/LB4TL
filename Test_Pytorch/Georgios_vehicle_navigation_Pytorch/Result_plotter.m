function Result_plotter(net)
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




figure(3)

%%%%  U 

y = 2:0.01:5;
x= 1*ones(size(y));
plot(x,y,'green')
hold on
x= 4*ones(size(y));
plot(x,y,'green')
hold on
x = 1:0.01:4;
y= 2*ones(size(x));
plot(x,y,'green')
hold on
y= 5*ones(size(x));
plot(x,y,'green')
hold on

%%%%%%%%%%%%%%

%%%%  G_1 

y = 3:0.01:4;
x= 5*ones(size(y));
plot(x,y,'blue')
hold on
x= 6*ones(size(y));
plot(x,y,'blue')
hold on
x = 5:0.01:6;
y= 3*ones(size(x));
plot(x,y,'blue')
hold on
y= 4*ones(size(x));
plot(x,y,'blue')
hold on

%%%%%%%%%%%%%%


%%%%  G_2 

y = 0:0.01:1;
x= 3*ones(size(y));
plot(x,y,'red')
hold on
x= 4*ones(size(y));
plot(x,y,'red')
hold on
x = 3:0.01:4;
y= 0*ones(size(x));
plot(x,y,'red')
hold on
y= 1*ones(size(x));
plot(x,y,'red')
hold on

%%%%%%%%%%%%%%
T = 40;
axis equal
s0s = [  [ 6 ; 8 ;  -3*pi/4 ]  ,  [ 6 ; 8 ;  -5*pi/8    ]  , [ 6 ; 8 ;  -4*pi/8 ]  ];
for i=1:3
    s0 =  s0s(:,i);
    a0 = NN( net , [s0;0] , []);
    xx = zeros(3,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(2,T);
    for ij=1:T-1
        a(:,ij)=NN(net, [xx(:,ij);ij], []);
        xx(:,ij+1)=model([xx(:,ij);a(:,ij)]);
    end

    color = '-g.' ;
    plot([s0(1,1) xx(1,:)],[s0(2,1) xx(2,:)], color, 'Linewidth', 0.75);
    hold on
end

load('init_weights')
Net.weights = W;
load('init_biases')
Net.biases = b;
Net.layers = net.layers;


for i=1:3
    s0 =  s0s(:,i);
    a0 = NN( Net , [s0;0] , []);
    xx = zeros(3,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(2,T);
    for ij=1:T-1
        a(:,ij)=NN(Net, [xx(:,ij);ij], []);
        xx(:,ij+1)=model([xx(:,ij);a(:,ij)]);
    end

    color = '-r.' ;
    plot([s0(1,1) xx(1,:)],[s0(2,1) xx(2,:)], color, 'Linewidth', 0.75);
    hold on
end

end
