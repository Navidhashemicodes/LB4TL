clear all
clc
close all 

T=35;
dim = [7,10,3];

s0 = [  0.035 ;  0.025 ;  0;  0;  0;  0];
dd=true;

C = eye(6); 
D=  zeros(6,1);

len = length(dim);



% [net.weights , net.biases, net.layers] = param2net(Param, [7,10,3], 'tanh');



while dd
    for i=1:len-1
        net.weights{i} = 1*(2*rand(dim(i+1), dim(i))-1);
        net.biases{i} =  1*( 2*rand(dim(i+1),1)-1);
        if i<=len-2
            L = cell(dim(i+1),1);
            L(:) = {'tanh'};
            net.layers{i} = L;
        end
    end
    
    a{1} = NN(net, [C*s0+D;0], []);
    s{1} = model([s0;a{1}]);
    for i=2:T
        a{i} = NN(net, [C*s{i-1}+D;i-1], []);
        s{i} = model([s{i-1};a{i}]);
    end
    
    dd=false;
%     S1 = s{end}(1,1)-s0(1,1)
%     S2 = s{end}(2,1)-s0(2,1)
%     if abs(S1+2)<0.5 && abs(S2+8)<0.5
%         dd=false;
%     end
end

ss=[s{:}];



figure(1)

%%%%  U 
center = [0.06  0.275  0.6];
scale  = [0.22  0.15   1.2];  
plotcube(center , scale, 'red', 0.3)

hold on

%%%%  G 
center = [0.075 0.54   0.6];
scale  = [0.05  0.08   0.2];  
plotcube(center , scale, 'blue', 0.3)

hold on


%%%%  start 
center = [0.035 0.025  0   ];
scale  = [0.03  0.05   0.01];  
plotcube(center , scale, 'green', 0.3)

hold on

%%%%%%%%%%%%%%

axis equal


color = '-g.' ;
plot3([s0(1,1) ss(1,:)],[s0(2,1) ss(2,:)],[s0(3,1) ss(3,:)], color, 'Linewidth', 0.75);
hold on


Param=net2param(net, dim);

clearvars -except Param

save('init10_badinit', 'Param')