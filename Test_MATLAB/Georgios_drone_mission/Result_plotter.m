function Result_plotter(net, Net)

n=6;

figure

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


s0s = [  0.02  0.02   0.05   0.05   0.035 ;...
         0     0.05   0      0.05   0.025 ;...
                       zeros(4,5)         ]; 

T = 35;
m=3;

for i=1:5
    s0 =  s0s(:,i);
    a0 = NN( net , [s0;0] , []);
    xx = zeros(n,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(m,T);
    for ij=1:T-1
        a(:,ij)=NN(net, [xx(:,ij);ij], []);
        xx(:,ij+1)=model([xx(:,ij);a(:,ij)]);
    end

    color = '-g.' ;
    plot3([s0(1,1) xx(1,:)],[s0(2,1) xx(2,:)],[s0(3,1) xx(3,:)], color, 'Linewidth', 0.75);
    hold on
end


for i=1:5
    s0 =  s0s(:,i);
    a0 = NN( Net , [s0;0] , []);
    xx = zeros(n,T);
    xx(:,1)=model( [ s0;a0]  );
    a =zeros(m,T);
    for ij=1:T-1
        a(:,ij)=NN(Net, [xx(:,ij);ij], []);
        xx(:,ij+1)=model([xx(:,ij);a(:,ij)]);
    end

    color = '-r.' ;
    plot3([s0(1,1) xx(1,:)],[s0(2,1) xx(2,:)],[s0(3,1) xx(3,:)], color, 'Linewidth', 0.75);
    hold on
end


end
