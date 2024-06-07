function plotcube(center , scale, color, transparency)
xc=center(1); yc=center(2); zc=center(3);    % coordinated of the center
                % cube size (length of an edge)
alpha=transparency;           % transparency (max=1=opaque)

X = [0 0 0 0 0 1; 1 0 1 1 1 1; 1 0 1 1 1 1; 0 0 0 0 0 1];
Y = [0 0 0 0 1 0; 0 1 0 0 1 1; 0 1 1 1 1 1; 0 0 1 1 1 0];
Z = [0 0 1 0 0 0; 0 0 1 0 0 0; 1 1 1 0 1 1; 1 1 1 0 1 1];

C=color;                  % unicolor

X = scale(1)*(X-0.5) + xc;
Y = scale(2)*(Y-0.5) + yc;
Z = scale(3)*(Z-0.5) + zc;

fill3(X,Y,Z,C,'FaceAlpha',alpha);    % draw cube
axis equal
end