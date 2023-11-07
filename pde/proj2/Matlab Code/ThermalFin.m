function [u, Troot] = ThermalFin(mesh, mu)
%
% -------------------------------------------------------------------------
%   Computes the temperature distribution and root temperature for a fin
%   using the Finite Element Method.
% -------------------------------------------------------------------------
%
%   INPUT   mesh    grid lebel (coarse, medium, or fine)
%           mu      thermal conductivities of sections and Biot number 1x5
%
%   OUTPUT  u       temperature disctribution in the fin
%           Troot   root temperature
%
% -------------------------------------------------------------------------


% Parameters setup: rearranges the values in mu
kappa = ones(6,1);
kappa(1:4) = mu(1:4);    % Bi = mu(6)
kappa(6) = mu(5);


% Initialization
A = sparse(mesh.nodes, mesh.nodes);
F = sparse(mesh.nodes,1);


% Domain Interior
for i = 1:5         % interior regions
    for n = 1:length(mesh.theta{i})
        % Get mesh node index phi
        phi = mesh.theta{i}(n,:)';
        % Get the node coordinates for the current element
        vert = mesh.coor(phi, :);
        % Calculate the area of the triangle
        area = polyarea(vert(:,1), vert(:,2));
        % Shape functions derivatives
        G = [vert(2,2)-vert(3,2), vert(3,2)-vert(1,2), vert(1,2)-vert(2,2);
         vert(3,1)-vert(2,1), vert(1,1)-vert(3,1), vert(2,1)-vert(1,1)] / (2*area);    
        % Compute the element stiffness matrix
        Alocal = area * (G' * G);
        % update
        A(phi,phi) = A(phi,phi) + mu(i)*Alocal;    
    end
end


% Boundaries (not root)
i = 6;
    for n = 1:length(mesh.theta{i})
        phi = mesh.theta{i}(n,:)';
        % Get the node coordinates for the current element
        vert = mesh.coor(phi, :);
        % Distance
        dist = sqrt(sum((vert(1,:) - vert(2,:)).^2));
        Alocal = [1/3,1/6;1/6,1/3]*dist;
        % update
        A(phi,phi) = A(phi,phi) + mu(5)*Alocal;
    end


% Root Boundary
i = 7;
    for n = 1:length(mesh.theta{i})
        phi = mesh.theta{i}(n,:)';
        % Get the node coordinates for the current element
        vert = mesh.coor(phi, :);
        % Distance
        dist = sqrt(sum((vert(1,:) - vert(2,:)).^2));
        Flocal = [1/2;1/2]*dist;
        % update
        F(phi) = F(phi) + Flocal;
    end


% Solve for the temperature distribution
u = full(A\F);      % use full since A and F are sparse


% Compute the temperature at the root
Troot = F'*u;
    