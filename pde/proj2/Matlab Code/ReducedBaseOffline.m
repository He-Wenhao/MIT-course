% -------------------------------------------------------------------------
%
%   ReducedBasisOffline.m
%
% -------------------------------------------------------------------------
%   Reduced basis off-line: computes the reduced basis matrices and source 
%   vector from a sample.
% -------------------------------------------------------------------------


% Load grid triangulations
load grids;

% Mesh level to be used
mesh = coarse;

% Load sample matrix and get the number of samples
load sn.dat;
[N,len2] = size(sn);


% Solve the sample cases and construct Z
Z = zeros(mesh.nodes,N);
for i = 1:N
    [z, Troot] = ThermalFin(mesh, sn(i,:));
    Z(:,i) = z';
end

% Initialization of reduced-base matrix and vector
ANq = cell(6,1);
FN = zeros(mesh.nodes,1);

% Calculate Anq and FN

% Domain Interior
for i = 1:5         % interior regions
    A = sparse(mesh.nodes, mesh.nodes);
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
        A(phi,phi) = A(phi,phi) + Alocal;    
    end
    ANq{i} = Z'*A*Z;
end


% Boundaries (not root)
i = 6;
A = sparse(mesh.nodes, mesh.nodes);
    for n = 1:length(mesh.theta{i})
        phi = mesh.theta{i}(n,:)';
        % Get the node coordinates for the current element
        vert = mesh.coor(phi, :);
        % Distance
        dist = sqrt(sum((vert(1,:) - vert(2,:)).^2));
        Alocal = [1/3,1/6;1/6,1/3]*dist;
        % update
        A(phi,phi) = A(phi,phi) + Alocal;
    end
ANq{i} = Z'*A*Z;

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
        FN(phi) = FN(phi) + Flocal;
    end
FN = Z'*FN;

% Save the reduced matrix and vector
save ReducedBasis.mat ANq FN N