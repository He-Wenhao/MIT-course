function [uN, TrootN] = ReducedBaseOnline(mu,N,ANq,FN)
%
% -------------------------------------------------------------------------
%
%   ReducedBasisOnline.m
%
% -------------------------------------------------------------------------
%   Reduced basis on-line: computes the temperature distribution and root
%   temperature using the reduced basis computed by the off-line part.
%
%   The reduced basis data ReducedBasis.mat should have been previsously
%   loaded using the comand "load ReducedBasis".
% -------------------------------------------------------------------------
%
%   INPUT   mu      thermal conductivities of sections and Biot number 1x5
%           N       dimension of the reduced basis
%           ANq     reduced matrix
%           FN      reduced source
%
%   OUTPUT  uN      temperature disctribution in the fin from reduced basis
%           TrootN  root temperature from reduced basis
%
% -------------------------------------------------------------------------


% Parameters - (domain 5 is the root)
sigma = ones(6,1);
sigma(1:5) = mu(1:5);    % Bi = mu(6)
sigma(6) = mu(5);

% Construct of A_N
AN = sparse(N, N);
for i=1:6
    AN = AN + sigma(i)*ANq{i};
end

% Solve the system
uN = AN\FN;

% Compute the temperature at the root
TrootN = FN'*uN;