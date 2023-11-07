ReducedBaseOffline
mu = [0.4, 0.6, 0.8, 1.2, 0.1];
[uN, TrootN] = ReducedBaseOnline(mu,N,ANq,FN);
TrootN

% fine: 7.0856
% medium: 7.0724
% coarse: 7.0242
% log2(Tc-Tf/Tm-Tf) = 2.2177
% expect is 2