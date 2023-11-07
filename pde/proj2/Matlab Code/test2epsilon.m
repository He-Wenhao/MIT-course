ReducedBaseOffline;
Nplot = 30
Bilst = linspace(0.01, 10, Nplot);
Costlst = zeros(1, Nplot);
for i = 1:Nplot
    Bi = Bilst(i);
    mu = [0.4, 0.6, 0.8, 1.2, Bi];
    [uN, TrootN] = ReducedBaseOnline(mu,N,ANq,FN);
    cost = 0.2*Bi+TrootN;
    Costlst(i) = cost;
end
Costlst

% Plot the function
plot(Bilst, Costlst);

% Add labels and title for clarity
xlabel('Bi');
ylabel('Cost');
title('Cost function');

% fine: 7.0856
% medium: 7.0724
% coarse: 7.0242
% log2(Tc-Tf/Tm-Tf) = 2.2177
% expect is 2