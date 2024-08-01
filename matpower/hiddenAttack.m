clear all;

load("yangData1stRound.mat", 'BfO')

%H matrix must be a singular matrix
Bf= full(BfO);
Bf(1, 1)= 16;
rankBf= rank(Bf);
[s, v, d]= svd(Bf);
x= d(:, end)*100;
ansT= Bf*(x);

sum(x)
sum(ansT)