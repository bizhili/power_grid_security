clear all;

load("yangData1stRound.mat", 'BfO')

busNum = 14;
branchNum= 20;
T= 1000;
%H matrix must be a singular matrix
Bf= full(BfO);
Bf(1, 1)= 16.4;
rankBf= rank(Bf);
[s, v, d]= svd(Bf);
svdT= svd(Bf);
x= d(:, end);
ansT= Bf*(x);
baseAngle= 0.1*rand([busNum, 1])+0.2;
powerFLs= zeros([T, branchNum]);
angles= zeros([T, busNum]);
labels= zeros([1, busNum]);
for i =1: T
    addAngle= 0.01*rand([busNum, 1]);
    sumAngle= baseAngle*(1+(0.2)*cos(i/(100+randn*2)*pi))+addAngle;
    sumAngle(1)= 0;
    if i>300 && i<500
        attack= -2*x*sin((i-300)/200*pi);
        attack(1)= attack(1)-0.005;
        sumAngle= attack+sumAngle;
        labels(i)= 1;
    end
    powerFL= Bf*sumAngle;
    powerFLs(i, :)= powerFL/norm(powerFL);
    angles(i, :)= sumAngle;
end
covT= eye(20);
estAngles= lscov(Bf, powerFLs.', covT);
scaleEstAngles= sqrt(sum(estAngles.*estAngles, 1));
figure(1);
plot(scaleEstAngles);
figure(2);
plot(powerFLs);
corT= corrcoef(estAngles);
figure(3);
imagesc(corT);
