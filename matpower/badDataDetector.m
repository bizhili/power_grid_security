clear;

load("data.mat", 'measureBranchActivePower', 'measureBusActivePower', ...
    'stateBusVoltageAngle', 'Bf')

%H matrix must be a singular matrix

rankBf= rank(full(Bf));

std= 0.1;

noisedbranchActive= measurementNoise(measureBranchActivePower, std);

attackedBranchActive= stealtyAttack(noisedbranchActive, Bf, 0.2);
%attackedBranchActive= randomAttack(noisedbranchActive, std*7);
cov= std^2*eye(length(measureBranchActivePower(1, :)));
stdMea= sqrt(diag(cov));%
estimateBusVoltageAngle= stateEstimator(attackedBranchActive, Bf, cov);

%chiSquare test
residuals= (Bf*estimateBusVoltageAngle.').'-attackedBranchActive;
stdResidual= residuals./stdMea.';
chiSquare= sum((stdResidual.^2).');% degree of flexiable: 19;
testPass= chiSquare<25;%chisquare table: 25

100-sum(testPass)

xs= [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5 , 6, 6.5, 7];
ys= [0, 12, 38, 65, 88, 92, 96, 99, 98, 99, 100, 100, 100, 100];