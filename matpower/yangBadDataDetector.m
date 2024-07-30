clear all;

load("yangData.mat", 'noisedbranchActive', 'stateBusVoltageAngle')
noisedbranchActive2nd= noisedbranchActive;
stateBusVoltageAngle2nd= stateBusVoltageAngle;
load("yangData1stRoundBase.mat", 'noisedbranchActive', 'stateBusVoltageAngle')
noisedbranchActiveBase= noisedbranchActive;
stateBusVoltageAngleBase= stateBusVoltageAngle;
load("yangData1stRound.mat", 'noisedbranchActive', 'stateBusVoltageAngle', 'BfO')

%H matrix must be a singular matrix

Bf= BfO;
rankBf= rank(full(Bf));

std= 0.01;

cov= std^2*eye(length(noisedbranchActive(1, :)));
stdMea= sqrt(diag(cov));%
estimateBusVoltageAngle= stateEstimator(noisedbranchActive, Bf, cov);
estimateBusVoltageAngle= estimateBusVoltageAngle- estimateBusVoltageAngle(:, 1);
%chiSquare test
residuals= (Bf*estimateBusVoltageAngle.').'-noisedbranchActive;
stdResidual= residuals./stdMea.';
chiSquare= sum((stdResidual.^2).');% degree of flexiable: 19;
testPass= chiSquare<25;%chisquare table: 25

100-sum(testPass)

stateBusVoltageAngleModify= stateBusVoltageAngle-stateBusVoltageAngle(:, 1);
rest= stateBusVoltageAngleModify-estimateBusVoltageAngle;
restMean= mean(rest);
error= (estimateBusVoltageAngle+restMean)-stateBusVoltageAngleModify(:, 1:end);
plot(noisedbranchActiveBase, 'DisplayName',"State estimate without reference bus", color="black");

hold on;
plot(noisedbranchActive, 'DisplayName', "Yang's estimation", color="blue");
xlabel("Time point")
ylabel("voltage angle")
