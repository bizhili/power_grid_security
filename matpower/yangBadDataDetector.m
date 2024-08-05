clear all;

load("yangData.mat", 'noisedbranchActive', 'stateBusVoltageAngle')
noisedbranchActive2nd= noisedbranchActive;
stateBusVoltageAngle2nd= stateBusVoltageAngle;
load("yangData1stRoundBase.mat", 'noisedbranchActive', 'stateBusVoltageAngle')
noisedbranchActiveBase= noisedbranchActive;
stateBusVoltageAngleBase= stateBusVoltageAngle;
load("yangData1stRound.mat", 'noisedbranchActive', 'stateBusVoltageAngle', 'BfO')

%H matrix must be a singular matrix
Bf= full(BfO);
rankBf= rank(Bf);

std= 0.01;

covT= std^2*eye(length(noisedbranchActive(1, :)));
stdMea= sqrt(diag(covT));%
estimateBusVoltageAngle= stateEstimator(noisedbranchActive, Bf, covT);
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
figure(1);
% noisedbranchActive= noisedbranchActive2nd;
noisedbranchActiveNorm= sqrt(sum(noisedbranchActive.*noisedbranchActive, 2));

% noisedbranchActive= noisedbranchActive./noisedbranchActiveNorm;
noisedbranchActiveNormMean= mean(noisedbranchActive, 1);
% noisedbranchActive= noisedbranchActive-noisedbranchActiveNormMean;

plot(stateBusVoltageAngle, 'DisplayName',"State estimate without reference bus", color="black");
% error2= (noisedbranchActive-noisedbranchActiveBase);
noisedbranchActiveMean= movmean(noisedbranchActive, 500);
figure(2);
plot(noisedbranchActive);
hold on;
plot(noisedbranchActiveBase);
figure(3);
corT= corrcoef(noisedbranchActive.');
imagesc(corT);
% corT2= corrcoef(error2.');
hold on;
%plot(error2, 'DisplayName', "Yang's estimation", color="blue");
xlabel("Time point")
ylabel("voltage angle")
