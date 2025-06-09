clear;
define_constants;
%verbose= 0;
% 0: print none, 1: print little
opt = mpoption('VERBOSE',0, 'OUT_ALL',0);
opt.model= "AC";
%case14, case30, case118, 
caseName= "case14";
mpc = loadcase(caseName);
busNum= length(mpc.bus);
branchNum= length(mpc.branch);
timeSteps= 30;
baseActiveLoad= mpc.bus(:, PD);
needLoad= (baseActiveLoad~=0);
sum(needLoad)
loadStd= 10;
measureBranchReactivePower= zeros(timeSteps, branchNum);
measureBranchActivePower= zeros(timeSteps, branchNum);
measureBusActivePower= zeros(timeSteps, busNum);
stateBusVoltageAngle= zeros(timeSteps, busNum);
for i = 1:timeSteps
    i
    activeLoadDiffer= 2*loadStd*(rand([busNum, 1])-0.5).*needLoad;
    setActiveLoad= abs(baseActiveLoad+activeLoadDiffer);
    mpc.bus(:, PD)= setActiveLoad;
    results= runopf(mpc, opt);
    %print non succeed optimal power flow
    if results.success==0
        rst= "error"
    end
    measureBranchActivePower(i, :)= results.branch(:, PF);
    measureBranchReactivePower(i, :)= results.branch(:, QF);
    measureBusActivePower(i, :)= results.bus(:, PD);
    stateBusVoltageAngle(i, :)= results.bus(:, VA);
end
plot(stateBusVoltageAngle);
[Bbus, BfO, Pbusinj, Pfinj] = makeBdc(mpc);%Bf*\theta+Pfinj
BfO= full(BfO);
Bf= BfO;
S = svd(Bf);
sumBias= 0;
for i = 1:timeSteps % verify 
    tempAngle= stateBusVoltageAngle(i,:).';
    tempbranchActive= Bf*tempAngle;% why *1e2 ?
    tempBias=measureBranchActivePower(i, :)-tempbranchActive.';
    sumBias=sumBias+sum(tempBias);
end
sumBias

ratio= mpc.branch(:, 9);
ratio(ratio<1e-6)= 1;
Bf= Bf.*ratio;

save('test_data/'+opt.model+'_'+caseName+'_.mat','measureBranchActivePower', 'measureBusActivePower', ...
    'stateBusVoltageAngle', 'Bf', 'measureBranchReactivePower')

buses = cell(busNum,1);  % Preallocate a cell array to hold the strings
for i = 1:busNum
    buses{i} = ['bus', num2str(i)];  % Concatenate 'branch' with the number
end

