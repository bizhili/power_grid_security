clear;
define_constants;
%verbose= 0;
% 0: print none, 1: print little
opt = mpoption('VERBOSE',0, 'OUT_ALL',0);
opt.model= "DC";
mpc = loadcase('case14');
busNum= length(mpc.bus);
branchNum= length(mpc.branch);
timeSteps= 100;
baseActiveLoad= mpc.bus(:, PD);
needLoad= (baseActiveLoad~=0);
loadStd= 2;

measureBranchActivePower= zeros(timeSteps, branchNum);
measureBusActivePower= zeros(timeSteps, busNum-1);
stateBusVoltageAngle= zeros(timeSteps, busNum-1);
for i = 1:timeSteps
    i
    activeLoadDiffer= 2*loadStd*(rand([busNum, 1])-0.5).*needLoad;
    setActiveLoad= baseActiveLoad+activeLoadDiffer;
    mpc.bus(:, PD)= setActiveLoad;
    results= runopf(mpc, opt);
    %print non succeed optimal power flow
    if results.success==0
        print(i)
    end
    measureBranchActivePower(i, :)= results.branch(:, PF);
    measureBusActivePower(i, :)= results.bus(2:end, PD);
    stateBusVoltageAngle(i, :)= results.bus(2:end, VA);
end
plot(measureBranchActivePower);
[Bbus, Bf, Pbusinj, Pfinj] = makeBdc(mpc);%Bf*\theta+Pfinj
Bf= full(Bf)*pi/180*1e2;
Bf= Bf(:, 2: end);
sumBias= 0;
for i = 1:timeSteps
    tempAngle= stateBusVoltageAngle(i,:).';
    tempbranchActive= Bf*tempAngle;% why *1e2 ?
    tempBias=measureBranchActivePower(i, :)-tempbranchActive.';
    sumBias=sumBias+sum(tempBias);
end
sumBias

save('data.mat','measureBranchActivePower', 'measureBusActivePower', ...
    'stateBusVoltageAngle', 'Bf')

buses = cell(13,1);  % Preallocate a cell array to hold the strings
for i = 1:13
    buses{i} = ['bus', num2str(i)];  % Concatenate 'branch' with the number
end

