clear;
define_constants;
%verbose= 0;
% 0: print none, 1: print little
opt = mpoption(  'pf.alg', 1, ...          % 1 = Newton-Raphson
                 'pf.tol', 1e-10, ...      % tighter tolerance
                 'pf.nr.max_it', 50, ...
                 'pf.enforce_q_lims', 1);
opt.model= "AC";
%case14, case30, case118, 
caseName= "case30";
mpc = loadcase(caseName);
busNum= length(mpc.bus);
branchNum= length(mpc.branch);
timeSteps= 500;
baseActiveLoad= mpc.bus(:, PD);
needLoad= (baseActiveLoad~=0);
sum(needLoad)
loadStd= 1;
measureBranchReactivePower= zeros(timeSteps, branchNum);
measureBranchActivePower= zeros(timeSteps, branchNum);
measureBusActivePower= zeros(timeSteps, busNum);
measureBusReactivePower= zeros(timeSteps, busNum);
stateBusVoltageAngle= zeros(timeSteps, busNum);
stateBusVoltageMagni= zeros(timeSteps, busNum);

for i = 1:timeSteps
    
%     activeLoadDiffer= 2*loadStd*(rand([busNum, 1])-0.5).*needLoad;
%     setActiveLoad= abs(baseActiveLoad+activeLoadDiffer);
%     mpc.bus(:, PD)= setActiveLoad;
    mpc_t = randomize_load_once(mpc, 'alpha',[0.7 1.15], 'pf_min',0.92, 'pf_max',0.99);
    results= runopf(mpc_t, opt);
    %print non succeed optimal power flow
    if results.success==0
        rst= "error"
    end
    measureBranchActivePower(i, :)= results.branch(:, PF);
    measureBranchReactivePower(i, :)= results.branch(:, QF);
    measureBusActivePower(i, :)= results.bus(:, PD);
    stateBusVoltageAngle(i, :)= results.bus(:, VA);
    measureBusReactivePower(i, :)= results.bus(:, QD);
    stateBusVoltageMagni(i, :)= results.bus(:, VM);
end
plot(measureBranchReactivePower);
[Bbus, BfO, Pbusinj, Pfinj] = makeBdc(mpc);%Bf*\theta+Pfinj
BfO= full(BfO);
Bf= BfO;
S = svd(Bf);
resistence = mpc.branch(:,3);
reactance = mpc.branch(:,4);  %
% sumBias= 0;
% for i = 1:timeSteps % verify 
%     tempAngle= stateBusVoltageAngle(i,:).';
%     tempbranchActive= Bf*tempAngle;% why *1e2 ?
%     tempBias=measureBranchActivePower(i, :)-tempbranchActive.';
%     sumBias=sumBias+sum(tempBias);
% end
% sumBias

ratio= mpc.branch(:, 9);
ratio(ratio<1e-6)= 1;
Bf= Bf.*ratio;

save('test_data/'+opt.model+'_'+caseName+'_.mat','measureBranchActivePower', 'measureBusActivePower', ...
    'stateBusVoltageAngle', 'Bf', 'measureBranchReactivePower', 'measureBusReactivePower', 'stateBusVoltageMagni', ...
    'resistence', 'reactance');

buses = cell(busNum,1);  % Preallocate a cell array to hold the strings
for i = 1:busNum
    buses{i} = ['bus', num2str(i)];  % Concatenate 'branch' with the number
end

