clear;
define_constants;
%verbose= 0;
% 0: print none, 1: print little
opt = mpoption('VERBOSE',0, 'OUT_ALL',0);
opt.model= "DC";
mpc = loadcase('case30');
busNum= length(mpc.bus);
[Bbus, BfO, Pbusinj, Pfinj] = makeBdc(mpc);%Bf*\theta+Pfinj
BfO= full(BfO);
Bf= BfO*pi/180*100;
nullT= null(Bf);
[ST, UT, VT]= svd(Bf);