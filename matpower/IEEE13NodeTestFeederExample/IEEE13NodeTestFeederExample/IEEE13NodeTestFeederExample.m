%% IEEE 13 Node Test Feeder
%
% This example shows the use of the Load Flow tool of Powergui to
% initialize the IEEE 13 Node Test Feeder circuit.
%
% G. Sybille (Hydro-Quebec)

% Copyright 2015 Hydro-Quebec, and The MathWorks, Inc.

%%

open_system('IEEE13NodeTestFeeder')

%% Description
% 
% Twelve Load Flow Bus blocks are used to compute an unbalanced load flow
% on a model representing the IEEE 13 Node Test Feeder circuit, originally published by the IEEE Distribution System Analysis Subcommittee Report.
% Note that the model does not include the regulating transformer between
% nodes 650 and 632 of the reference test model.

%% Simulation
%
% Open the Load Flow Tool of Powergui and press the Compute button.
% Press the 'Apply' button to apply the load flow solution to the model in order to start the simulation in steady-state.
% Note that you can view the individual bus voltage magnitude and phase angle values in the corresponding 'Load flow Bus' block's 'Load Flow' tab or you can specify them as block annotations for convenience.
%
% Press the 'Report' button to get a report that shows a load flow summary and detailed load flow results at each bus.
% Partial load flow report results are reproduced in the Power-Flow Result subsystem in the model (The Green block).
% Start simulation and check that it starts in steady state, with expected power flow.
%

