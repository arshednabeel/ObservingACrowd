% PARAMETERS

% Simulation parameters
Time=100;
res=0.1;

% SYSTEM
% N=80;  % elite agent+Inert agents
v_A=-1; v_B=1; % non dimensionalised desired x-velocity; y-velocity is assumed to be zero
R=1; % DO NOT CHANGE: non dimensionalised radius of the circular agent

% PARAMETERS
gam1=0.2; % Time scale ratio
gam2=0.2; % Inter-agent interaction parameter
B=2; % Repulsion between agents to occupy the same space
lcr=3; % Inter-agent interaction cut-off length
% fluc=0; % fluctuation in INERT agents

gam2=gam2*B;