% PURPOSE of agents

v0=zeros(2*N,1);

% Initial velocities
In_Vx=v0(1:N); In_Vx(E_agents)=In_Vx(E_agents);
In_Vy=v0(N+1:2*N);