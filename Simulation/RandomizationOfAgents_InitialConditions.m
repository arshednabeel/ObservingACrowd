function [X_a, Y_a, Vx_a, Vy_a, Mx_a, My_a]=RandomizationOfAgents_InitialConditions(N,K,fluc,X0,Y0,wall,len,E_agents,delt,t)

parameters
parameters_additional

T=ceil(Time/res);

X_a=zeros(N,T);
Y_a=zeros(N,T);
Vx_a=zeros(N,T);
Vy_a=zeros(N,T);
Mx_a=zeros(N,T);
My_a=zeros(N,T);

% INITIAL CONDITIONS
k=1;
X_a(:,k)=X0;
Y_a(:,k)=Y0;
Vx_a(:,k)=In_Vx;
Vy_a(:,k)=In_Vy;
Mx_a(:,k)=v0(1:N)-Vx_a(:,1);
My_a(:,k)=v0(N+1:2*N)-Vy_a(:,1);
%--------------------------------------------------------------------------
% i=2 dynamics at Second time step
i=2;
X=X_a(:,i-1);
Y=Y_a(:,i-1);
Vx=Vx_a(:,i-1);
Vy=Vy_a(:,i-1);
Mx=Mx_a(:,i-1);
My=My_a(:,i-1);

% [fvx,fvy,fmx,fmy]=agents_Expmemory_per1D(X,Y,Vx,Vy,Mx,My,K,fluc,wall,len,E_agents);
[fvx,fvy,fmx,fmy]=agents_Expmemory_per2D_Randomization(X,Y,Vx,Vy,Mx,My,N,K,fluc,wall,len,E_agents,delt);

Xprev=X;
Yprev=Y;
Vxprev=Vx;
Vyprev=Vy;
Mxprev=Mx;
Myprev=My;

X=Xprev+gam1*Vxprev*delt+1/2*fvx*(delt^2);
Y=Yprev+gam1*Vyprev*delt+1/2*fvy*(delt^2);
Vx=(X-Xprev)/delt;
Vy=(Y-Yprev)/delt;
Mx=Mxprev+fmx*delt;
My=Myprev+fmy*delt;

X_a(:,i)=X;
Y_a(:,i)=Y;
Vx_a(:,i)=Vx;
Vy_a(:,i)=Vy;
Mx_a(:,i)=Mx;
My_a(:,i)=My;

%--------------------------------------------------------------------------
% TIME EVOLUTION from i=3 onwards
for i=3:t
    [fvx,fvy,fmx,fmy]=agents_Expmemory_per2D_Randomization(X,Y,Vx,Vy,Mx,My,N,K,fluc,wall,len,E_agents,delt);
    
    Xt=2*X-Xprev+fvx*delt^2;
    Yt=2*Y-Yprev+fvy*delt^2;
    Vxt=(Xt-Xprev)/(2*delt);
    Vyt=(Yt-Yprev)/(2*delt);
    Mxt=Mx+fmx*delt;
    Myt=My+fmy*delt;
    
    Xprev=X;Yprev=Y; %Vxprev=Vx; Vyprev=Vy; Mxprev=Mx; Myprev=My;
    X=Xt;Y=Yt;Vx=Vxt;Vy=Vyt;Mx=Mxt;My=Myt;
    
    X_a(:,i)=X;
    Y_a(:,i)=Y;
    Vx_a(:,i)=Vx;
    Vy_a(:,i)=Vy;
    Mx_a(:,i)=Mx;
    My_a(:,i)=My;
end
% Vx_a(:,T)=(X_a(:,T)-X_a(:,T-1))/delt/gam1;
% Vy_a(:,T)=(Y_a(:,T)-Y_a(:,T-1))/delt/gam1;
end