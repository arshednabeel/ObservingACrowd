function [XA, YA, VXA, VYA]=ABM_bidispese_delv_rho_Nr(N,i0,fluc,X0,Y0,wall,len,delt,t,delV,packdens,iens,Nr)

parameters

v0=zeros(2*N,1);
v0(1:N,1)=v_B*ones(N,1)*delV;
v0(i0)=v_A*delV;

T=ceil(Time/res);

XA=zeros(N,T);
YA=zeros(N,T);
VXA=zeros(N,T);
VYA=zeros(N,T);


% INITIAL CONDITIONS
k=1;
XA(:,k)=X0;
YA(:,k)=Y0;

VXA(:,k)=v0(1:N);
VYA(:,k)=v0(N+1:2*N);

k=k+1; %Update in the storing time step
%--------------------------------------------------------------------------
% i=2 dynamics at Second time step
i=2;
X=XA(:,i-1);
Y=YA(:,i-1);
Vx=VXA(:,i-1);
Vy=VYA(:,i-1);

% [fvx,fvy,fmx,fmy]=agents_Expmemory_per1D(X,Y,Vx,Vy,Mx,My,K,fluc,wall,len,E_agents);
[fvx,fvy]=agents_Expmemory_per2D(X,Y,Vx,Vy,N,v0,fluc,wall,len,delt);

Xprev=X;
Yprev=Y;
Vxprev=Vx;
Vyprev=Vy;


X=Xprev+Vxprev*delt+1/2*fvx*(delt^2);
Y=Yprev+Vyprev*delt+1/2*fvy*(delt^2);
Vx=(X-Xprev)/delt;
Vy=(Y-Yprev)/delt;

if ((i-1)*delt)>=((k-1)*res)
    XA(:,k)=X;
    YA(:,k)=Y;
    VXA(:,k)=Vx;
    VYA(:,k)=Vy;
    k=k+1;
end

%--------------------------------------------------------------------------
% TIME EVOLUTION from i=3 onwards
for i=3:t
%          if rem(i,100)==0; disp(i*delt); end
    [fvx,fvy]=agents_Expmemory_per2D(X,Y,Vx,Vy,N,v0,fluc,wall,len,delt);
    
    Xt=2*X-Xprev+fvx*delt^2;
    Yt=2*Y-Yprev+fvy*delt^2;
    Vxt=(Xt-Xprev)/(2*delt);
    Vyt=(Yt-Yprev)/(2*delt);
    
    Xprev=X;Yprev=Y; %Vxprev=Vx; Vyprev=Vy; Mxprev=Mx; Myprev=My;
    X=Xt;Y=Yt;Vx=Vxt;Vy=Vyt;
    
    if ((i-1)*delt)>=((k-1)*res)
        XA(:,k)=X;
        YA(:,k)=Y;
        VXA(:,k)=Vx;
        VYA(:,k)=Vy;
        k=k+1;
    end
end

save(strcat('ObservingAndInferring_29April2019_N',num2str(N),'_NumberRatio_',num2str(Nr),'_packdens_',num2str(packdens),...
    '_delV_',num2str(delV),'_Fluc_',num2str(fluc),'_Realization_',num2str(iens),'.mat'))
end