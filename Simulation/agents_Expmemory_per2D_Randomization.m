% Agents, with an Exponentially decaying memory, in a domain periodic in X
% The rate of change of position fx, velocity fv and memory fm of the agents
function [fvx,fvy,fmx,fmy]=agents_Expmemory_per2D_Randomization(X,Y,Vx,Vy,Mx,My,N,~,fluc,wall,len,E_agents,delt)
parameters
parameters_additional

X=mod(X,len);
Y=mod(Y+wall,2*wall)-wall;

X_left=X-len;
Y_left=Y;

X_right=X+len;
Y_right=Y;

X_up=X;
Y_up=Y+2*wall;

X_down=X;
Y_down=Y-2*wall;

X_upleft=X-len;
Y_upleft=Y+2*wall;

X_upright=X+len;
Y_upright=Y+2*wall;

X_downleft=X-len;
Y_downleft=Y-2*wall;

X_downright=X+len;
Y_downright=Y-2*wall;

% Finding inter-agent repulsive forces
f_aax=zeros(N,1);
f_aay=zeros(N,1);

for j=1:N

    d_aab=(sqrt((X(j)-X).^2+(Y(j)-Y).^2));
    V_app=-(Vx.*(X(j)-X)+Vy.*(Y(j)-Y)-Vx(j)*(X(j)-X)-Vy(j)*(Y(j)-Y))./d_aab;
    V_app(j)=1;
    f_aabx=sum((gam2*((d_aab-2*R).^(-B-1)).*(X(j)-X)./(d_aab+eps)).*(d_aab<lcr*R).*(V_app<0));
    f_aaby=sum((gam2*((d_aab-2*R).^(-B-1)).*(Y(j)-Y)./(d_aab+eps)).*(d_aab<lcr*R).*(V_app<0));
    
    %     Right
    d_aar=(sqrt((X(j)-X_right).^2+(Y(j)-Y_right).^2));
    V_app_right=-(Vx.*(X(j)-X_right)+Vy.*(Y(j)-Y_right)-Vx(j)*(X(j)-X_right)-Vy(j)*(Y(j)-Y_right))./d_aar;
    V_app_right(j)=1;
    f_aarx=sum((gam2*((d_aar-2*R).^(-B-1)).*(X(j)-X_right)./(d_aar+eps)).*(d_aar<lcr*R).*(V_app_right<0));
    f_aary=sum((gam2*((d_aar-2*R).^(-B-1)).*(Y(j)-Y_right)./(d_aar+eps)).*(d_aar<lcr*R).*(V_app_right<0));
    
    %     Left
    d_aal=(sqrt((X(j)-X_left).^2+(Y(j)-Y_left).^2));
    V_app_left=-(Vx.*(X(j)-X_left)+Vy.*(Y(j)-Y_left)-Vx(j)*(X(j)-X_left)-Vy(j)*(Y(j)-Y_left))./d_aal;
    V_app_left(j)=1;
    f_aalx=sum((gam2*((d_aal-2*R).^(-B-1)).*(X(j)-X_left)./(d_aal+eps)).*(d_aal<lcr*R).*(V_app_left<0));
    f_aaly=sum((gam2*((d_aal-2*R).^(-B-1)).*(Y(j)-Y_left)./(d_aal+eps)).*(d_aal<lcr*R).*(V_app_left<0));
    
    %     Up
    d_aau=(sqrt((X(j)-X_up).^2+(Y(j)-Y_up).^2));
    V_app_up=-(Vx.*(X(j)-X_up)+Vy.*(Y(j)-Y_up)-Vx(j)*(X(j)-X_up)-Vy(j)*(Y(j)-Y_up))./d_aau;
    V_app_up(j)=1;
    f_aaux=sum((gam2*((d_aau-2*R).^(-B-1)).*(X(j)-X_up)./(d_aau+eps)).*(d_aau<lcr*R).*(V_app_up<0));
    f_aauy=sum((gam2*((d_aau-2*R).^(-B-1)).*(Y(j)-Y_up)./(d_aau+eps)).*(d_aau<lcr*R).*(V_app_up<0));
    
    %     Down
    d_aad=(sqrt((X(j)-X_down).^2+(Y(j)-Y_down).^2));
    V_app_down=-(Vx.*(X(j)-X_down)+Vy.*(Y(j)-Y_down)-Vx(j)*(X(j)-X_down)-Vy(j)*(Y(j)-Y_down))./d_aad;
    V_app_down(j)=1;
    f_aadx=sum((gam2*((d_aad-2*R).^(-B-1)).*(X(j)-X_down)./(d_aad+eps)).*(d_aad<lcr*R).*(V_app_down<0));
    f_aady=sum((gam2*((d_aad-2*R).^(-B-1)).*(Y(j)-Y_down)./(d_aad+eps)).*(d_aad<lcr*R).*(V_app_down<0));
    
    %     Up-left
    d_aaul=(sqrt((X(j)-X_upleft).^2+(Y(j)-Y_upleft).^2));
    V_app_upleft=-(Vx.*(X(j)-X_upleft)+Vy.*(Y(j)-Y_upleft)-Vx(j)*(X(j)-X_upleft)-Vy(j)*(Y(j)-Y_upleft))./d_aaul;
    V_app_upleft(j)=1;
    f_aaulx=sum((gam2*((d_aaul-2*R).^(-B-1)).*(X(j)-X_upleft)./(d_aaul+eps)).*(d_aaul<lcr*R).*(V_app_upleft<0));
    f_aauly=sum((gam2*((d_aaul-2*R).^(-B-1)).*(Y(j)-Y_upleft)./(d_aaul+eps)).*(d_aaul<lcr*R).*(V_app_upleft<0));
    
    %     Up-right
    d_aaur=(sqrt((X(j)-X_upright).^2+(Y(j)-Y_upright).^2));
    V_app_upright=-(Vx.*(X(j)-X_upright)+Vy.*(Y(j)-Y_upright)-Vx(j)*(X(j)-X_upright)-Vy(j)*(Y(j)-Y_upright))./d_aaur;
    V_app_upright(j)=1;
    f_aaurx=sum((gam2*((d_aaur-2*R).^(-B-1)).*(X(j)-X_upright)./(d_aaur+eps)).*(d_aaur<lcr*R).*(V_app_upright<0));
    f_aaury=sum((gam2*((d_aaur-2*R).^(-B-1)).*(Y(j)-Y_upright)./(d_aaur+eps)).*(d_aaur<lcr*R).*(V_app_upright<0));
    
    %     Down-left
    d_aadl=(sqrt((X(j)-X_downleft).^2+(Y(j)-Y_downleft).^2));
    V_app_downleft=-(Vx.*(X(j)-X_downleft)+Vy.*(Y(j)-Y_downleft)-Vx(j)*(X(j)-X_downleft)-Vy(j)*(Y(j)-Y_downleft))./d_aadl;
    V_app_downleft(j)=1;
    f_aadlx=sum((gam2*((d_aadl-2*R).^(-B-1)).*(X(j)-X_downleft)./(d_aadl+eps)).*(d_aadl<lcr*R).*(V_app_downleft<0));
    f_aadly=sum((gam2*((d_aadl-2*R).^(-B-1)).*(Y(j)-Y_downleft)./(d_aadl+eps)).*(d_aadl<lcr*R).*(V_app_downleft<0));
    
    %     Down-right
    d_aadr=(sqrt((X(j)-X_downright).^2+(Y(j)-Y_downright).^2));
    V_app_downright=-(Vx.*(X(j)-X_downright)+Vy.*(Y(j)-Y_downright)-Vx(j)*(X(j)-X_downright)-Vy(j)*(Y(j)-Y_downright))./d_aadr;
    V_app_downright(j)=1;
    f_aadrx=sum((gam2*((d_aadr-2*R).^(-B-1)).*(X(j)-X_downright)./(d_aadr+eps)).*(d_aadr<lcr*R).*(V_app_downright<0));
    f_aadry=sum((gam2*((d_aadr-2*R).^(-B-1)).*(Y(j)-Y_downright)./(d_aadr+eps)).*(d_aadr<lcr*R).*(V_app_downright<0));
    
    f_aax(j,1)=f_aabx+f_aarx+f_aalx+f_aaux+f_aadx+f_aaulx+f_aaurx+f_aadlx+f_aadrx;
    f_aay(j,1)=f_aaby+f_aary+f_aaly+f_aauy+f_aady+f_aauly+f_aaury+f_aadly+f_aadry;
end

% Memory effect
f_Memx(1:N,1)=0*Mx(1:N,1);
f_Memy(1:N,1)=0*My(1:N,1);

f_Memx(E_agents)=0*Mx(1);
f_Memy(E_agents)=0*My(1);

% Purpose or self-propulsion force
f_px=0*v0(1:N)-Vx;
f_py=0*v0(N+1:2*N)-Vy;

% Random force
f_rand=fluc*randn(2*N,1);

% The rate of change of velocity
fvx=f_px+f_Memx+f_aax+f_rand(1:N,1)/sqrt(delt);
fvy=f_py+f_Memy+f_aay+f_rand(N+1:2*N,1)/sqrt(delt);

%The rate of change of position
% fxx=gam1*Vx;
% fxy=gam1*Vy;

% The rate of change of memory
fmx=-Mx/1+(v0(1:N)-Vx);
fmy=-My/1+(v0(N+1:2*N)-Vy);
end