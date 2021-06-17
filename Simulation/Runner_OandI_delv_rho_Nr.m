% ELITE agent movement- HIGH DENSITY CROWD
clc
clear

parameters

ensemble=100;
fluc=0; % Random forces in the system

mm=6;nn=7;N=mm*nn;
R=1;

Nr_all=[1/N linspace(0.05, 0.5, 9)]; % Number Ratio
delV_all=[0.1 0.25 0.5 0.75 1 1.25 1.5 2 2.5 3]; % Relative Velocity
fac=linspace(1,0.24,6); % Packing Density

v0=zeros(2*N,1);
v0(1:N,1)=v_B*ones(N,1);
ind=zeros(N,1);

for rh=1:length(fac)
    tic
    
    disp(rh)
    
    R1=R+fac(rh);
    
    X0=zeros(N,1);
    Y0=zeros(N,1);
    
    for j=1:N
        X0(j,1)=sqrt(3)*R1*(ceil(j/mm)-1);
        Y0(j,1)=2*R1*(rem(j,mm)-1)+(1-rem(ceil(j/mm),2))*R1;
    end
    
    Xc=min(X0)-R1; Yc=mean(Y0);
    X0=X0-Xc; Y0=Y0-Yc;
    [xx,yy]=cylinder(R,20);
    xx=xx(1,:); yy=yy(1,:);
    
    len=(max(X0)+R1)-(min(X0)-R1);
    wall=(max(Y0)+R1-(min(Y0)))/2;
    
    packdens=(N*pi*R^2)/(2*wall*len);
    %     X0=X0+len/2; % shifting right
    
    % SELECTED AGENT INDICES FOR N=199
    parameters
    
    % ---------------------------------------------
    % Time resolution for simulation
    delt=0.01;
    % ---------------------------------------------
    
    t=ceil(Time/delt)+1; % Number of time steps
    
    for  m=1:length(delV_all)
        disp('delV')
        disp(m)
        delV=delV_all(m);
        
        for iens=1:ensemble
            disp('Ens #')
            disp(iens)
            % Randomizing for initial conditions
            tr=round(10/delt);
            op=1;
            while op==1 
                [X_a, Y_a, Vx_a, Vy_a, Mx_a, My_a]=RandomizationOfAgents_InitialConditions(N,0,0.5,X0,Y0,wall,len,1,delt,tr);
                if max(X_a(:))>1e2
                    op=1;
                else
                    op=0;
                end
            end
            
            X0r=X_a(:,end);
            Y0r=Y_a(:,end);
            disp('Randomization done')
            
            parfor iNr=1:numel(Nr_all)
                Nr=round(N*Nr_all(iNr));
                i0=randperm(N,Nr);
%                 tic;
                [XA,YA, VXA, VYA]=ABM_bidispese_delv_rho_Nr(N,i0,fluc,X0r,Y0r,wall,len,delt,t,delV,packdens,iens,Nr);
%                 toc
            end
            
        end
    end
end
