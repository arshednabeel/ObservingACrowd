clc
% periodic boundary (cylindrical space)
per2allSIM=[0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.82 0.83 0.84 0.85 ...
    0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97];
E_agents=104;
fluc=0;

for kj=1:numel(per2allSIM)
    per2=per2allSIM(kj);
    X0=data(:,2);
    Y0=data(:,3);
    dist_XY0=100*ones(N);
    for l=1:N
        for m=l+1:N
            dist_XY0(l,m)=norm([X0(l)-X0(m); Y0(l)-Y0(m)]);
            dist_XY0(m,l)=dist_XY0(l,m);
        end
    end
    
    rA=per2*(min(dist_XY0(:))/2); % Size of agent before scaling
    R=rA/rA; X0=X0/rA; Y0=Y0/rA;
    leftA=min(X0)-(1+per1); rightA=max(X0)+(1+per1); upA=max(Y0)+(1+per1); downA=min(Y0)-(1+per1);
    
    len=rightA-leftA; wall=(upA-downA)/2;
    packdens=(N*pi*R^2)/(2*wall*len);
    
    pd(kj)=packdens;
    dataname=(strcat('EliteAgentsFixdDensityCrowds_N',num2str(N),'_packdens_',num2str(packdens),...
        '_percentR_',num2str(per2),'_EliteNo_',num2str(E_agents),'_Fluc_',num2str(fluc),'.mat'));
    load(dataname)
    
    K=numel(K_full);
    X_a=XA(:,:,[1 3 6 K]);Y_a=YA(:,:,[1 3 6 K]);
    
    Xn=[X_a;Y_a];
    Xn(1:N,:,:)=mod(Xn(1:N,:,:),len);
    Xn(N+1:2*N,:,:)=mod(Xn(N+1:2*N,:,:)+wall,2*wall)-wall;
    del_T=delt;
    %--------------------------------------------------------------------------
    
    [xx,yy]=cylinder(R,20);
    
    % cc(p,:)=cc(p,:)+0.7;
    fig=figure( 'position', [100, 100, 1500, 600]);
    set(gcf,'color','w')
    name=(strcat('EliteAgentsFixdDensityCrowds_N',num2str(N),'_packdens_',num2str(packdens),...
        '_percentR_',num2str(per2),'_EliteNo_',num2str(E_agents),'_Fluc_',num2str(fluc)));
    mo=VideoWriter(name,'MPEG-4'); %#ok<*TNMLP>
    mo.FrameRate = 20;
    mo.Quality=100;
    open(mo);
    
    for j=1:numel(X_a(1,:,1))
        if rem(j,100)==0
            disp(j)
        end
        lf=0.22;
        wf=0.9;
        
        sfh1=subplot(1,4,1);
        sfh1.Position =[0.01 + 0.02 0.05 lf wf];
        voronoi(Xn(1:N,j,1),Xn(N+1:2*N,j,1))
        hold all
        axis equal
        fill(Xn(E_agents,j,1)+xx(1,:), Xn(N+E_agents,j,1)+yy(1,:),[0 0.5 0.5])
        axis([0 len -wall wall])
        hold off
        title(strcat('\rho_d~',num2str(packdens),';','K=',num2str(0)))
        
        sfh2=subplot(1,4,2);
        sfh2.Position = [sfh1.Position(1)+0.02+lf sfh1.Position(2) lf wf];
        voronoi(Xn(1:N,j,2),Xn(N+1:2*N,j,2))
        hold all
        axis equal
        fill(Xn(E_agents,j,2)+xx(1,:), Xn(N+E_agents,j,2)+yy(1,:),[0.9290, 0.6940, 0.1250])
        axis([0 len -wall wall])
        hold off
        title(strcat('K=',num2str(1)))
        
        sfh3=subplot(1,4,3);
        sfh3.Position =[sfh2.Position(1)+0.02+lf sfh2.Position(2) lf wf];
        voronoi(Xn(1:N,j,3),Xn(N+1:2*N,j,3))
        hold all
        axis equal
        fill(Xn(E_agents,j,3)+xx(1,:), Xn(N+E_agents,j,3)+yy(1,:),[1 0.5 0.5])
        axis([0 len -wall wall])
        hold off
        title(strcat('K=',num2str(4)))
        
        sfh4=subplot(1,4,4);
        sfh4.Position =[sfh3.Position(1)+0.02+lf sfh3.Position(2) lf wf];
        voronoi(Xn(1:N,j,4),Xn(N+1:2*N,j,4))
        hold all
        axis equal
        fill(Xn(E_agents,j,4)+xx(1,:), Xn(N+E_agents,j,4)+yy(1,:),[0.25, 0.25, 0.25])
        axis([0 len -wall wall])
        hold off
        title(strcat('K=',num2str(10)))
        
        image=getframe(fig);
        writeVideo(mo,image);
    end
    close(mo)
end