close all
clc

K=6;
% periodic boundary (cylindrical space)
X_a=XA(:,:,K);
Y_a=YA(:,:,K);

Xn=[X_a;Y_a];
Xn(1:N,:)=mod(Xn(1:N,:),len);
Xn(N+1:2*N,:)=mod(Xn(N+1:2*N,:)+wall,2*wall)-wall;
del_T=delt;
%--------------------------------------------------------------------------

[xx,yy]=cylinder(R,20);
cc=zeros(N,3);
if K==1
    cc(E_agents,:)=[1 0.5 0.5];
else
    cc(E_agents,:)=[0 0.5 0.5];
end

% cc(p,:)=cc(p,:)+0.7;
fig=figure;
name=(strcat('RegularCrowdEliteMotion_N',num2str(N),'_packdens_'...
    ,num2str(packdens),'_EliteNo_',num2str(E_agents),'_K_',num2str(K_full(K))));
mo=VideoWriter(name,'MPEG-4');
mo.FrameRate = 20;
open(mo);

for j=1:1:numel(X_a(1,:))
    if rem(j,100)==0
        disp(j)
    end
    for i=1:N
        
        fill(Xn(i,j)+xx(1,:), Xn(N+i,j)+yy(1,:),cc(i,:))
        hold on
        axis equal
        
        if (Xn(i,j)-R)<0
            fill(Xn(i,j)+len+xx(1,:), Xn(N+i,j)+yy(1,:),[0.3 0.3 0.3])
        end
        if (Xn(i,j)+R)>len
            fill(Xn(i,j)-len+xx(1,:), Xn(N+i,j)+yy(1,:),[0.3 0.3 0.3])
        end
        if (Xn(N+i,j)-R)<-wall
            fill(Xn(i,j)+xx(1,:), Xn(N+i,j)+2*wall+yy(1,:),[0.3 0.3 0.3])
        end
        if (Xn(N+i,j)+R)>wall
            fill(Xn(i,j)+xx(1,:), Xn(N+i,j)-2*wall+yy(1,:),[0.3 0.3 0.3])
        end
                
        axis([0 len -wall wall])
        title(num2str((j-1)*res))
    end
    image=getframe(fig);
    writeVideo(mo,image);
    hold off
end
close(mo)