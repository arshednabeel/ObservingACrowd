% periodic boundary (cylindrical space)
figure

K=6;
X_a=XA(:,:,K);
Y_a=YA(:,:,K);

parameters
parameters_additional

TT=250;length(X_a(1,:));
delt=(Time/length(X_a(1,:)));

Xn=[X_a;Y_a];
Xn(1:N,:)=mod(Xn(1:N,:),len);
Xn(N+1:2*N,:)=mod(Xn(N+1:2*N,:)+wall,2*wall)-wall;
%--------------------------------------------------------------------------

[xx,yy]=cylinder(R,20);
cc=zeros(N,3);
for i=1:N
    if ind(i)==1
        cc(i,:)=[1 0.5 0.5];
    else
        cc(i,:)=[0 0 0];
    end
end
% cc(p,:)=cc(p,:)+0.7;
kk=0;
for j=round(linspace(1,TT,10))
    disp(j)
    kk=kk+1;
    subplot(2,5,kk)

    hold on
    for i=1:N
        
        fill(Xn(i,j)+xx(1,:), Xn(N+i,j)+yy(1,:),cc(i,:))
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
        
        hold on
        axis equal
        axis([0 len -wall wall])
    end
    plot([0 len len 0],[-wall -wall wall wall],'--k')
    title(num2str((j-1)*delt))
    
end