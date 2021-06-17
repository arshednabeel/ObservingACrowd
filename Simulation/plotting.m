% periodic boundary (cylindrical space)
close all
figure

parameters

ind=zeros(N,1);
ind(i0)=1;

X_a=XA;Y_a=YA;

TT=length(X_a(1,:));
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

for j=[1 TT]%1:round(15/delt):t
    if rem(j,100)==0
        disp(j)
    end

    for i=1:N
        fill(Xn(i,j)+xx(1,:), Xn(N+i,j)+yy(1,:),cc(i,:))
        hold on
        axis equal
        axis([0 len -wall wall])
    end
    
    plot([0 len len 0],[-wall -wall wall wall],'--k')
    title(num2str((j-1)*delt))
    axis([0 len -wall wall])
    
    drawnow limitrate
    hold off
%     pause 
end