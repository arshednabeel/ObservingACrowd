clc
clear

mm=10;nn=15;N=mm*nn;
R=1;

R1=1.10999;

X0=zeros(N,1);
Y0=zeros(N,1);

for j=1:N
    X0(j,1)=sqrt(3)*R1*(ceil(j/mm)-1);
    Y0(j,1)=2*R1*(rem(j,mm)-1)+(1-rem(ceil(j/mm),2))*R1;
end

Xc=min(X0)-R1; Yc=mean(Y0);
X0=X0-Xc; Y0=Y0-Yc;
[xx,yy]=cylinder(R,20);x
xx=xx(1,:); yy=yy(1,:);

len=(max(X0)+R1)-(min(X0)-R1);
wall=(max(Y0)+R1-(min(Y0)))/2;

for j=1:N
    fill(X0(j)+xx,Y0(j)+yy,[0 0 0])
    hold all
    
    if (X0(j)-R)<0
        fill(X0(j)+len+xx, Y0(j)+yy,[0.3 0.3 0.3])
    end
    if (X0(j)+R)>len
        fill(X0(j)-len+xx, Y0(j)+yy,[0.3 0.3 0.3])
    end
    if (Y0(j)-R)< -wall
        fill(X0(j)+xx, Y0(j)+2*wall+yy,[0.3 0.3 0.3])
    end
    if (Y0(j)+R)>wall
        fill(X0(j)+xx, Y0(j)-2*wall+yy,[0.3 0.3 0.3])
    end
    
end
axis equal
axis([0 len -wall wall])
hold off

disp( N*(pi*R^2)/(wall*2*len) )