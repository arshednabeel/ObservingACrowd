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