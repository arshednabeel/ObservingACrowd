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

Sanity=zeros(ensemble,length(fac),length(delV_all),numel(Nr_all));
for rh=1:length(fac)
    disp(rh)
    packdensfinder
    for idv=1:numel(delV_all)
        delV=delV_all(idv);
        for iNr=1:numel(Nr_all)
            Nr=round(N*Nr_all(iNr));
            for iens=1:ensemble
                R1=R+fac(rh);
                X0=zeros(N,1); Y0=zeros(N,1);
                for j=1:N
                    X0(j,1)=sqrt(3)*R1*(ceil(j/mm)-1);
                    Y0(j,1)=2*R1*(rem(j,mm)-1)+(1-rem(ceil(j/mm),2))*R1;
                end
                Xc=min(X0)-R1; Yc=mean(Y0);
                X0=X0-Xc; Y0=Y0-Yc;[xx,yy]=cylinder(R,20);
                xx=xx(1,:); yy=yy(1,:);
                
                len=(max(X0)+R1)-(min(X0)-R1);wall=(max(Y0)+R1-(min(Y0)))/2;
                packdens=(N*pi*R^2)/(2*wall*len);
                
            end
        end
    end
end