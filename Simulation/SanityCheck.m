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
            k=0;
            for iens=1:ensemble
                load(strcat('ObservingAndInferring_29April2019_N',num2str(N),...
                    '_NumberRatio_',num2str(Nr),'_packdens_',num2str(packdens),...
                    '_delV_',num2str(delV),'_Fluc_',num2str(fluc),'_Realization_',...
                    num2str(iens),'.mat'))
                
                if (max(XA(:))+max(YA(:)))>1e3
                    Sanity(iens,rh,idv,iNr)=1;
                    disp('insane')
                else
                    k=k+1;
                    ClassificationIdentification
                end
            end
        end
    end
end