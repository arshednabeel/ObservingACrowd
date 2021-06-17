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
k=0;
for rh=3%1:length(fac)
    packdensfinder
    for idv=4:numel(delV_all)
        delV=delV_all(idv);
        for iNr=10%1:numel(Nr_all)
            Nr=round(N*Nr_all(iNr));
            for iens=1:ensemble
                name1=strcat('ObservingAndInferring_29April2019_N',num2str(N),...
                    '_NumberRatio_',num2str(Nr),'_packdens_',num2str(packdens),...
                    '_delV_',num2str(delV),'_Fluc_',num2str(fluc),'_Realization_',...
                    num2str(iens),'.mat');
                k=k+1;
            end
        end
    end
end