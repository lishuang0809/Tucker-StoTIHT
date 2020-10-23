
% Copyright 2020. All Rights Reserved
% Code by Shuang Li
% For Paper, "Stochastic Iterative Hard Thresholding for Low-Tucker-Rank Tensor Recovery"
% Proc. Information Theory and Applications, La Jolla, California, February 2020.
% by Rachel Grotheer, Shuang Li, Anna Ma, Deanna Needell, and Jing Qin
% This code is used to reproduce Figure 5. You may need to adjust the
% parameters according to our paper.

%%

clc; clear; close all;

addpath(genpath('tensorlab/'));

N1 = 5; N2 = 5; N3 = 6;  % tensor dimension
r_all = [1 1 2;1 2 2; 2 2 2;2 2 3];
m_all = 240:20:460;

%Epoches = 80;
maxEpochs = 1e3;
%bs = 20;    % blocksize
%%
Trial = 100;
%
count_TIHT = zeros(size(r_all,1),length(m_all));
count_StoTIHT = zeros(size(r_all,1),length(m_all));

for tt = 1:Trial
    fprintf('Trial = %d \n',tt);
    for ri = 1:size(r_all,1)
        r = r_all(ri,:);
        r1 = r(1); r2 = r(2); r3 = r(3);
        
        S = randn(r1,r2,r3);
        U1 = randn(N1,r1);  
        U2 = randn(N2,r2);  
        U3 = randn(N3,r3);
        U = cell(1,3);
        U{1} = U1; U{2} = U2; U{3} = U3;
        X = lmlragen(U,S);     % The target low-rank tensor with rank (r1,r2,r3);
        Amax = randn(max(m_all),N1*N2*N3); % Matrix version of sensing tensors.
        for mi = 1:length(m_all)
            m = m_all(mi);
            %fprintf('Trial = %d, r = (%d,%d,%d), m = %d \n',tt,r1,r2,r3,m);
            mu = 0.4*m; % stepsize
            A = Amax(1:m,:); % Matrix version of sensing tensors.
            normA = norm(A,'fro');
            A = A/normA;
            y = A*vec(X);          % Given measurements
            
            % Perform TIHT
            X0 = zeros(N1*N2*N3,1);  % Initialization
            numIter = maxEpochs;       % Number of iterations
            
            XT = X0;
            for i = 1:numIter
                X_temp = XT+mu*A'*(y-A*XT);
                X_temp = reshape(X_temp,N1,N2,N3);
                [U_temp, S_temp] = mlsvd(X_temp, r);
                XT = lmlragen(U_temp,S_temp);
                XT = vec(XT);
                rel_err = norm(vec(X)-XT)/norm(vec(X));
                if rel_err <= 1e-5
                    count_TIHT(ri,mi) = count_TIHT(ri,mi) + i;
                    break;
                end              
            end
            
            
            % Perform StoTIHT
            XT = X0;
            bs = round(m/4);
            numb = round(m/bs);
            numIter = numb;
            for e = 1:maxEpochs
                for i = 1:numIter
                    % randomize
                    mm = randperm(m,bs);
                    X_temp = XT+mu*A(mm,:)'*(y(mm)-A(mm,:)*XT);
                    X_temp = reshape(X_temp,N1,N2,N3);
                    [U_temp, S_temp] = mlsvd(X_temp, r);
                    XT = lmlragen(U_temp,S_temp);
                    XT = vec(XT);
                end
                rel_err = norm(vec(X)-XT)/norm(vec(X));
                if rel_err <= 1e-5
                    count_StoTIHT(ri,mi) = count_StoTIHT(ri,mi) + e;
                    break;
                end
            end
            
            
        end        
    end
    count_TIHT
    count_StoTIHT
end

p_TIHT = count_TIHT./Trial;
p_StoTIHT = count_StoTIHT./Trial;

%%

figure
fs = 18;
lw = 2;
ms = 12;
plot(m_all,p_TIHT(1,:),'r-d','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_TIHT(2,:),'m-s','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_TIHT(3,:),'b-*','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_TIHT(4,:),'k-o','linewidth',lw,'markersize',ms);hold on;
h=legend('$\mathbf{r}=(1,1,2)$','$\mathbf{r}=(1,2,2)$','$\mathbf{r}=(2,2,2)$','$\mathbf{r}=(2,2,3)$');
xlabel('$m$','Interpreter','LaTex','FontSize',fs)
ylabel('Number of epochs','FontSize',fs)
set(h,'Interpreter','LaTex','FontSize',fs,'location','northwest')
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w');
axis tight;



figure
fs = 18;
lw = 2;
ms = 12;
plot(m_all,p_StoTIHT(1,:),'r-d','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_StoTIHT(2,:),'m-s','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_StoTIHT(3,:),'b-*','linewidth',lw,'markersize',ms);hold on;
plot(m_all,p_StoTIHT(4,:),'k-o','linewidth',lw,'markersize',ms);hold on;
h=legend('$\mathbf{r}=(1,1,2)$','$\mathbf{r}=(1,2,2)$','$\mathbf{r}=(2,2,2)$','$\mathbf{r}=(2,2,3)$');
xlabel('$m$','Interpreter','LaTex','FontSize',fs)
ylabel('Number of epochs','FontSize',fs)
set(h,'Interpreter','LaTex','FontSize',fs,'location','northeast')
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w');
axis tight;




