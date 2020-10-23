
% Copyright 2020. All Rights Reserved
% Code by Shuang Li
% For Paper, "Stochastic Iterative Hard Thresholding for Low-Tucker-Rank Tensor Recovery"
% Proc. Information Theory and Applications, La Jolla, California, February 2020.
% by Rachel Grotheer, Shuang Li, Anna Ma, Deanna Needell, and Jing Qin
% This code is used to reproduce Figures 1 and 2. You may need to adjust the
% parameters according to our paper.

%%
clc; clear; close all;

addpath(genpath('tensorlab/'));

N1 = 5; N2 = 5; N3 = 6;
r1 = 1; r2 = 2; r3 = 2;
r = [r1 r2 r3];
M = 360;

Epoches = 250;
Bs = [10 30 60 90 120 180 M];
numb = round(M./Bs);
%%
mu = 0.46*M;                  % Step size
Trial = 100;
%
fcost = zeros(1,Epoches);
errX = zeros(1,Epoches);
runtime = zeros(1,Epoches);
fcost_sto = zeros(length(Bs),Epoches);
errX_sto = zeros(length(Bs),Epoches);
runtime_sto = zeros(length(Bs),Epoches);
for tt = 1:Trial
    fprintf('Trial = %d \n',tt)
    
    S = randn(r1,r2,r3);
    U1 = randn(N1,r1); U2 = randn(N2,r2); U3 = randn(N3,r3);  
    U = cell(1,3);
    U{1} = U1; U{2} = U2; U{3} = U3;
    X = lmlragen(U,S);     % The target low-rank tensor with rank (r1,r2,r3);
    A = randn(M,N1*N2*N3); % Matrix version of sensing tensors.
    normA = norm(A,'fro');
    A = A/normA;
    y = A*vec(X);          % Given measurements
    
    
    % Perform Tensor IHT
    X0 = zeros(N1*N2*N3,1);  % Initialization
    numIter = Epoches;           % Number of iterations    
    XT = X0;   
    %mu = M;       % Step size
    for i = 1:numIter       
        if rem(i,1000) == 0
            fprintf('Iter = %d \n',i)
        end
        t=cputime;
        X_temp = XT+mu*A'*(y-A*XT);
        X_temp = reshape(X_temp,N1,N2,N3);
        [U_temp, S_temp] = mlsvd(X_temp, r);
        XT = lmlragen(U_temp,S_temp);
        XT = vec(XT);
        runtime(1,i) = runtime(1,i) + (cputime-t);
        fcost(1,i) = fcost(1,i)+norm(y-A*XT)^2/2;
        errX(1,i) = errX(1,i)+norm(vec(X)-XT)/norm(vec(X));        
    end
    
    
    % Perform Stochastic Tensor IHT
    XT = X0;    
    for bb = 1:length(Bs)
        bs = Bs(bb);
        fprintf('Batch_size = %d \n',bs)
        numIter = numb(bb);
        mus = mu;%mu/numIter; 
        %mu = M/numIter;
        XT = X0;
        for e = 1:Epoches
            t = cputime;
            for i = 1:numIter               
                % randomize  
                mm = randperm(numIter,1); %randperm(M,bs);
                X_temp = XT+mus*A((mm-1)*bs+1:mm*bs,:)'*(y((mm-1)*bs+1:mm*bs)-A((mm-1)*bs+1:mm*bs,:)*XT); %XT+mus*A(mm,:)'*(y(mm)-A(mm,:)*XT);
                X_temp = reshape(X_temp,N1,N2,N3);
                [U_temp, S_temp] = mlsvd(X_temp, r);
                XT = lmlragen(U_temp,S_temp);
                XT = vec(XT);
            end
            runtime_sto(bb,e) = runtime_sto(bb,e) + (cputime-t);
            fcost_sto(bb,e) = fcost_sto(bb,e)+norm(y-A*XT)^2/2;
            errX_sto(bb,e) = errX_sto(bb,e)+norm(vec(X)-XT)/norm(vec(X));          
        end
    end
    
end

fcost = fcost/Trial;
errX = errX/Trial;
runtime  = runtime/Trial;
fcost_sto =fcost_sto/Trial;
errX_sto = errX_sto/Trial;
runtime_sto = runtime_sto/Trial;

%%
figure
fs = 18;
lw = 2;
set(0, 'defaultlinelinewidth', lw)
set(0,'defaultAxesFontSize', fs)
markers = {'--+','--x', '-.*', ':^', '-o','--d','r-+','k--'};

for jj=1:length(Bs)+1
    if jj == length(Bs)+1
        semilogy(fcost, markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', 'TIHT')
    else
        semilogy(fcost_sto(jj,:), markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', ['StoTIHT b= ' num2str(Bs(jj))])
    end
	hold on    
end
hold off;
legend('show')
xlabel('Epoch','fontsize',fs)
ylabel('$\frac 1 2\|\mathbf{y}-\mathcal{A}(\mathbf{X})\|_2^2$','Interpreter','LaTex','FontSize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w'); 



figure
fs = 18;
lw = 2;
set(0, 'defaultlinelinewidth', lw)
set(0,'defaultAxesFontSize', fs)
markers = {'--+','--x', '-.*', ':^', '-o','--d','r-+','k--'};

for jj=1:length(Bs)+1
    if jj == length(Bs)+1
        semilogy(errX, markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', 'TIHT')
    else
        semilogy(errX_sto(jj,:), markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', ['StoTIHT b= ' num2str(Bs(jj))])
    end
	hold on    
end
hold off;
legend('show')
xlabel('Epoch','fontsize',fs)
ylabel('$\|\mathbf{X}^\star-\widehat{\mathbf{X}}\|_F/\|\mathbf{X}^\star\|_F$','Interpreter','LaTex','FontSize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w'); 




figure
fs = 18;
lw = 2;
set(0, 'defaultlinelinewidth', lw)
set(0,'defaultAxesFontSize', fs)
markers = {'--+','--x', '-.*', ':^', '-o','--d','r-+','k--'};

for jj=1:length(Bs)+1
    if jj == length(Bs)+1
        semilogy(cumsum(runtime),fcost, markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', 'TIHT')
    else
        semilogy(cumsum(runtime_sto(jj,:)),fcost_sto(jj,:), markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', ['StoTIHT b= ' num2str(Bs(jj))])
    end
	hold on    
end
hold off;
legend('show')
xlabel('Running time','fontsize',fs)
ylabel('$\frac 1 2\|\mathbf{y}-\mathcal{A}(\mathbf{X})\|_2^2$','Interpreter','LaTex','FontSize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w'); 



%
figure
fs = 18;
lw = 2;
set(0, 'defaultlinelinewidth', lw)
set(0,'defaultAxesFontSize', fs)
markers = {'--+','--x', '-.*', ':^', '-o','--d','r-+','k--'};

for jj=1:length(Bs)+1
    if jj == length(Bs)+1
        semilogy(cumsum(runtime),errX, markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', 'TIHT')
    else
        semilogy(cumsum(runtime_sto(jj,:)),errX_sto(jj,:), markers{jj}, 'MarkerIndices',1:Epoches/10:Epoches, 'DisplayName', ['StoTIHT b= ' num2str(Bs(jj))])
    end
	hold on    
end
hold off;
legend('show')
xlabel('Running time','fontsize',fs)
ylabel('$\|\mathbf{X}^\star-\widehat{\mathbf{X}}\|_F/\|\mathbf{X}^\star\|_F$','Interpreter','LaTex','FontSize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w'); 






