
% Copyright 2020. All Rights Reserved
% Code by Shuang Li
% For Paper, "Stochastic Iterative Hard Thresholding for Low-Tucker-Rank Tensor Recovery"
% Proc. Information Theory and Applications, La Jolla, California, February 2020.
% by Rachel Grotheer, Shuang Li, Anna Ma, Deanna Needell, and Jing Qin
% This code is used to reproduce Figures 6 and 7. You may need to adjust the
% parameters according to our paper.

%%

clc; clear; close all;
load candle_5_A

indx = 61:90;
indy = 71:100;
indf = 1:10;
candle = grayMat(indx,indy,indf);
N1 = size(candle,1);
N2 = size(candle,2);
N3 = size(candle,3);

figure(1)
for i = 1:N3
    imagesc(candle(:,:,i),[0 255]);
    colormap gray;
    axis off;
    title('Candle');
    
    drawnow;
    %pause(0.3)
end

%%
M = 3e4;   % number of measurements
r1 = 8; r2 = 8; r3 = 2;
r = [r1 r2 r3];  % estimated rank
Epoches = 20;
bs = round(M/4);
numb = round(M/bs);
mu = 0.45*M;                  % Step size

A = randn(M,N1*N2*N3); % Matrix version of sensing tensors
normA = norm(A,'fro');
A = A/normA;
X = double(candle)./255;            % The target low-rank tensor with estimated rank r
y = A*vec(X);          % Given measurements

%%

fcost = zeros(1,Epoches);
errX = zeros(1,Epoches);
fcost_sto = zeros(1,Epoches);
errX_sto = zeros(1,Epoches);
% Perform Tensor IHT
fprintf('Perform TIHT: \n')
X0 = zeros(N1*N2*N3,1);  % Initialization
numIter = Epoches;           % Number of iterations
XT = X0;
for i = 1:numIter
    if rem(i,10) == 0
        fprintf('Epoch = %d \n',i)
    end
    X_temp = XT+mu*A'*(y-A*XT);
    X_temp = reshape(X_temp,N1,N2,N3);
    [U_temp, S_temp] = mlsvd(X_temp, r);
    XT = lmlragen(U_temp,S_temp);
    XT = vec(XT);
    fcost(i) = norm(y-A*XT)^2/2;
    errX(i) = norm(vec(X)-XT)/norm(vec(X));
end

XT_candle = reshape(XT,N1,N2,N3).*255;

% Perform Stochastic Tensor IHT
fprintf('Perform StoTIHT: \n')
XT_sto = X0;
numIter = numb;
for e = 1:Epoches
    if rem(e,10) == 0
        fprintf('Epoch = %d \n',e)
    end
    for i = 1:numIter
        % randomize
        mm = randperm(M,bs);
        X_temp = XT_sto+mu*A(mm,:)'*(y(mm)-A(mm,:)*XT_sto);
        X_temp = reshape(X_temp,N1,N2,N3);
        [U_temp, S_temp] = mlsvd(X_temp, r);
        XT_sto = lmlragen(U_temp,S_temp);
        XT_sto = vec(XT_sto);
    end
    fcost_sto(e) = norm(y-A*XT_sto)^2/2;
    errX_sto(e) = norm(vec(X)-XT_sto)/norm(vec(X));
end

XTsto_candle = reshape(XT_sto,N1,N2,N3).*255;

%%

figure
fs = 18;
lw = 2;
semilogy(fcost_sto,'r-','linewidth',lw);hold on;
semilogy(fcost,'k--','linewidth',lw);hold on;
h=legend('StoTIHT','TIHT');
xlabel('Epoch','fontsize',fs)
ylabel('$\frac 1 2\|\mathbf{y}-\mathcal{A}(\mathbf{X})\|_2^2$','Interpreter','LaTex','FontSize',fs)
set(h,'fontsize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w');



figure
fs = 18;
lw = 2;
semilogy(errX_sto,'r-','linewidth',lw);hold on;
semilogy(errX,'k--','linewidth',lw);hold on;
h=legend('StoTIHT','TIHT');
xlabel('Epoch','fontsize',fs)
ylabel('$\|\mathbf{X}^\star-\widehat{\mathbf{X}}\|_F/\|\mathbf{X}^\star\|_F$','Interpreter','LaTex','FontSize',fs)
set(h,'fontsize',fs)
set(gca,'fontsize',fs)
set(gcf, 'Color', 'w');


%%
figure
fs1 = 15;
fs2 = 15;
set(gcf,'position',[10,500,800,320])
for i = 1:N3
    subplot(2,N3/2,i)
    imagesc(candle(:,:,i),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs1)
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');

%export_fig '/Users/SS/Dropbox (Personal)/Shared WISDM materials/SWiM/Tucker rank paper/fig/test_StoTIHT_video_true2.pdf'



figure
fs1 = 15;
fs2 = 15;
set(gcf,'position',[10,500,800,320])
for i = 1:N3
    subplot(2,N3/2,i)
    imagesc(XT_candle(:,:,i),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs1)
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');



figure
fs1 = 15;
fs2 = 15;
set(gcf,'position',[10,500,800,320])
for i = 1:N3
    subplot(2,N3/2,i)
    imagesc(XTsto_candle(:,:,i),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs1)
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');


%%
figure
fs1 = 15;
fs2 = 18;
set(gcf,'position',[10,500,420,280])
for i = N3-1:N3
    subplot(1,2,i-(N3-2))
    imagesc(candle(:,:,i-(N3-2)),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');



figure
fs1 = 15;
fs2 = 18;
set(gcf,'position',[10,500,420,280])
for i = N3-1:N3
    subplot(1,2,i-(N3-2))
    imagesc(XT_candle(:,:,i-(N3-2)),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');




figure
fs1 = 15;
fs2 = 18;
set(gcf,'position',[10,500,420,280])
for i = N3-1:N3
    subplot(1,2,i-(N3-2))
    imagesc(XTsto_candle(:,:,i-(N3-2)),[0 255]);
    colormap gray;
    title(['Frame ', num2str(i)],'FontSize',fs1);
    set(gca,'FontSize',fs2)
end
set(gcf, 'Color', 'w');


