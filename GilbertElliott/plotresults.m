%% Description:
% This script plots results of the genie KF, IMM, and TCN for the 
% manuevering targets problem.  Generally, the following sequence
% of scripts will be run:
%  1. main_datagen.m (creates trajectory data)
%  2. imm.m (loads trajectory data, saves result to file)
%  3. main_tcn.py TCN code in Python (loads trajectory data, saves result to file)
%  4. mtplot.m (this script, which loads three files from the above steps)

%% tweakable system parameters
infile1 = 'data/test.mat';      % file of input data (target trajectories, system parameters)
infile2 = 'data/test_imm.mat';  % file of IMM results
infile3 = 'data/test_tcn.mat';  % file of TCN results... should contain a single matrix called YY
                            % with length equal to N*numSims -by- 2
windowSize=35;              % determines how many samples after jump is still 
                              % considered "transition" region in computing conditioned MSE
                            
%% load data files, rename some stuff
load(infile1)
seqLength=size(XX,2);
clear XX YY
x=x_imm;
z=z_imm;
load(infile2)
load(infile3,'YY')
if SHUFFLE
    error('This script not meant for use with SHUFFLE mode.')
end

%% determine some intermediate params
[~,N2,numSims]=size(x_imm);
if INCLUDE_ZEROS
    N=N2-1;
else
    N=N2-seqLength-1;
end

%% process TCN data (reshape, compute MSE)
YY=reshape(YY',2,N,numSims);            % reshape into realizations
if INCLUDE_ZEROS % for each case prepend with appropriate # of NaNs where TCN didn't estimate first seqLength time steps
    YY=cat(2,NaN*ones(2,1,numSims),YY);  
else
    YY=cat(2,NaN*ones(2,seqLength+1,numSims),YY);
end
YY=YY(1,:,:)+1j*YY(2,:,:);
SqErr_TCN = squeeze((abs(YY-x_imm)).^2);  % compute MSE of bias-adjusted NN predicted outputs with true states
x_hat_pred_tcn=YY(:,:,end);             % grab last realization


%% plot trajectory of last realization
figure(1)
set(0, 'DefaultLineLineWidth', 3);
plot(find(mode(:,end)==0),real(x(1,mode(:,end)==0,end)),'ko')
hold on
plot(find(mode(:,end)==1),real(x(1,mode(:,end)==1,end)),'ks','Linewidth',4)
plot(real(z(1,:,end)),'co')
plot(real(x_hat_pred_gkf(1,:)),'mo')
plot(real(x_hat_pred_imm(1,:)),'go')
plot(real(x_hat_pred_tcn(1,:)),'b*')
plot(real(x(1,:,end)),'k-','Linewidth',1)
hold off
legend('true state (good mode)','true state (bad mode)','noisy observation','genie KF prediction')
grid on
xlabel('time')
ylabel('channel gain (real part only)')


%% plot IMM verification params
figure(2)
plot(mu(2,:))
grid on
xlabel('discrete time')
ylabel('probability')
title('IMM estimated "bad" mode probability')

%% plot MSE performance
figure(3)
plot(0:N2-2,mean(SqErr_GKF(2:end,:),2),0:N2-2,mean(SqErr_IMM(2:end,:),2),0:N2-2,mean(SqErr_TCN(2:end,:),2))
xlabel('discrete time sample index')
ylabel('mean squared error')
grid on
legend('genie KF','IMM','TCN')
title(['MSE over all ' num2str(numSims) ' realizations'])

%% compute conditional MSE performance in 4 regimes (steady state good and bad, transition between the two)
% find location of all mode jumps and extend these to "transition" regions
CVtransitionInd=[];   % for storing transitions into good mode
CTtransitionInd=[];   % for storing transitions into bad mode
CVsteadystateInd=[];  % for steady state good mode samples
CTsteadystateInd=[];  % for steady state bad mode samples
for i=1:numSims
    
    % first find all jumps and transition windows, regardless of whether bad->good or good->bad
    idx_jump=find(abs(diff(mode(:,i)))~=0)+1;
    idx_window=[];
    for j=1:size(idx_jump,1)
        idx_window=[idx_window; (idx_jump(j):min([N2 idx_jump(j)+windowSize-1]))'];
    end
    idx_window=unique(idx_window);  % remove dupes for cases when soujourns into one state are shorter than windowSize
    idx_window=idx_window(idx_window>seqLength+1);  % skip if in initial transient
    
    CVtransitionInd=[CVtransitionInd; idx_window(find(mode(idx_window,i)==0))+N2*(i-1)];  % save them all
    CTtransitionInd=[CTtransitionInd; idx_window(find(mode(idx_window,i)==1))+N2*(i-1)];  % save them all
    
    idx_steadystate=setdiff((seqLength+2:N2)', idx_window);
    CVsteadystateInd=[CVsteadystateInd; idx_steadystate(find(mode(idx_steadystate,i)==0))+N2*(i-1)];  % save good steady state modes
    CTsteadystateInd=[CTsteadystateInd; idx_steadystate(find(mode(idx_steadystate,i)==1))+N2*(i-1)];  % save bad steady state modes
    
end

%% output results
fprintf('The following is a table of prediction MSEs,\nincluding MSE conditioned on being in either the two steady\nstate modes or the two types of transitions. Moreover,\nit was created from %d realizations with %d samples each,\nand the MSE includes the x- and y-coordinates.\n\n',numSims,N2-seqLength-1)
fprintf('Approach | Overall MSE | good SS| bad SS | gd->bd | bd->gd \n')
fprintf('---------|-------------|--------|--------|--------|--------\n')
fprintf('Genie KF |   %6.4f    | %6.4f | %6.4f | %6.4f | %6.4f\n',mean(mean(SqErr_GKF(seqLength+2:end,:))),mean(SqErr_GKF(CVsteadystateInd)),mean(SqErr_GKF(CTsteadystateInd)),mean(SqErr_GKF(CTtransitionInd)),mean(SqErr_GKF(CVtransitionInd)))
fprintf('  IMM    |   %6.4f    | %6.4f | %6.4f | %6.4f | %6.4f\n',mean(mean(SqErr_IMM(seqLength+2:end,:))),mean(SqErr_IMM(CVsteadystateInd)),mean(SqErr_IMM(CTsteadystateInd)),mean(SqErr_IMM(CTtransitionInd)),mean(SqErr_IMM(CVtransitionInd)))
fprintf('  TCN    |   %6.4f    | %6.4f | %6.4f | %6.4f | %6.4f\n',mean(mean(SqErr_TCN(seqLength+2:end,:))),mean(SqErr_TCN(CVsteadystateInd)),mean(SqErr_TCN(CTsteadystateInd)),mean(SqErr_TCN(CTtransitionInd)),mean(SqErr_TCN(CVtransitionInd)))
fprintf('---------|-------------|--------|--------|--------|--------\n')
fprintf('fraction |             |        |        |        |\n')
fprintf(' of time |             | %6.3f | %6.3f | %6.3f | %6.3f\n\n',size(CVsteadystateInd,1)/N/numSims,size(CTsteadystateInd,1)/N/numSims,size(CTtransitionInd,1)/N/numSims,size(CVtransitionInd,1)/N/numSims)


%% plot some gd->bd transition MSE behavior 
CV2CT=[];
CT2CV=[];
skipPlot=seqLength-5;  % # samples to skip in plotting to right of jump
for i=1:numSims
    idxCV2CT=strfind(mode(:,i)',[zeros(windowSize,1); ones(2*windowSize,1)]')';
    idxCV2CT=idxCV2CT((idxCV2CT<=N2-2*windowSize+1) & (idxCV2CT>=seqLength+2));  % avoid a window that exceeds array bounds or starts too early
    idxCT2CV=strfind(mode(:,i)',[ones(windowSize,1); zeros(2*windowSize,1)]')';
    idxCT2CV=idxCT2CV((idxCT2CV<=N2-2*windowSize+1) & (idxCT2CV>=seqLength+2));  % avoid a window that exceeds array bounds or starts too early
    for j=1:size(idxCV2CT,1)
        CV2CT=[CV2CT; (idxCV2CT(j):idxCV2CT(j)+2*windowSize-1)+N2*(i-1)];
    end
    for j=1:size(idxCT2CV,1)
        CT2CV=[CT2CV; (idxCT2CV(j):idxCT2CV(j)+2*windowSize-1)+N2*(i-1)];
    end
end
CT2CV=CT2CV(:,skipPlot+1:end);
CV2CT=CV2CT(:,skipPlot+1:end);
figure(4)
idx=-windowSize+skipPlot:windowSize-1;
plot(idx,mean(SqErr_GKF(CT2CV)),'-o',idx,mean(SqErr_IMM(CT2CV)),'-o',idx,mean(SqErr_TCN(CT2CV)),'-o')
xlabel('discrete time index (time=0 is where jump occurs)')
ylabel('MSE')
grid on
title('prediction MSE during jump from bad to good mode')
legend('GKF','IMM','TCN')
figure(5)
plot(idx,mean(SqErr_GKF(CV2CT)),'-o',idx,mean(SqErr_IMM(CV2CT)),'-o',idx,mean(SqErr_TCN(CV2CT)),'-o')
xlabel('discrete time index (time=0 is where jump occurs)')
ylabel('MSE')
title('prediction MSE during jump from good to bad mode')
grid on
legend('GKF','IMM','TCN')
