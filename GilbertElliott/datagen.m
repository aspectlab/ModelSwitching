function datagen(dataset)
%% Description:
% This script is designed to generate flight trajectories using the maneuvering targets model discussed
% in Section 11.7 of Bar-Shalom's "Estimation with Applications to Tracking and Navigation".  Where
% possible, the same notation was used.  The script also includes a genie Kalman filter and
% generates formatted data for use in training and testing a neural network (NN).
%
% Output files from this script can be passed to imm.m and/or the NN in
% Python, and the finals results ultimately plotted with mtplot.m


%% tweakable system parameters
T=1;                        % radar sampling interval (seconds)
P=[0.9995 0.0005; 0.0005 0.9995];   % mode transition probabilities into/out of each mode, rows must sum to 1 (if T changes, avg sojourn time in each mode changes)
%P=[0.99 0.01; 0.01 0.99];   % mode transition probabilities into/out of each mode, rows must sum to 1 (if T changes, avg sojourn time in each mode changes)
F0=[0.3, 0.1; 1 0];            % AR-2 coefficients go in top row of matrix
F1=[1.949, -0.95; 1 0];            % AR-2 coefficients go in top row of matrix
sigma_v2=0.1;               % variance of v (i.e., process noise)
sigma_w2=0.1;               % variance of measurement noise, decreases with time (remaps Kay's time indices)

p=size(F0,1);                   % get dimensions of state and process noise vectors
Q=zeros(p); Q(1,1)=sigma_v2;   % declare Q matrix
R = sigma_w2;  % covariance of measurement noise [maybe makes sense to increase w/velocity?]


%% tweakable data generation parameters for NN
seqLength = 10;             % length of each sequence input to NN

if dataset == 0 % Training set params --
    numSims=4e3;                 % number of realizations
    N=1e3;                       % length of each realization
    SHUFFLE = true;              % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;       % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/train.mat';  % output filename for storing XX and YY variables for NN
    seed = 16;                   % seed for random generator, for reproducible results
    
elseif dataset == 1 % Validation set params --
    numSims=1e3;                 % number of realizations
    N=1e3;                       % length of each realization
    SHUFFLE = false;             % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;       % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/valid.mat';  % output filename for storing XX and YY variables for NN
    seed = 17;                   % seed for random generator, for reproducible results
    
elseif dataset == 2 % Test set params (for performance comparison) --
    numSims=4e3;                 % number of realizations
    N=1e3;                       % length of each realization
    SHUFFLE = false;             % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;       % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/test.mat';   % output filename for storing XX and YY variables for NN
    seed = 18;                   % seed for random generator, for reproducible results
else
    error('Invalid input parameter.')
end

%% non-tweakable parameters and intermediate variables
H=[1 zeros(1,p-1)];              % matrix for system model (observation eq)
numModes = size(P,1);            % number of modes
if INCLUDE_ZEROS
    N2=N+1;                      % need extra samples since last observation isn't used
else
    N2=N+seqLength+1;            % need extra samples since we're skipping initial transient for NN
end


%% allocate space and initialize
x=zeros(p,N2);                      % allocate space for x (state vector)
z=zeros(1,N2);                      % allocate space for z (observation)
x_hat=zeros(p,N2);                  % allocate space for x_hat (estimate), and implictly initialize [zero mean assumption]
x_hat_pred=zeros(p,N2);             % allocate space for x_hat_pred (prediction)
K=zeros(p,1,N2);                    % allocate space for K (genie Kalman gain)
M=zeros(p,p,N2);                    % allocate space for M (minimum mean squared error)
M(:,:,1)=eye(p);                    % initialize M
M_pred=zeros(p,p,N2);               % allocate space for M_pred (minimum predicted mean squared error)
sampleM=zeros(p,p,N2);              % allocate space for sampleM (sample minimum mean squared error)
sampleM_pred=zeros(p,p,N2);         % allocate space for sampleM_pred (sample minimum predicted mean squared error)
XX=zeros(N*numSims, seqLength, 2);  % allocate space for saved data for NN (inputs)
YY=zeros(N*numSims, 2);             % allocate space for saved data for NN (desired outputs)
z_imm=zeros(1, N2, numSims);        % allocate space for saved data for IMM (inputs)
x_imm=zeros(1, N2, numSims);        % allocate space for saved data for IMM (desired outputs)
SqErr_GKF = zeros(N2, numSims);     % allocate space for per-sample and per-realization squared-error


%% generate all random variables
randn('seed',seed);  %#ok<RAND> % set random seed using Octave-compatible legacy approach
rand('seed',seed);   %#ok<RAND>
v_all=[(randn(1,N2,numSims)+1j*randn(1,N2,numSims))/sqrt(2); zeros(p-1,N2,numSims)]*sqrt(sigma_v2);    % generate all process noise
w_all=(randn(1,N2,numSims)+1j*randn(1,N2,numSims))/sqrt(2).*sqrt(sigma_w2);                           % generate all measurement noise
x_init=randn(p,numSims)/sqrt(2)+1j*randn(p,numSims)/sqrt(2);                                        % initial value of x (zero mean, unit variance)
mode = zeros(N2,numSims);                % generate Markov chains to indicate current mode
mode(1,:) = floor(rand(1,numSims)*numModes)+1;
CP = cumsum(P,2);
for n=2:N2
    [~,mode(n,:)] = max(rand(numSims,1)<CP(mode(n-1,:),:),[],2);
end
mode=mode-1; % subtract 1 to index modes by 0, 1, ...
% To only run in GOOD mode, uncomment this --
%mode = zeros(size(mode));
% To only run in BAD mode, uncomment this --
%mode = ones(size(mode));

%% main loop
for k=1:numSims
    
    %% extract random variables for kth realization
    v=v_all(:,:,k);
    w=w_all(:,:,k);
    x(:,1)=x_init(:,k);
    
    %% main loop
    for n=2:N2
        
        %% system update equations
        if mode(n,k)
            F=F1;
        else
            F=F0;
        end
        x(:,n)=F*x(:,n-1)+v(:,n-1);
        z(:,n)=H*x(:,n)+w(:,n);
        
        %% Genie Kalman filter updates (implictly has knowledge of turn rate)
        x_hat_pred(:,n)=F*x_hat(:,n-1);
        M_pred(:,:,n)=F*M(:,:,n-1)*F'+Q;
        K(:,:,n)=M_pred(:,:,n)*H'/(H*M_pred(:,:,n)*H'+R);
        x_hat(:,n)=x_hat_pred(:,n)+K(:,:,n)*(z(:,n)-H*x_hat_pred(:,n));
        M(:,:,n)=(eye(p)-K(:,:,n)*H)*M_pred(:,:,n);
        
        %% compute sample mean square error (i.e., actual, not theoretical)
        e=x_hat(:,n)-x(:,n);                         % error between estimate and true state
        e_pred=x_hat_pred(:,n)-x(:,n);               % error between prediction and true state
        sampleM(:,:,n)=sampleM(:,:,n)+e*e'/numSims;  % sample MSE
        temp=e_pred*e_pred';
        sampleM_pred(:,:,n)=sampleM_pred(:,:,n)+temp/numSims;  % sample prediction MSE
        SqErr_GKF(n,k) = temp(1,1);
        
    end
    
    % store data for NN training / test in Toeplitz-ified form
    if INCLUDE_ZEROS
        X=cat(3,toeplitz(zeros(seqLength,1),real(z(1,1:end-1)))', toeplitz(zeros(seqLength,1),imag(z(1,1:end-1)))'); 
        Y=[real(x(1,2:end))' imag(x(1,2:end))'];
    else
        X=cat(3,toeplitz(real(z(1,seqLength+1:-1:2)),real(z(1,seqLength+1:end-1)))', toeplitz(imag(z(1,seqLength+1:-1:2)),imag(z(1,seqLength+1:end-1)))');
        Y=[real(x(1,seqLength+2:end))' imag(x(1,seqLength+2:end))'];
    end
    X=flip(X,2);
    XX((k-1)*N+1:k*N,:,:)=X;
    YY((k-1)*N+1:k*N,:,:)=Y;
    
    % store the same data serially for IMM and LS implementations
    z_imm(:,:,k) = z;
    x_imm(:,:,k) = x(1,:);
    
end

%% save data to file for NN
if SHUFFLE % useful for training, not so useful for test and debug
    idx=randperm(N*numSims); 
    XX=XX(idx,:,:);
    YY=YY(idx,:);
end
x_hat_pred_gkf = x_hat_pred; % save last genie KF trajectory for plotting
save(outfile,'XX','YY','z_imm', 'x_imm', 'F0', 'F1', 'T', 'P', 'Q', 'R', 'SHUFFLE', 'INCLUDE_ZEROS', 'seed', 'SqErr_GKF', 'x_hat_pred_gkf', 'mode')

% %% report performance of algorithms
% disp(['Averaged over ' num2str(numSims) ' realizations with ' num2str(N2-seqLength-1) ' samples each, the mean'])
% disp('squared prediction error (x and y coords combined) is as follows:')
% disp(' ')
% disp(['genie KF: ' num2str(mean(mean(SqErr_GKF(seqLength+2:end,:))))])
% 
% %% plot trajectory of last realization
% figure(1)
% set(0, 'DefaultLineLineWidth', 2);
% plot(find(mode(:,end)==0),real(x(1,mode(:,end)==0)),'ko')
% hold on
% plot(find(mode(:,end)==1),real(x(1,mode(:,end)==1)),'ks','Linewidth',4)
% plot(real(z),'co')
% plot(real(x_hat_pred(1,:)),'mo')
% plot(real(x(1,:)),'k-','Linewidth',1)
% plot(real(x(1,1)),'g*','Linewidth',8)
% hold off
% legend('true state (good mode)','true state (bad mode)','noisy observation','genie KF prediction')
% grid on
% xlabel('time')
% ylabel('channel gain (real part only)')
% 
% %% plot MSE performance
% figure(3)
% plot(0:N2-2,squeeze(M_pred(1,1,2:end)),0:N2-2,mean(SqErr_GKF(2:end,:),2))
% xlabel('discrete time sample index')
% ylabel('mean squared error')
% grid on
% legend('theoretical genie KF MSE (last run only)', 'experimental genie KF MSE (averaged over all runs)')
% title(['sum of MSE on x and y coordinates over all ' num2str(numSims) ' runs'])
