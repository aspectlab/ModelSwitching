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
P=[0.99 0.01; 0.01 0.99];   % mode transition probabilities into/out of each mode, rows must sum to 1 (if T changes, avg sojourn time in each mode changes)
velocity_init_mean = 120;   % mean of initial velocity
velocity_init_var = 30^2;   % variance of initial velocity
position_init_var = 1e3;    % variance of initial location (about the origin)
Om_init_mean = 4*pi/180;    % mean of initial turn rate each time system enters CT state (rad/sec)
Om_init_var = (pi/180)^2;   % variance of initial turn rate
Om_var = (.2*pi/180)^2;     % variance of process noise (turn rate, v2) [might be too large?]
Q = 1*eye(2);               % covariance of process noise (position and velocity, v1) [could prob crank this up?]
R = (0.2*velocity_init_mean)^2*eye(2);  % covariance of measurement noise [maybe makes sense to increase w/velocity?]

%% tweakable data generation parameters for NN
seqLength = 25;             % length of each sequence input to NN

if dataset == 0 % Training set params --
    numSims=4e3;                % number of realizations
    N=1e3;                      % length of each realization
    SHUFFLE = true;             % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;      % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/train.mat';      % output filename for storing XX and YY variables for NN
    seed = 17;                   % seed for random generator, for reproducible results
elseif dataset == 1 % Validation set params --
    numSims=1e3;                % number of realizations
    N=1e3;                      % length of each realization
    SHUFFLE = false;            % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;      % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/valid.mat';      % output filename for storing XX and YY variables for NN
    seed = 18;                   % seed for random generator, for reproducible results
elseif dataset == 2 % Test set params (for performance comparison) --
    numSims=4e3;                % number of realizations
    N=1e3;                      % length of each realization
    SHUFFLE = false;            % shuffles data (across realizations) before saving
    INCLUDE_ZEROS = false;      % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
    outfile = 'data/test.mat';      % output filename for storing XX and YY variables for NN
    seed = 19;                   % seed for random generator, for reproducible results
else
    error('Invalid input parameter.')
end

%% non-tweakable parameters and intermediate variables
G=[T^2/2 0; T 0; 0 T^2/2; 0 T];  % matrix for system model (state eq)
H=[1 0 0 0; 0 0 1 0];            % matrix for system model (observation eq)
numModes = size(P,1);            % number of modes
if INCLUDE_ZEROS
    N2=N+1;                      %#ok<UNRCH> % need extra samples since last observation isn't used
else
    N2=N+seqLength+1;            % need extra samples since we're skipping initial transient for NN
end


%% allocate space and initialize
x=zeros(4,N2);                      % allocate space for x (state vector)
Om=zeros(N2,1);                     % allocate space for Om (turn rate state)
z=zeros(2,N2);                      % allocate space for z (observation)
x_hat=zeros(4,N2);                  % allocate space for x_hat (estimate), and implictly initialize [zero mean assumption]
x_hat_pred=zeros(4,N2);             % allocate space for x_hat_pred (prediction)
K=zeros(4,2,N2);                    % allocate space for K (genie Kalman gain)
M=zeros(4,4,N2);                    % allocate space for M (minimum mean squared error), and initialize
temp=(velocity_init_mean^2+velocity_init_var)/2;  % temp var used in initializing M
M(:,:,1)=diag([position_init_var/2 temp position_init_var/2 temp]);     % initialize M
M_pred=zeros(4,4,N2);               % allocate space for M_pred (minimum predicted mean squared error)
sampleM=zeros(4,4,N2);              % allocate space for sampleM (sample minimum mean squared error)
sampleM_pred=zeros(4,4,N2);         % allocate space for sampleM_pred (sample minimum predicted mean squared error)
XX=zeros(N*numSims, seqLength, 2);  % allocate space for saved data for NN (inputs)
YY=zeros(N*numSims, 2);             % allocate space for saved data for NN (desired outputs)
z_imm=zeros(2, N2, numSims);        % allocate space for saved data for IMM (inputs)
x_imm=zeros(2, N2, numSims);        % allocate space for saved data for IMM (desired outputs)
SqErr_GKF = zeros(N2, numSims);     % allocate space for per-sample and per-realization squared-error

%% generate all random variables
randn('seed',seed);  %#ok<RAND> % set random seed using Octave-compatible legacy approach
rand('seed',seed);   %#ok<RAND>
v1_all=zeros(2,N2,numSims);
w_all=zeros(2,N2,numSims);
for i=1:numSims
    v1_all(:,:,i)=sqrtm(Q)*randn(2,N2);  % generate all process noise for position/velocity
    w_all(:,:,i)=sqrtm(R)*randn(2,N2);   % generate all measurement noise
end
v2_all=sqrt(Om_var)*randn(N2,numSims);   % generate all process noise for turn rate
magvel=randn(1,numSims)*sqrt(velocity_init_var)+velocity_init_mean;  % mag of init velocity is Gaussian
direction=rand(1,numSims)*2*pi;          % direction of init velocity is uniform on (0,2pi)
x_init=[randn(1,numSims)*sqrt(position_init_var/2);  % random initial x position
    magvel.*cos(direction);                      % random initial x velocity
    randn(1,numSims)*sqrt(position_init_var/2);  % random initial y position
    magvel.*sin(direction)];                     % random initial y velocity
mode = zeros(N2,numSims);                % generate Markov chains to indicate current mode
mode(1,:) = floor(rand(1,numSims)*numModes)+1;
CP = cumsum(P,2);
for n=2:N2
    [~,mode(n,:)] = max(rand(numSims,1)<CP(mode(n-1,:),:),[],2);
end
mode=mode-1; % subtract 1 to index modes by 0, 1, ...
% To only run in CV mode, uncomment this --
%mode = zeros(size(mode));
% To only run in CT mode, uncomment this --
%mode = ones(size(mode));
maxTurns=max(floor(sum((diff(mode))~=0)/2)+1);
Om_init=(randn(maxTurns,numSims)*sqrt(Om_init_var)+Om_init_mean).*sign(randn(maxTurns,numSims));  % random initial turn rate at start of each turn, folded normal distribution to account for L or R turns

%% main loop
for k=1:numSims
    
    %% extract random variables for kth realization
    v1=v1_all(:,:,k);
    v2=v2_all(:,k);
    w=w_all(:,:,k);
    x(:,1)=x_init(:,k);
    turnCtr=0;  % initialize turn counter
    
    %% main loop
    for n=2:N2
        
        %% system update equations
        % update turn rate state Om
        if mode(n,k) && ((mode(n,k) ~= mode(n-1,k)) || n==2) % just entered CT mode
            turnCtr=turnCtr+1;
            Om(n)=Om_init(turnCtr,k);
        elseif mode(n,k) % CT mode (but not first time entering this mode)
            Om(n)=Om(n-1)+T*v2(n-1);
        else % CV mode
            Om(n)=0;
        end
        
        % update all other states and observation
        F = getF(Om(n),T);
        x(:,n)=F*x(:,n-1)+G*v1(:,n-1);
        z(:,n)=H*x(:,n)+w(:,n);
        
        %% Genie Kalman filter updates (implictly has knowledge of turn rate)
        x_hat_pred(:,n)=F*x_hat(:,n-1);
        M_pred(:,:,n)=F*M(:,:,n-1)*F'+G*Q*G';
        K(:,:,n)=M_pred(:,:,n)*H'/(H*M_pred(:,:,n)*H'+R);
        x_hat(:,n)=x_hat_pred(:,n)+K(:,:,n)*(z(:,n)-H*x_hat_pred(:,n));
        M(:,:,n)=(eye(4)-K(:,:,n)*H)*M_pred(:,:,n);
        
        %% compute sample mean square error (i.e., actual, not theoretical)
        e=x_hat(:,n)-x(:,n);                         % error between estimate and true state
        e_pred=x_hat_pred(:,n)-x(:,n);               % error between prediction and true state
        sampleM(:,:,n)=sampleM(:,:,n)+e*e'/numSims;  % sample MSE
        temp=e_pred*e_pred';
        sampleM_pred(:,:,n)=sampleM_pred(:,:,n)+temp/numSims;  % sample prediction MSE
        SqErr_GKF(n,k) = temp(1,1)+temp(3,3);
        
    end
    
    % store data for NN training / test in Toeplitz-ified form
    if INCLUDE_ZEROS
        X=cat(3,toeplitz(zeros(seqLength,1),z(1,1:end-1))', toeplitz(zeros(seqLength,1),z(2,1:end-1))'); %#ok<UNRCH>
        Y=[x(1,2:end)' x(3,2:end)'];
    else
        X=cat(3,toeplitz(z(1,seqLength+1:-1:2),z(1,seqLength+1:end-1))', toeplitz(z(2,seqLength+1:-1:2),z(2,seqLength+1:end-1))');
        Y=[x(1,seqLength+2:end)' x(3,seqLength+2:end)'];
    end
    X=flip(X,2);
    offset=X(:,end,:);
    XX((k-1)*N+1:k*N,:,:)=X-offset;
    YY((k-1)*N+1:k*N,:,:)=Y-squeeze(offset);
    
    % store the same data serially for IMM and LS implementations
    z_imm(:,:,k) = z;
    x_imm(:,:,k) = x([1 3],:);
    
end

%% save data to file for NN
if SHUFFLE % useful for training, not so useful for test and debug
    idx=randperm(N*numSims); %#ok<UNRCH>
    XX=XX(idx,:,:);
    YY=YY(idx,:);
end
x_hat_pred_gkf = x_hat_pred; % save last genie KF trajectory for plotting
save(outfile,'XX','YY','z_imm', 'x_imm', 'T', 'P', 'velocity_init_mean', 'velocity_init_var', 'position_init_var', 'Om_init_mean', 'Om_init_var', 'Om_var', 'Q', 'R', 'SHUFFLE', 'INCLUDE_ZEROS', 'seed', 'SqErr_GKF', 'x_hat_pred_gkf', 'mode', 'Om')

% %% report performance of algorithms
% disp(['Averaged over ' num2str(numSims) ' realizations with ' num2str(N2-seqLength-1) ' samples each, the mean'])
% disp('squared prediction error (x and y coords combined) is as follows:')
% disp(' ')
% disp(['genie KF: ' num2str(mean(mean(SqErr_GKF(seqLength+2:end,:))))])
% 
% %% plot trajectory of last realization
% figure(1)
% set(0, 'DefaultLineLineWidth', 2);
% plot(x(1,mode(:,end)==0),x(3,mode(:,end)==0),'ko')
% hold on
% plot(x(1,mode(:,end)==1),x(3,mode(:,end)==1),'ks','Linewidth',4)
% plot(z(1,:),z(2,:),'co')
% plot(x_hat_pred(1,:),x_hat_pred(3,:),'mo')
% plot(x(1,:),x(3,:),'k-','Linewidth',1)
% plot(x(1,1),x(3,1),'g*','Linewidth',8)
% hold off
% legend('true state (CV mode)','true state (CT mode)','noisy observation','genie KF prediction')
% axis equal
% grid on
% xlabel('x (meters)')
% ylabel('y (meters)')
% title(['target trajectory (duration: ' num2str(round(N2*T/60*10)/10) ' minutes)'])
% 
% %% plot turn rate and velocity of last realization
% figure(2)
% subplot(211)
% t=(0:N2-1)*T/60;
% plot(t,sqrt(x(2,:).^2+x(4,:).^2))
% grid on
% xlabel('time (minutes)')
% ylabel('speed (m/s)')
% title('airspeed (top) and turn rate (bottom)')
% subplot(212)
% plot(t,Om)
% grid on
% xlabel('time (minutes)')
% ylabel('turn rate (radians/s)')
% 
% %% plot MSE performance
% figure(3)
% plot(0:N2-2,squeeze(M_pred(1,1,2:end)+M_pred(3,3,2:end)),0:N2-2,mean(SqErr_GKF(2:end,:),2))
% xlabel('discrete time sample index')
% ylabel('mean squared error')
% grid on
% legend('theoretical genie KF MSE', 'experimental genie KF MSE')
% title(['sum of MSE on x and y coordinates over all ' num2str(numSims) ' runs'])
