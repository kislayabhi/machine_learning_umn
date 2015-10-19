% Pegasos: Primal Estimated sub-gradient solver for SVM

% Inputs
% 1. S: Training set
% 2. Lambda: Regularization constant.
% 3. T: 10,000.
% 4. k: length of the mini-batch

% Take the whole training set

% Start Measuring time
tic;

data_s=data(:,2:end);
labels_s=data(:,1);
labels_s(labels_s==3)=-1;

% Set the Lambda=1
L=1;

% Set the maximum iterations T
T=10000;

% Set the length of the Mini-batch algorithm
k=15;

% Initialize the W at the start of iteration to zero.
W=zeros(size(data,2)-1,1);

for t=1:T
    
    % Choose k data-points at Random.
    At=randsample(size(data_s,1), k);
    
    % Find all the data-points in At that misclassify on the current W.
    At_plus=At(labels_s(At).*(data_s(At,:)*W)<1);
    
    % Calculate the step-size
    nt=1/(L*t);
    
    % Update the weight vectors.
    Wold=W;
    W=W*(1-nt*L)+(nt/k)*sum(repmat(labels_s(At_plus),1,length(W)).*data_s(At_plus,:))';
    
    % Using projection method.
    W=min(1, 1/(L^(0.5)*norm(W)))*W;
   
    
    if norm(Wold-W)<=0.1
        break;
    end
end

toc;
