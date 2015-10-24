function[] = mysgdsvm(filename, k, numruns)

%filename='MNIST-13.csv';
%k=5;
%numruns=5;

% Pegasos: Primal Estimated sub-gradient solver for SVM

% Inputs
% 1. S: Training set
% 2. Lambda: Regularization constant.
% 3. T: 10,000.
% 4. k: length of the mini-batch

% Take the whole training set

% Start Measuring time
data=csvread(filename);
data_s=data(:,2:end);
labels_s=data(:,1);
labels_s(labels_s==3)=-1;

% Set the Lambda=1
L=2;

% Set the maximum iterations T
T=10000;

% Divide the data in two zones. We have to pick equal percentages of data
% from each one of them.
dataset_plus1=data_s(labels_s==1,:);
labels_plus1=labels_s(labels_s==1,:);

dataset_minus1=data_s(labels_s==-1,:);
labels_minus1=labels_s(labels_s==-1,:);

data_percentage=k/length(labels_s);
no_data_for_1 = data_percentage*size(dataset_plus1,1);
no_data_for_m1 = data_percentage*size(dataset_minus1,1);

dump_matrix=cell(5,1);

% Primal objective
%figure;
%hold on;
%cc=hsv(numruns);
runtimes=zeros(numruns,1);
if k==1
    for iter=1:numruns
        tic;
        W=zeros(size(data,2)-1,1);
       
        for t=1:T

        % Choose k data-points at Random.
        At=randsample(size(data_s,1), k);

        % Find all the data-points in At that misclassify on the current W.
        
        if labels_s(At)*(data_s(At,:)*W)<1
            At_plus=At;
        else 
            At_plus=[];
        end
        
        % Calculate the step-size
        nt=1/(L*t);

        % Update the weight vectors.
        Wold=W;
        if isempty(At_plus)
            W=W*(1-nt*L);
        else
            W=W*(1-nt*L)+(nt/k)*(labels_s(At_plus)*data_s(At_plus,:))';
        end

        % Using projection method.
        W=min(1, 1/(L^(0.5)*norm(W)))*W;
        
        loss = 1 - (labels_s.*(data_s*W));
        loss = loss(loss > 0);
        primalobj(t) = L/2*(W'*W)+sum(loss)/k;
        
        if norm(Wold-W)<=0.1
            break;
        end
        
        end
        runtimes(iter)=toc;
        dumpmatrix{iter}=primalobj;
        %plot(primalobj, 'color', cc(iter,:));
    end

else
    
    for iter=1:numruns
        tic;
        % Initialize the W at the start of iteration to zero.
        W=zeros(size(data,2)-1,1);

        for t=1:T

            % Choose the required data-points from each set.
            At1=randsample(size(dataset_plus1, 1), ceil(no_data_for_1));
            minibatch_data_At1 = dataset_plus1(At1, :);
            minibatch_labels_At1 = labels_plus1(At1);
            % Find all the data-points in At1 that misclassify on the current W.
            At1_cross=minibatch_data_At1(minibatch_labels_At1.*(minibatch_data_At1*W)<1, :);

            At2=randsample(size(dataset_minus1, 1), floor(no_data_for_m1));
            minibatch_data_At2 = dataset_minus1(At2, :);
            minibatch_labels_At2 = labels_minus1(At2);
            % Find all the data-points in At2 that misclassify on the current W.
            At2_cross=minibatch_data_At2(minibatch_labels_At2.*(minibatch_data_At2*W)<1, :);

            At_plus=[At1_cross; At2_cross];
            labels_plus=[ones(size(At1_cross,1),1); -1*ones(size(At2_cross,1),1)];

            % Find all the data-points in At that misclassify on the current W.
            % At_plus=At(labels_s(At).*(data_s(At,:)*W)<1);

            % Calculate the step-size
            nt=1/(L*t);

            % Update the weight vectors.
            Wold=W;
            %W=W*(1-nt*L)+(nt/k)*sum(repmat(labels_s(At_plus),1,length(W)).*data_s(At_plus,:))';
            if isempty(At_plus)
                
                W=W*(1-nt*L);
            else
                W=W*(1-nt*L)+(nt/k)*sum(repmat(labels_plus,1,length(W)).*At_plus)';
            end
            % Using projection method.
            W=min(1, 1/(L^(0.5)*norm(W)))*W;

            %disp(sum(sign(data_s*W)~=labels_s));

            loss = 1 - (labels_s.*(data_s*W));
            loss = loss(loss > 0);
            primalobj(t) = L/2*(W'*W)+sum(loss)/k;

            if norm(Wold-W)<=0.1
                break;
            end
        end

        runtimes(iter)=toc;
        dumpmatrix{iter}=primalobj;
        %plot(primalobj, 'color', cc(iter,:));
    end
end
%hold off
% xlabel('number of iterations');
% ylabel('Objective Function');
fprintf('Average runtime for %d runs with minibatch size of %d is: %f secs \n\n', numruns, k, mean(runtimes));
fprintf('Std. runtime for %d runs with minibatch size of %d is: %f secs \n\n', numruns, k, std(runtimes));

csvwrite('.temp.txt', dumpmatrix)
disp('The plot data has been written to .temp.txt');

end