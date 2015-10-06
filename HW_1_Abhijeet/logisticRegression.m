function [] = logisticRegression(filename, num_splits, train_percent)


%% Logistic_regression.
%data = csvread('data/spam.csv');
%rep_no = 10;
%N = [5, 10, 15, 20, 25, 30];
data=csvread(filename);
rep_no=num_splits;
N=train_percent;
result_matrix = ones(rep_no, length(N));

% Centering the data for gradient descent.
data(:,2:end) = data(:,2:end)-repmat(mean(data(:,2:end), 1),size(data,1),1);

for repetation=1:rep_no
    X=data;
    
    % partition 20% of spam and non-spam data and remove it from X.
    testing_data_numbers_0 = randsample(find(X(:,1)==0), round(0.2*sum(X(:,1)==0)));
    testing_data_numbers_1 = randsample(find(X(:,1)==1), round(0.2*sum(X(:,1)==1)));
    testing_data_numbers=[testing_data_numbers_0; testing_data_numbers_1];
    random_test_rows=testing_data_numbers(randperm(length(testing_data_numbers_0)+length(testing_data_numbers_1)));
    testing_data = X(random_test_rows, :);
    X(random_test_rows, :) = [];
    
    % separate the 1st column as the labels from the testing data.
    testing_labels = testing_data(:,1);
    
    % Augment the x_0 = 1
    testing_data(:,1) = 1;
  
    % For each split, run for each training set data percentage.
    for n=1:length(N)

        % Randomly partition N(n)% of data from each class of X for training
        training_data_numbers_0 = randsample(find(X(:,1)==0), round(0.01*N(n)*sum(X(:,1)==0)));
        training_data_numbers_1 = randsample(find(X(:,1)==1), round(0.01*N(n)*sum(X(:,1)==1)));
        training_data_numbers = [training_data_numbers_0; training_data_numbers_1];
        random_train_rows=training_data_numbers(randperm(length(training_data_numbers_0)+length(training_data_numbers_1)));
        training_data = X(random_train_rows, :);

        %% training the classifier
        training_labels = training_data(:,1);
        training_data(:,1) = 1;

        % This is the design vector W. (calling params)
        params = ones(size(training_data, 2), 1);

        % I am using Gradient Descent method here.
        for i=1:900
            for j=1:length(training_labels)
                result=training_data(j,:)*params;
                % Below is the sigmoid function.
                sig_f=1./(1+exp(-result));
                % Finding misclassified points
                grad=sig_f-training_labels(j);
                % I have taken the learning rate to be 0.1
                params=params-(grad*training_data(j,:))'./10;
            end

            train_final_result=training_data*params;
            train_final_f=1./(1+exp(-train_final_result));
            train_final_f(train_final_f>=0.5)=1;
            train_final_f(train_final_f<0.5)=0;
            % Break from the loop is the training set error goes below 0.01
            if (sum(train_final_f==training_labels)/length(training_labels) < 0.01)
                break;
            end
        end
        %fprintf('Training error on %d is %f \n\n', N(n), sum(train_final_f~=training_labels)/length(training_labels));

        %testing the classifier on the 20% separated data
        test_final_result=testing_data*params;
        test_final_f=1./(1+exp(-test_final_result));
        test_final_f(test_final_f>=0.5)=1;
        test_final_f(test_final_f<0.5)=0;
        testing_error=sum(test_final_f~=testing_labels)/length(testing_labels);
        fprintf('Testing error on %d%% is %f \n\n', N(n), testing_error);
        
        % Then we will get the required things inside the result matrix.
        result_matrix(repetation, n) = testing_error;
    end
end
% display the mean and the standard deviation of the errors.
disp('mean:')
disp(mean(result_matrix));
disp('standard_deviation')
disp(std(result_matrix));