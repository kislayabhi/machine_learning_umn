function[] = SqClass(filename, num_crossval)

% We are consider the MNIST-1378 dataset for 4-class classification of 
% hand written digits: 1,3,7 and 8. We have to train and evaluate the 
% linear discriminant for the least squares.

M = csvread(filename);
labels_ = M(:, 1);
% New
M(:, 1)=1;
data_ = M;
%data_ = M(:, 2:end);

% Set the num_crossval value for the k-fold cross-validation.
k=num_crossval;

% For the 10-fold cross validation, the original sample is randomly
% partitioned into 10 equal sized subsamples. Of the 10 subsamples, a
% single subsample is retained as the validation data for testing the
% model, and the remaining 9 subsamples are used as training data.
% The cross-validation process is then repeated 10 times, with each of the
% 10 subsamples used exactly once as the validation data.
% The 10 results from the folds can then be averaged to produce a single
% estimation.
random_perm = randperm(length(M));
intervals = round(linspace(1,length(M), k+1));

% For storing the error on testing data for each fold.
test_error_array=zeros(1,k);

% For storing the error on training data for each fold.
train_error_array=zeros(1,k);

for iter = 1:k
    
    data = data_;
    labels = labels_;
    % Test data
    test_data = data(random_perm(intervals(iter):intervals(iter+1)), :);
    test_labels = labels(random_perm(intervals(iter):intervals(iter+1)), 1);
    
    % Training_set data
    data(random_perm(intervals(iter):intervals(iter+1)), :) = [];
    labels(random_perm(intervals(iter):intervals(iter+1)), :) = [];
    
    % We will create a target vector for each of the 4 labels that we have.
    % for 1, we will set t = [1,0,0,0]
    % for 3, we will set t = [0,1,0,0]
    % for 7, we will set t = [0,0,1,0]
    % for 8, we will set t = [0,0,0,1]
    
    % What we do below can be termed as training the least square
    % classifier. It is basically just finding the correct design matrix.
    T = zeros(length(labels), 4);
    T(labels==1, 1) = 1;
    T(labels==3, 2) = 1;
    T(labels==7, 3) = 1;
    T(labels==8, 4) = 1;
    
    % Find the pinv(X)
    pinv_x = pinv(data);
    % pinv(X)*T = Design matrix W.
    W = pinv_x*T;
    
    % The training is completed. Now we test both our training
    % set data and the test set data.
    
    % Training error-rate.
    training_probability_result = data*W;
    [~, training_labels_result] = max(training_probability_result, [], 2);
    [~, actual_training_labels_result] = max(T, [], 2);
    training_error = sum(actual_training_labels_result~=training_labels_result)/length(T);
    
    fprintf('error on training dataset for case %d: %f \n', iter, training_error);
    
    % Error rate on the current testing sample.
    test_T = zeros(length(test_labels), 4);
    test_T(test_labels==1, 1) = 1;
    test_T(test_labels==3, 2) = 1;
    test_T(test_labels==7, 3) = 1;
    test_T(test_labels==8, 4) = 1;
    
    testing_probability_result = test_data*W;
    [~, testing_labels_result] = max(testing_probability_result, [], 2);
    [~, actual_testing_labels_result] = max(test_T, [], 2);
    testing_error = sum(actual_testing_labels_result~=testing_labels_result)/length(test_T);
    
    fprintf('error on test dataset for case %d: %f \n\n', iter, testing_error);
    
    train_error_array(iter)=training_error;
    test_error_array(iter)=testing_error;
end

fprintf('net training data error of Least Squares : %f%% \n\n', 100*mean(train_error_array));
fprintf('standard deviation of the training data error of Least Squares : %4.2f%%\n\n', std(train_error_array)*100/mean(train_error_array));
fprintf('net testing data error of Least Squares : %f%% \n\n', 100*mean(test_error_array));
fprintf('standard deviation of the testing data error of Least Squares : %4.2f%%\n\n', std(test_error_array)*100/mean(test_error_array));
end