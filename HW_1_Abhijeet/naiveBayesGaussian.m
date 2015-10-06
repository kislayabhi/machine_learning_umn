function [] = naiveBayesGaussian(filename, num_splits, train_percent)


%% Naive Bayes.
X = csvread(filename);
rep_no = num_splits;

% from the rest data, we have to take that percentage for training that is
% specified by the vector given by the user. (say N)
N=train_percent;

% The result matrix is for storing all the error rates in them. 
result_matrix = ones(rep_no, length(N));


% Run for num_splits iterations.
for repetation=1:rep_no

    % partition 20% of spam and non-spam data and remove it from X.
    testing_data_numbers_0 = randsample(find(X(:,1)==0), round(0.2*sum(X(:,1)==0)));
    testing_data_numbers_1 = randsample(find(X(:,1)==1), round(0.2*sum(X(:,1)==1)));
    testing_data_numbers=[testing_data_numbers_0; testing_data_numbers_1];
    random_test_rows=testing_data_numbers(randperm(length(testing_data_numbers_0)+length(testing_data_numbers_1)));
    testing_data = X(random_test_rows, :);
    X(random_test_rows, :) = [];
    
    % separate the 1st column as the labels from the testing data.
    testing_labels = testing_data(:,1);
    % remove those labels from the original data.
    testing_data(:,1) = [];
    
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
        training_data(:,1) = [];

        % Separate the label-1 data, their mean and variances for each
        % dimensions.
        label_1_row=find(training_labels==1);
        label_0_row=find(training_labels==0);

        label_1_data = training_data(label_1_row,:);
        label_1_mean = mean(label_1_data);
        % Sometimes, it may happen that the training set we have taken has
        % all the feature values of a dimension 0. In those cases, it is
        % not possible to fit a gaussian. I a just adding certain
        % perturbation here to be vary of such outliers.
        label_1_variance = diag(cov(label_1_data))+0.0001;

        % Similarly separate the label-0 data, their mean and variances for
        % each dimensions.
        label_0_data = training_data(label_0_row,:);
        label_0_mean = mean(label_0_data);
        % Similar explaination here for the explicit addition of 0.0001
        label_0_variance = diag(cov(label_0_data))+0.0001;
        
        % So now we have Essentially 57 means and 57 variances for 
        % the label_0 data in the training set. Similarly for the label_1
        % dataset.
        pred = ones(length(testing_labels),1);
        for x=1:length(testing_labels)
            first_t_eg = testing_data(x,:);
            p_0 = length(label_0_row)/length(training_labels);
            % Fit the univariate gaussian for the 57 dimensions for the
            % label_0 data. find the probability of the current point lying
            % in it.
            % Finding the probabilty is a general multiplication of all the
            % probabilities of the 57 dimensions with p(label_0). This is
            % exactly the Bayes Theorem.
            for i=1:size(testing_data,2)
                p_0 = p_0 * mvnpdf(first_t_eg(i), label_0_mean(i), label_0_variance(i));
            end
            % Do the same thing for label_1 data
            p_1 = length(label_1_row)/length(training_labels);
            for i=1:size(testing_data,2)
                p_1 = p_1 * mvnpdf(first_t_eg(i), label_1_mean(i), label_1_variance(i));
            end
            % The class having the higher value of the posterior
            % probability is the winner.Assign the test data to thal class.
            if p_0>p_1
                pred(x)=0;
            else
                pred(x)=1;
            end
        end
        % Record the misclassification error rate by comparing it to the
        % original labels.
        testing_error = sum(pred~=testing_labels)/length(testing_labels);
        fprintf('Testing error on %d %% is %f \n\n', N(n), testing_error);
        % Then we will get the required things inside the result matrix.
        result_matrix(repetation, n) = testing_error;
    end
end
% display the mean and the standard deviation of the errors.
disp('mean:')
disp(mean(result_matrix));
disp('standard_deviation')
disp(std(result_matrix));