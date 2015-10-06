function[] = Fisher(filename, num_crossval)

% We are considering the MNIST-1378 dataset for 4-class classification of 
% hand written digits: 1,3,7 and 8. We have to train and evaluate the 
% Fisher’s linear discriminant in the general case followed by multi-variate 
% Gaussian generative modeling of each class in the projected space.

M = csvread(filename);
labels_ = M(:, 1);
data_ = M(:, 2:end);

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
    
    % Separate each class of data.
    data_1 = data(labels==1, :);
    data_3 = data(labels==3, :);
    data_7 = data(labels==7, :);
    data_8 = data(labels==8, :);

    % Get the mean of each class.
    m1 = mean(data_1, 1);
    m3 = mean(data_3, 1);
    m7 = mean(data_7, 1);
    m8 = mean(data_8, 1);

    % Total mean
    m = (m1+m3+m7+m8)/4;

    % Within-class scatter matrix

    % 2 ways to do this:
    % Sw1 = bsxfun(@minus, data_1, m1);
    % Sw3 = bsxfun(@minus, data_3, m3);
    % Sw7 = bsxfun(@minus, data_7, m7);
    % Sw8 = bsxfun(@minus, data_8, m8);

    Sw1 = data_1 - repmat(m1, length(data_1), 1);
    Sw3 = data_3 - repmat(m3, length(data_3), 1);
    Sw7 = data_7 - repmat(m7, length(data_7), 1);
    Sw8 = data_8 - repmat(m8, length(data_8), 1);
    Sw = Sw1'*Sw1 + Sw3'*Sw3 + Sw7'*Sw7 + Sw8'*Sw8;
    
    % Between-class scatter matrix
    Sb = ((m1-m)'*(m1-m) + (m3-m)'*(m3-m) + (m7-m)'*(m7-m) + ...
                    (m8-m)'*(m8-m))/4;

    % Solving the generalized eigenvalue problem for the matrix SW^(-1)*SB
    [Sw_inv_Sb_vector, Sw_inv_Sb_value] = eig(pinv(Sw)*Sb);

    % The top k-1 eigenvalues of this system is to be found now.
    [~, index] = sort(diag(Sw_inv_Sb_value), 'descend');

    % The top k-1 eigenvectors of this system are the following:
    eigenvector_1 = Sw_inv_Sb_vector(:, index(1));
    eigenvector_2 = Sw_inv_Sb_vector(:, index(2));
    eigenvector_3 = Sw_inv_Sb_vector(:, index(3));

    % The final transformatinal matrix is somewhat to be:
    W = [eigenvector_1,eigenvector_2,eigenvector_3];

    % Visualizing the workings of W on the training dataset.
    f_1 = data_1*W;
    f_3 = data_3*W;
    f_7 = data_7*W;
    f_8 = data_8*W;

    figure;
    scatter3(f_1(:,1), f_1(:,2), f_1(:,3), 'r', 'filled')
    hold on
    scatter3(f_3(:,1), f_3(:,2), f_3(:,3), 'g', 'filled')
    scatter3(f_7(:,1), f_7(:,2), f_7(:,3), 'b', 'filled')
    scatter3(f_8(:,1), f_8(:,2), f_8(:,3), 'm', 'filled')

    % Now we will be fitting the Gaussian on each of the 4 classes that we
    % see in the plot.(i.e f_1, f_3, f_7 and f_8)

    % Estimated Gaussian parameters for the dataset pertaining to digits 1.
    mean_1=mean(f_1);
    cov_1=cov(f_1);
    h1=plot_gaussian_ellipsoid(mean_1, cov_1);

    % Estimated Gaussian parameters for the dataset pertaining to digits 3.
    mean_3=mean(f_3);
    cov_3=cov(f_3);
    h3=plot_gaussian_ellipsoid(mean_3, cov_3);

    % Estimated Gaussian parameters for the dataset pertaining to digits 7.
    mean_7=mean(f_7);
    cov_7=cov(f_7);
    h7=plot_gaussian_ellipsoid(mean_7, cov_7);

    % Estimated Gaussian parameters for the dataset pertaining to digits 8.
    mean_8=mean(f_8);
    cov_8=cov(f_8);
    h8=plot_gaussian_ellipsoid(mean_8, cov_8);

    view(129,36); 
    set(gca,'proj','perspective'); 
    grid on; 
    axis equal; 
    axis tight;
    hold off

    % Now we have to pursue generative modelling here.
    % P(1/X) = P(X/1)P(1);
    % P(3/X) = P(X/3)P(3);
    % P(7/X) = P(X/7)P(7);
    % P(8/X) = P(X/8)P(8);
    
    %P(1), P(3), P(7) and P(8) is calculated here:
    data_length = length(data);
    P_1 = length(data_1)/data_length;
    P_3 = length(data_3)/data_length;
    P_7 = length(data_7)/data_length;
    P_8 = length(data_8)/data_length;
    
    % Find the error in the current test data.
    res_vec = [1, 3, 7, 8];
    [~, test_lab] = max([P_1 .* mvnpdf(test_data*W, mean_1, cov_1), ...
        P_3 .* mvnpdf(test_data*W, mean_3, cov_3), ...
        P_7 .* mvnpdf(test_data*W, mean_7, cov_7), ...
        P_8 .* mvnpdf(test_data*W, mean_8, cov_8)],[],2);
    test_error = sum(res_vec(test_lab) ~= test_labels')/length(test_labels);
    
    % Find the error in the current training set.
    [~, train_lab] = max([P_1 .* mvnpdf(data*W, mean_1, cov_1), ...
        P_3 .* mvnpdf(data*W, mean_3, cov_3), ...
        P_7 .* mvnpdf(data*W, mean_7, cov_7), ...
        P_8 .* mvnpdf(data*W, mean_8, cov_8)],[],2);
    train_error = sum(res_vec(train_lab) ~= labels')/length(labels);
    
    fprintf('case %d \n', iter);
    fprintf('test error: %f \n', test_error);
    fprintf('train error: %f \n\n', train_error);
    
    train_error_array(iter)=train_error;
    test_error_array(iter)=test_error;
end

fprintf('net training data error of FLDA : %f%% \n\n', 100*mean(train_error_array));
fprintf('standard deviation of the training data error of FLDA : %4.2f %%\n\n', std(train_error_array)*100/mean(train_error_array));
fprintf('net testing data error of FLDA : %f%% \n\n', 100*mean(test_error_array));
fprintf('standard deviation of the testing data error of FLDA : %4.2f %%\n\n', std(test_error_array)*100/mean(test_error_array));
end