tic;
% Input
% C
% tol
% b
% alphas

% Initialize the data. I am using only 40 points for now.

% Take only 40 data-points as the training set
%data_40=data(1:40,2:end);
%labels_40=data(1:40,1);

% Take the whole training set

data_40=data(1:300,2:end);
labels_40=data(1:300,1);
labels_40(labels_40==3)=-1;

% Initialize alpha array to all zero
alpha=zeros(size(labels_40));

% Initialize b to zero
b=0;

% Initialize the tolerance
tol=0.001;

% Initialize the penalty term
C=1;

max_passes = 1;
passes = 0;

while passes<max_passes
    num_changed_alphas=0;

    % Repeat until KKT is satisfied(within tolerance)
    for i=1:length(labels_40)

        % To find if the current i(if non-bound) violates the KKT or not
        % 
        % The condition of KKT for non-bound cases satisfies:
        %  0 < alpha_i < C ==> labels(i) * f_SVM(data_i)==1
        % The implied term can also be written as:
        % labels(i) * f_SVM(data_i) == labels(i)^2
        % or labels(i)*(f_SVM(data_i)-labels(i)) == 0
        % or labels(i)*(E_i) == 0
        % Here E_i is the error on the current data.
        % SInce we have assumed a tolerance term of 0.001, the above can be
        % restated as:
        %  -tolerance < labels(i)*(E_i) < tolerance
        % SUMMARY: If the above inequality is not followed by any non-bound i,
        % then we can say that it is violating the required KKT conditions.

        %% Find a non-bound example that violates KKT 

        % Error of the sample i
        % Ei = f_SVM(xi)-yi
        % f_SVM(xi) = w'xi+b
        % Current w = sum(alpha.*target.*data_40)
        w = sum(repmat(alpha.*labels_40, 1, size(data_40,2)).*data_40);
        % I have assumed linear kernel function for now.
        % Error on the current data.
        
        E1 = data_40(i,:)*w' + b - labels_40(i);
        %E1 = data_40(i,:)*w' - labels_40(i);

        % Proceed only for non-bound data-points that violate KKT.
        curr_label = labels_40(i);
        curr_error = E1;
        if ((curr_error*curr_label < -tol) && alpha(i)<C) || ((curr_error*curr_label > tol) && (alpha(i)>0))

            % We have found a data_point that violates KKT and is non-bound.
            % Now we just have to find the another point

            % Search over all j
            for j=1:length(labels_40)

                % We don't want i and j to be the same.
                % There is more to the 2nd choice Heuristics. 

                % 1. SMO keeps a cached error value E for every non-bound
                % example in the training set and then chooses an error to
                % approximately maximize the step size |E2-E1|

                % 2. If the above heuristic doesnot make positive progress,
                % then SMO starts iterating through the non-bound examples,
                % searching for a second example that can make positive
                % progress. Try to make this iteration for heuristics random.

                % 3. If none of the non-bound examples make positive progress,
                % then SMO starts iterating through the entire training set
                % untill an example is found that makes positive progress.
                % Again try to make this iteration for heuristics random.

                % This randomness is needed to ensure that SMO doesn't get
                % biased towards the examples at the beginning of the training
                % set.

                % 4. In an extreamly degenerate case, when none of the examples
                % make an adequate second example, we skip the first example
                % and continue the SMO with another choosen first example.

                % I will do that once the following works.
                if j~=i

                     x1 = data_40(i,:);
                     x2 = data_40(j,:);
                     y1 = labels_40(i);
                     y2 = labels_40(j);

                    % Current w = sum(alpha.*target.*data_40)
                    w = sum(repmat(alpha.*labels_40, 1, size(data_40,2)).*data_40);

                    % Error on the current data.
                    
                    E2 = x2*w' + b - y2;
                    %E2 = x2*w' - y2;

                    alpha1_old=alpha(i);
                    alpha2_old=alpha(j);

                    % Compute L and H
                    if y1 == y2;
                        L = max([0 alpha1_old + alpha2_old - C]);
                        H = min([C alpha1_old + alpha2_old]);
                    else
                        L = max([0 alpha2_old - alpha1_old]);
                        H = min([C C + alpha2_old - alpha1_old]);
                    end

                    % Compute the second derivative(eta) of the objective 
                    % function for a linear kernel.
                    eta = 2*(x1*x2') - (x1*x1') - (x2*x2');

                     % We want the second derivative to be less than 0.
                     % If it is not, then it is not a maximum.
                     if eta >=0
                         continue;
                     end

                     % Compute the alpha2_new.
                     alpha2_new = alpha2_old - y2*(E1-E2)/eta;

                     % Clip alpha2_new to the ends of the line segment.
                     if alpha2_new > H
                         alpha2_new = H;
                     elseif alpha2_new < L
                         alpha2_new = L;
                     end

                     % update alpha_2.
                     alpha(j) = alpha2_new;

                     % If alpha2_new is not much different than alpha2_old,
                     % then there is no need to adjust alpha1_old since they
                     % are almost equal only.
                     if abs(alpha2_new-alpha2_old) < tol
                         continue;
                     end

                     % update alpha_1
                     s = y1*y2;
                     alpha1_new = alpha1_old + s*(alpha2_old-alpha2_new);
                     alpha(i) = alpha1_new;
                     
                     
                     % Solving for the Langrange multipliers alpha does not
                     % determine the threshold b of SVM, so b must be computed
                     % separately. After each step, b is re-computed so that
                     % the KKT conditions are fulfilled for both optimized
                     % examples.

                     b1 = b + E1 + y1*(alpha1_new-alpha1_old)*(x1*x1') + y2*(alpha2_new-alpha2_old)*(x1*x2');
                     b2 = b + E2 + y1*(alpha1_new-alpha1_old)*(x1*x2') + y2*(alpha2_new-alpha2_old)*(x2*x2');

                     % Compute b
                     if ((0 < alpha1_new) && (alpha1_new < C))
                         b = b1;
                     elseif ((0 < alpha2_new) && (alpha2_new < C))
                         b = b2;
                     else
                         b = (b1+b2)/2;
                     end

                     num_changed_alphas = num_changed_alphas+1;
                     
                end
                
            end

        end
    end
    if num_changed_alphas==0
       passes = passes+1;
    else
       passes = 0;
    end
   
    
end
toc;
% Calculate the final W
totalSum = 0;
for i=1:length(labels_40)
    totalSum = totalSum + alpha(i)*labels_40(i)*data_40(i,:);
end

W = totalSum;
b = labels_40(1) - data_40(1,:)*W';

disp('--------------------------------')
disp('----------- Results: -----------')
disp('--------------------------------')
% alpha
% W
% b

% Testing
for i=1:length(labels_40)
res(i) = data_40(i,:)*W'+b;
end
disp(sum(labels_40==sign(res'))/length(labels_40))