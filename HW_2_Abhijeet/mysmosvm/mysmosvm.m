function[] = mysmosvm(filename, numruns)

% filename='MNIST-13.csv';
% numruns=1;

% Read the data.
data=csvread(filename);

% Variable for reading the data in.
timing_data=zeros(length(numruns),1);

% Result temp data.
temp_result_data = cell(numruns, 1);

% This is the code for SMO also known as the Sequential Minimal
% Optimization. 
% Reference: "Fast Training of Support Vector Machines using Sequential 
%             Minimal Optimization" by J.C.Platt.

%% Main Routine

%% Initializations
% Initialize the data. I am using only 40 points for now.
global data_40;
global labels_40;
global alpha;
global tol;
global C;
global support_vector_indices;
global Err_cch;
global b;
global iter;
global result;


data_40=data(1:100,2:end);
labels_40=data(1:100,1);
labels_40(labels_40==3)=-1;

for run=1:numruns
    
    tic;
    
    iter=0;


    % Initialize alpha array to all zero
    alpha=zeros(size(labels_40));

    % Initialize b to zero
    b=0;

    % numChanged stores how many alphas are changing in each while loop.
    numChanged = 0;

    % examineAll = True or False.
    examineAll = true;

    % Initialize the tolerance
    tol=0.001;

    % Initialize the penalty term
    C=10;

    % We keep on repeating the below while loop until
    % 1. examineAll==false
    % 2. and numChanged==0.

    % Error cache. It has -1 for the alphas whose value is equal to C or 0.
    Err_cch=-1*ones(size(alpha));

    %% The alternating outer loop
    while(numChanged>0 || examineAll==true)

        numChanged=0;

        % Find all the current support vector.
        support_vector_indices=find(alpha~=0 & alpha~=C);


        % If all the data-points are to be examined.
        if (examineAll)
            % Loop over all training examples.
            for i=1:length(labels_40)
                % Remember, we are just giving the row number of the data.
                numChanged = numChanged + examineExample(i);
            end
        % else we are interested in a subset only (non bound areas.)
        else
            % Loop over examples where alpha is not 0 and not C
            for i=1:length(support_vector_indices)
                % Remember, we are just giving the row number of the data.
                numChanged = numChanged + examineExample(support_vector_indices(i));
            end
        end
        % After the above step, we expect the numChanged to increase.

        % If in this iteration(while), we had examined all the data-points.
        % Then in the next iteration, we are only interested in a subset only.
        % So no need to examineALL!
        if (examineAll==true)
            examineAll=false;
        % If in this iteration, even after running 'examineExample' ... we are
        % not able to increase numChanged then that means we might have
        % exhausted all the non-bound areas. The time has come to again iterate
        % over all the training data points.
        elseif (numChanged ==0)
            examineAll=true;
        end
    end

    %f=alpha;

    temp_result_data{run}=result;
    plot(result,'b--*')
    hold on;

    timing_data(run)=toc;
end
xlabel('Increasing interation of valid optimization over the two alphas') % x-axis label
ylabel('The dual objective') % y-axis label

hold off;

disp('The mean of time taken over numruns time(in secs):');
disp(mean(timing_data));

disp('The standard deviation for the above mean(in secs):');
disp(std(timing_data));

csvwrite('.temp.txt', temp_result_data)
disp('The plot data has been written to .temp.txt');
end