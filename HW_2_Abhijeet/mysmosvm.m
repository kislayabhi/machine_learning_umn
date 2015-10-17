% This is the code for SMO also known as the Sequential Minimal
% Optimization. 
% Reference: "Fast Training of Support Vector Machines using Sequential 
%             Minimal Optimization" by J.C.Platt.

%% Main Routine

% Initialize the data. I am using only 40 points for now.
data_40=data(1:40,2:end);
labels_40=data(1:40,1);

% Initialize alpha array to all zero
alpha=zeros(size(labels_40));

% Initialize b to zero
b=0;

% numChanged stores how many alphas we have taken care of till now.
numChanged = 0;

% examineAll = 1 or 0. It tells whether to examine all the training egs in
% the current iteration or not. If we are not examining all the training
% examples then the loop 1 of the SVM iterates over the non-bound alphas
% only.
examineAll = 1;

% We keep on repeating the below while loop until
% 1. examineAll==0
% 2. and numChanged==0.
% Except for the first iteration where numChanged==0, it will be equal to
% zero next only when both the 'examineExample' fail to change the value of
% numChanged. And untill and unless we have not examined all the training
% points, we can't stop.
while(numChanged>0 || examineAll)
    % set numChanged=1; 
    numChanged=0;
    
    % If all the data-points are to be examined
    if (examineAll)
        % Loop over all training examples.
        for i=1:length(labels_40)
        numChanged = numChanged + examineExample(i);
        end
    % else we are interested in a subset only (non bound areas.)
    else
        % Loop over examples where alpha is not 0 and not C
        % ---------------------------------------------------
        % Todo : Write the looping function that does that! |
        % ---------------------------------------------------
        numChanged = numChanged + examineExample();
    end
    % After the above step, we expect the numChanged to increase.
    
    % If in this iteration(while), we had examined all the data-points.
    % Then in the next iteration, we are only interested in a subset only.
    % So no need to examineALL!
    if (examineAll==1)
        examineAll=0;
    % If in this iteration, even after running 'examineExample' ... we are
    % not able to increase numChanged then that means we might have
    % exhausted all the non-bound areas. The time has come to again iterate
    % over all the training data points.
    else if (numChanged ==0)
        examineAll=1;
        end
    end
end

