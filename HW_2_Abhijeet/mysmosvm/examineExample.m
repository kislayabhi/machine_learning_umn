function [ one_if_changed ] = examineExample( row_idx2 )

global data_40;
global labels_40;
global alpha;
global tol;
global C;
global support_vector_indices;
global Err_cch;
global w x1 x2 y1 y2;
global b;
global alpha2 E2;

%% Choose the 1st point.
% Choosing the first point is based on violation of the KKT conditions.

% Current Data
x2 = data_40(row_idx2, :);
% Current Label
y2 = labels_40(row_idx2);
% The alpha contributed by alpha2
alpha2 = alpha(row_idx2);

% Current w = sum(alpha.*target.*data_40)
w = sum(repmat(alpha.*labels_40, 1, size(data_40,2)).*data_40);

E2 = x2*w' + b - y2;

% Find a non-bound example that violates the KKT.
r2=E2*y2;

%% Choose the 2nd point
if((r2 < -tol) && (alpha2<C)) || ((r2 > tol) && (alpha2>0))
    %% Heuristic 1
    if(length(support_vector_indices) > 1)
        
        % Find the row number in Err_cch which is not -1 and for which the
        % value of |E2-E1| is maximized
        E_abs = zeros(size(Err_cch));
        E_abs(Err_cch ~= -1) =  abs(Err_cch(Err_cch ~= -1)-E2);
        
        % row_1_idx is the result of the 2nd choice heuristics.
        [~, row_idx1]=max(E_abs);
        
        % Now we have got both row_idx1 and row_idx2. See if we are able to
        % change alpha's via this step!
        if takeStep(row_idx1, row_idx2)==true
            one_if_changed=1;
            return;
        end
    end
    
    %% Heuristic 2
    % Loop over all non-bound data-points. Keep starting point random.
    random_support_vector_indices=support_vector_indices(randperm(length(support_vector_indices)));
    for j=1:length(random_support_vector_indices)
        row_idx1=random_support_vector_indices(j);
        if takeStep(row_idx1, row_idx2)==true
            one_if_changed=1;
            return;
        end
    end
    
    %% Heuristic 3
    % Loop over all possible data_points, starting at a random point.
    random_labels=randperm(length(labels_40));
    for j=1:length(random_labels)
        row_idx1=random_labels(j);
        if takeStep(row_idx1, row_idx2)==true
            one_if_changed=1;
            return;
        end
    end
    
    
end
one_if_changed=0;
end

