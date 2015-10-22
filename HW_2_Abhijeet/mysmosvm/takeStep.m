function [ true_or_false ] = takeStep( row_idx1, row_idx2 )

global data_40;
global labels_40;
global alpha;
global C;
global support_vector_indices;
global Err_cch;
global w x1 x2 y1 y2;
global b;
global alpha2 E2;
global iter;
global result

if row_idx1==row_idx2
    true_or_false=false;
    return;
end

% Find y1, alpha1 and E1
x1 = data_40(row_idx1, :);
y1 = labels_40(row_idx1);
alpha1 = alpha(row_idx1);


% Current w = sum(alpha.*target.*data_40)
%w = sum(repmat(alpha.*labels_40, 1, size(data_40,2)).*data_40);
E1 = x1*w' + b - y1;
s= y1*y2;

%% Compute L and H.
if y1==y2
    L = max([0 alpha1 + alpha2 - C]);
    H = min([C alpha1 + alpha2]);
else
    L = max([0 alpha2 - alpha1]);
    H = min([C C + alpha2 - alpha1]);
end

%% Compute the second derivative of the objective function (i.e eta)
eta = 2*(x1*x2') - (x1*x1') - (x2*x2');
%% compare eta, clip a2
if eta < 0
    a2 = alpha2-y2*(E1-E2)/eta;
    if a2<L
        a2=L;
    elseif a2>H
        a2=H;
    end
else
    % Find Hobj and Lobj at a2.
    c1 = eta/2;
    c2 = y2*(E1-E2)-eta*alpha2;
    Lobj=c1*L*L+c2*L;
    Hobj=c1*H*H+c2*H;
    
    if Lobj > Hobj+0.001
        a2=L;
    elseif Lobj<Hobj-0.001
        a2=H;
    else
        a2=alpha2;
    end
end

if a2<1e-8
    a2=0;
elseif a2>C-1e-8
    a2=C;
end

e=1e-3;
if abs(a2-alpha2) < e*(a2+alpha2+e)
    true_or_false=false;
    return;
end

a1 = alpha1 + s*(alpha2 - a2);
%% Update threshold to reflect change in Langrange multipliers
b1 = b + E1 + y1*(a1-alpha1)*(x1*x1') + y2*(a2-alpha2)*(x1*x2');
b2 = b + E2 + y1*(a1-alpha1)*(x1*x2') + y2*(a2-alpha2)*(x2*x2');

% Compute b
if ((0 < a1) && (a1 < C))
    b_new = b1;
elseif ((0 < a2) && (a2 < C))
    b_new = b2;
else
    b_new = (b1+b2)/2;
end
delta_b=b_new-b;
b=b_new;


%% Update weight vector to reflect change in a1 and a2, if Linear SVM

%% Update error cache using new Langrange Multipliers
for j=1:length(support_vector_indices)
    if alpha(j)>0 && alpha(j)<C
        Err_cch(j)=Err_cch(j) + y1*(a1-alpha1)*(data_40(j, :)*x1') + y2*(a2-alpha2)*(data_40(j, :)*x2')-delta_b;
    end
end

Err_cch(row_idx1)=0;
Err_cch(row_idx2)=0;

alpha(row_idx1)=a1;
alpha(row_idx2)=a2;


true_or_false=true;

%% Find the dual objective function
iter=iter+1;
result(iter) = sum(alpha)-0.5*w*w';
end

