function [time,Label] = Onestepknnc(traindata,Ytrain,testdata,classnum,para)
%% one step Knn
%Ytrain represents the label of training data
%Classnum represents the number of class labels of data
X = traindata;   %Training data
Y = testdata;     %test data
alpha = para.alpha;  %parameter alpha
beta = para.beta;     %parameter beta
[n,d] = size(X);
[m,~] = size(Y);


classnumber = 2;  % Number of groups
%% Group by fuzzy c-means clustering
index = zeros(classnumber,n);
[~,U,~] = fcm(X,classnumber); % U is the membership matrix
IGg = [];
for ii = 1:classnumber
    indexx = [];
    indexx =  find(U(ii,:) > 0.01);
    index(ii,1:length(indexx)) = indexx;  
    Ig = zeros(1,n);
    Ig(sub2ind(size(Ig), indexx)) = 1;
    IGg(ii,:) = Ig; %Constructing IGg matrix
end

%% initialization
W = rand(n,m);
iter = 1;
obji = 1;
Wsum = sum(W,2);
while 1
    clear FF;
    for i = 1:n
        Groupsum = 0;
        for ii = 1:classnumber
            indexx = [];
            indexx =  find(U(ii,:) > 0.01);
            group = [Wsum(indexx)];
            Groupsum = Groupsum+(IGg(ii,i)*norm(group,1))/norm(W(i,:),1);
        end
        FF(i) = Groupsum;
    end
    F = diag(FF);
    
    
    %%  Çó³öW
    for i =1:n
        dn(i) = sqrt(sum((sum(W.*W,2)+eps)))./sum(W(i,:));
    end
    N = diag(dn);
    W = (X*X'+alpha*N+beta*F)\(X*Y');
    W(W<mean(W)) = 0;
    Wi = sqrt(sum(W.*W,2)+eps);
    W21 = sum(Wi);
    Wd = sum(W,2);
    obj(iter) =  norm(Y' - X'*W, 'fro')^2  +  alpha * W21+ beta * Wd'*F*Wd;
    cver = abs((obj(iter)-obji)/obji);
    obji = obj(iter);
    iter = iter + 1;
    if (cver < eps && iter > 2) || iter == 2,    break,     end
end
tic;
for ii = 1:m
    idx = find(W(:,ii) ~= 0);
    for jj = 1:classnum
        if ~isempty(Ytrain(idx) == jj)
            
            idxnum = find(Ytrain(idx) == jj);
            weight(jj) = sum(W(idxnum,ii));
        end
    end
    if isnan(weight)       
        tbl = tabulate(Ytrain);
        [C,id] = find(max(tbl(:,3)));
        idxmax = tbl(id,1);
        Labels(ii) = idxmax;
    end
    if ~isnan(weight)
        idxmax = find(weight == max(weight));
        if size(idxmax,2)>1
            Labels(ii) = idxmax(1);
        end
        if size(idxmax,2) == 1
            Labels(ii) = idxmax;
        end
    end   
end
Label = Labels';
time = toc;
end
