%%We use the glass dataset to test our one step KNN algorithm
load('glass.mat')
classnum = 6;
para.alpha = 10000;
para.beta = 1;
ind = crossvalind('Kfold',size(find(Y),1),10);
for k = 1:10
    testindex = ind(:) == k;
    trainindex = ~testindex;
    [time,label] = Onestepknnc(X(trainindex,:),Y(trainindex,:),X(testindex,:),classnum,para);
    predY{k}=label;
    Time(k) = time;
end
bb = [];
for tt = 1:10
    aa = predY{tt};
    bb = vertcat(bb,aa);
    aa = [];
end
pr_Y(:,1)= bb;
Acc = Accuracy( pr_Y(:,1),Y );
sumTime = sum(Time);
fprintf('The accuracy is %8.5f\n',Acc)
fprintf('The running cost is %8.5f\n',sumTime)