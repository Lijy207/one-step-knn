function acc  = Accuracy( y_pre, y_true )
n = length(y_true);
count = 0;
for i = 1:n
    if y_true(i) == y_pre(i)
        count = count + 1;
end
end
acc = count/n;
