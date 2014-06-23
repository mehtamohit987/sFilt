k_values = [1, 2, 10, 20, 50, 100];

trainFrac = .5;

RUNS_PER_K_VALUE = 10;
NUM_K_VALUES = length(k_values);
DIRNAME ='../Data/enron1';


testErrorMat     = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);

testFalsePosMat  = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);

for idx_k_value = 1:NUM_K_VALUES
    k = k_values(idx_k_value)
    for run=1:RUNS_PER_K_VALUE
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
                

        trainVectors = train(:,1:end-1);

        trainLabels = train(:,end);

        trainLabels = 2*trainLabels - 1;
        

        testVectors = test(:,1:end-1);

        testLabels = test(:,end);

        testLabels = 2*testLabels - 1;
        
        [testErrorMat(run,idx_k_value), ...
         testFalsePosMat(run,idx_k_value)] ...
            = knnClassify(trainVectors, trainLabels, testVectors, testLabels, k, 0.5);
                
    end
end




meanTestErrorMat = mean(testErrorMat, 1);
meanTestFalsePosMat= mean(testFalsePosMat, 1);

save knn_test_k_set5.mat meanTestErrorMat meanTestFalsePosMat;

h = figure; 
hold on;
plot(k_values,meanTestErrorMat, 'r-o');
plot(k_values,meanTestFalsePosMat, 'g-o');
xlabel('K value (# of neighbors)');
ylabel('Error rate');
legend('Test', 'false pos');
txt = sprintf('Training set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('results_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
