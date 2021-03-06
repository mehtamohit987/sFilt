k_values = [1, 2, 10, 20, 50, 100];

supermaj_factor_values = [.5, .6, .7, .8, .9];

dim = 40;

trainFrac = .5;

RUNS_PER_K_VALUE = 10;
NUM_K_VALUES = length(k_values);
NUM_FACTOR_VALUES = length(supermaj_factor_values);

load('knn_test_majority_set1.mat');
X= meanTestErrorMat;
Y= meanTestFalsePosMat;

load('knn_test_majority_set2.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_majority_set3.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_majority_set4.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_majority_set5.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

meanTestErrorMat=X./5;
meanTestFalsePosMat=Y./5;

save merged_knn_majority.mat meanTestErrorMat meanTestFalsePosMat;

colors = 'brgcm';
factor_values_temp = cellstr(num2str(supermaj_factor_values'))';
factor_values_str = arrayfun(@(x)strcat('factor=', x), factor_values_temp);

h = figure;
hold on;

for idk_factor_value = 1:NUM_FACTOR_VALUES
    plot(k_values, meanTestErrorMat(idk_factor_value, :), strcat(colors(idk_factor_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('Error rate');
legend(factor_values_str);
txt = sprintf('Error Rate\nDim=%d\nTraining set size is %g of all data\nAverage of %d runs per K value', dim, trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('merge_knn_majority_FINAL_error_rate_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);

h = figure;
hold on;

for idk_factor_value = 1:NUM_FACTOR_VALUES
    plot(k_values, meanTestFalsePosMat(idk_factor_value, :), strcat(colors(idk_factor_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('False pos');
legend(factor_values_str);
txt = sprintf('False Positive Ratio\nDim=%d\nTraining set size is %g of all data\nAverage of %d runs per K value', dim, trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('merge_knn_majority_FINAL_false_pos_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
