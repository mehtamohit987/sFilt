
k_values = [1, 2, 10, 20, 50, 100];


dim_values = [20, 40, 70, 100];

trainFrac = .5;
RUNS_PER_K_VALUE = 10;
NUM_K_VALUES = length(k_values);
NUM_DIM_VALUES = length(dim_values);

load('knn_test_dim_set1.mat');
X= meanTestErrorMat;
Y= meanTestFalsePosMat;

load('knn_test_dim_set2.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_dim_set3.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_dim_set4.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_test_dim_set5.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

meanTestErrorMat=X./5;
meanTestFalsePosMat=Y./5;

colors = 'brgc';
dim_values_temp = cellstr(int2str(dim_values'))';
dim_values_str = arrayfun(@(x)strcat('dim=', x), dim_values_temp);

h = figure;
hold on;

for idk_dim_value = 1:NUM_DIM_VALUES
    plot(k_values, meanTestErrorMat(idk_dim_value, :), strcat(colors(idk_dim_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('Error rate');
legend(dim_values_str);
txt = sprintf('Error Rate\nTraining set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('merge_knn_dim_FINAL_error_rate_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);


h = figure;
hold on;

for idk_dim_value = 1:NUM_DIM_VALUES
    plot(k_values, meanTestFalsePosMat(idk_dim_value, :), strcat(colors(idk_dim_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('False pos');
legend(dim_values_str);
txt = sprintf('False Positive Ratio\nTraining set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('merge_knn_dim_FINAL_false_pos_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
