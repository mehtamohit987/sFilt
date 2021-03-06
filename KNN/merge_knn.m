k=1;

TRAIN_FRACS = .1:.1:.9;

RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);

load('knn_set1.mat');
X= meanTestErrorMat;
Y= meanTestFalsePosMat;

load('knn_set2.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

load('knn_set3.mat');
X= X+meanTestErrorMat;
Y= Y+meanTestFalsePosMat;

%load('knn_set4.mat');
%X= X+meanTestErrorMat;
%Y= Y+meanTestFalsePosMat;

%load('knn_set5.mat');
%X= X+meanTestErrorMat;
%Y= Y+meanTestFalsePosMat;

meanTestErrorMat=X./3;
meanTestFalsePosMat=Y./3;

save merged_knn.mat meanTestErrorMat meanTestFalsePosMat;

h = figure; 
hold on;
%plot(TRAIN_FRACS,meanTrainErrorMat, 'b-*');
plot(TRAIN_FRACS,meanTestErrorMat, 'r-o');
plot(TRAIN_FRACS,meanTestFalsePosMat, 'g-o');
xlabel('Training Fraction');
ylabel('Error rate');
legend('Test', 'false pos');
txt = sprintf('K=%d, Average of %d runs per training size', k, RUNS_PER_FRAC);
title(txt)
fname = sprintf('merge_knn_FINAL_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
