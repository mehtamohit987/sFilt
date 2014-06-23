TRAIN_FRACS = .1:.1:.9;
RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
%num of neighbours
k=1;

load('merged_perceptron.mat');
perE=meanTestErrorMat_const_w;
perF=meanTestFalsePosMat_const_w;

load('merged_knn.mat');
knnE=meanTestErrorMat;
knnF=meanTestFalsePosMat;

load('merged_naive.mat');
naiE=meanTestErrorMat;
naiF=meanTestFalsePosMat;


h = figure; 
hold on;

plot(TRAIN_FRACS,perE, 'r-o');
plot(TRAIN_FRACS,perF, 'r-.o');

plot(TRAIN_FRACS,knnE, 'g-o');
plot(TRAIN_FRACS,knnF, 'g-.o');

plot(TRAIN_FRACS,naiE, 'b-o');
plot(TRAIN_FRACS,naiF, 'b-.o');



xlabel('Training Fraction');
ylabel('Error rate');
legend('Perceptron Test Error', 'Perceptron false pos','k-NN Test Error', 'k-NN false pos','Naive Test Error', 'Naive false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'comparative.fig');
