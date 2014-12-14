TRAIN_FRACS = .1:.1:.9;
RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
%num of neighbours
k=1;

load('merged_perceptron.mat');
perE=meanTestErrorMat_const_w;
perF=meanTestFalsePosMat_const_w;

load('merged_naive.mat');
naiE=meanTestErrorMat;
naiF=meanTestFalsePosMat;


h = figure; 
hold on;

plot(TRAIN_FRACS,perE, 'r-o');
plot(TRAIN_FRACS,perF, 'r-.o');

plot(TRAIN_FRACS,naiE, 'b-o');
plot(TRAIN_FRACS,naiF, 'b-.o');



xlabel('Training Fraction');
ylabel('Error rate');
legend('Content-based Test Error', 'Content-based false pos','Collaborative Test Error', 'Collaborative false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'comparativeMOD.fig');
