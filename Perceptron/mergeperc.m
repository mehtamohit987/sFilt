TRAIN_FRACS = .1:.1:.9;
RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);

load('perctest1.mat');
X1= meanTrainErrorMat;
X2= meanTestErrorMat_const_w;
Y2= meanTestFalsePosMat_const_w;
X3= meanTestErrorMat_updated_w;
Y3= meanTestFalsePosMat_updated_w;

load('perctest2.mat');
X1= X1+meanTrainErrorMat;
X2= X2+meanTestErrorMat_const_w;
Y2= Y2+meanTestFalsePosMat_const_w;
X3= X3+meanTestErrorMat_updated_w;
Y3= Y3+meanTestFalsePosMat_updated_w;

load('perctest3.mat');
X1= X1+meanTrainErrorMat;
X2= X2+meanTestErrorMat_const_w;
Y2= Y2+meanTestFalsePosMat_const_w;
X3= X3+meanTestErrorMat_updated_w;
Y3= Y3+meanTestFalsePosMat_updated_w;

load('perctest4.mat');
X1= X1+meanTrainErrorMat;
X2= X2+meanTestErrorMat_const_w;
Y2= Y2+meanTestFalsePosMat_const_w;
X3= X3+meanTestErrorMat_updated_w;
Y3= Y3+meanTestFalsePosMat_updated_w;

load('perctest5.mat');
X1= X1+meanTrainErrorMat;
X2= X2+meanTestErrorMat_const_w;
Y2= Y2+meanTestFalsePosMat_const_w;
X3= X3+meanTestErrorMat_updated_w;
Y3= Y3+meanTestFalsePosMat_updated_w;


meanTrainErrorMat=X1/5;
meanTestErrorMat_const_w=X2/5;
meanTestFalsePosMat_const_w=Y2/5;
meanTestErrorMat_updated_w=X3/5;
meanTestFalsePosMat_updated_w = Y3/5 ;


save merged_perceptron.mat meanTrainErrorMat meanTestErrorMat_const_w meanTestErrorMat_updated_w meanTestFalsePosMat_const_w meanTestFalsePosMat_updated_w;


h = figure; 
hold on;
plot(TRAIN_FRACS,meanTrainErrorMat, 'b-o');
plot(TRAIN_FRACS,meanTestErrorMat_const_w, 'r-o');
plot(TRAIN_FRACS,meanTestFalsePosMat_const_w, 'g-o');
plot(TRAIN_FRACS,meanTestErrorMat_updated_w, 'r-.o');
plot(TRAIN_FRACS,meanTestFalsePosMat_updated_w, 'g-.o');

xlabel('Training Fraction');
ylabel('Error rate');
legend('Train', 'Test (const w)', 'false pos (const w)', 'Test (updated w)', 'false pos (updated w)');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
fname = sprintf('resultsMERGE_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
