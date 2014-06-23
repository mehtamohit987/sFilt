THRESHOLDS=0:2:20;
NUM_THRESHOLDS=length(THRESHOLDS);
RUNS_PER_FRAC = 10;
TRAIN_FRACS = .1:.1:.9;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);


load('naivetest1.mat');
X1= meanTestErrorMat;
Y1= meanTrainErrorMat;
Z1= meanTestFalsePosMat;
X2=meanTestErrorMat2;
Y2=meanTrainErrorMat2;
Z2=meanTestFalsePosMat2;
X3=meanTestErrorMatT;
Y3=meanTrainErrorMatT;
Z3=meanTestFalsePosMatT;


load('naivetest2.mat');
X1= X1+meanTestErrorMat;
Y1= Y1+meanTrainErrorMat;
Z1= Z1+meanTestFalsePosMat;
X2= X2+meanTestErrorMat2;
Y2= Y2+meanTrainErrorMat2;
Z2= Z2+meanTestFalsePosMat2;
X3= X3+meanTestErrorMatT;
Y3= Y3+meanTrainErrorMatT;
Z3= Z3+meanTestFalsePosMatT;


%load('naivetest3.mat');
%X1= X1+meanTestErrorMat;
%Y1= Y1+meanTrainErrorMat;
%Z1= Z1+meanTestFalsePosMat;
%X2= X2+meanTestErrorMat2;
%Y2= Y2+meanTrainErrorMat2;
%Z2= Z2+meanTestFalsePosMat2;
%X3= X3+meanTestErrorMatT;
%Y3= Y3+meanTrainErrorMatT;
%Z3= Z3+meanTestFalsePosMatT;


load('naivetest4.mat');
X1= X1+meanTestErrorMat;
Y1= Y1+meanTrainErrorMat;
Z1= Z1+meanTestFalsePosMat;
X2= X2+meanTestErrorMat2;
Y2= Y2+meanTrainErrorMat2;
Z2= Z2+meanTestFalsePosMat2;
X3= X3+meanTestErrorMatT;
Y3= Y3+meanTrainErrorMatT;
Z3= Z3+meanTestFalsePosMatT;



%load('naivetest5.mat');
%X1= X1+meanTestErrorMat;
%Y1= Y1+meanTrainErrorMat;
%Z1= Z1+meanTestFalsePosMat;
%X2= X2+meanTestErrorMat2;
%Y2= Y2+meanTrainErrorMat2;
%Z2= Z2+meanTestFalsePosMat2;
%X3= X3+meanTestErrorMatT;
%Y3= Y3+meanTrainErrorMatT;
%Z3= Z3+meanTestFalsePosMatT;


meanTestErrorMat=X1./3;
meanTrainErrorMat=Y1./3;
meanTestFalsePosMat=Z1./3;
meanTestErrorMat2=X2./3;
meanTrainErrorMat2=Y2./3;
meanTestFalsePosMat2=Z2./3;
meanTestErrorMatT=X3./3;
meanTrainErrorMatT=Y3./3;
meanTestFalsePosMatT=Z3./3;


save merged_naive.mat meanTestErrorMat meanTestErrorMat2 meanTestErrorMatT meanTrainErrorMat meanTrainErrorMat2 meanTrainErrorMatT meanTestFalsePosMat meanTestFalsePosMat2 meanTestFalsePosMatT;


h = figure; 
hold on;
plot(THRESHOLDS,meanTestErrorMatT, 'r-o');
plot(THRESHOLDS,meanTrainErrorMatT, 'b-o');
plot(THRESHOLDS,meanTestFalsePosMatT, 'g-o');
xlabel('Threshold');
ylabel('Error rate');
legend('Test', 'Train', 'false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'thresh.fig');

h = figure; 
hold on;
plot(TRAIN_FRACS,meanTestErrorMat, 'r-o');
plot(TRAIN_FRACS,meanTrainErrorMat, 'b-o');
plot(TRAIN_FRACS,meanTestFalsePosMat, 'g-o');
plot(TRAIN_FRACS,meanTestErrorMat2, 'r-.o');
plot(TRAIN_FRACS,meanTrainErrorMat2, 'b-.o');
plot(TRAIN_FRACS,meanTestFalsePosMat2, 'g-.o');
xlabel('Training Fraction');
ylabel('Error rate');
legend('Test', 'Train', 'false pos', 'Test 2', 'Train 2', 'false pos 2');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'train_frac.fig');
