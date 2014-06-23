addpath('..');


TRAIN_FRACS = .1:.1:.9;
RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
DIRNAME ='../Data/enron1';


trainErrorMat    = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);

testErrorMat_const_w   = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);

testErrorMat_updated_w = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);

testFalsePosMat_const_w  = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);

testFalsePosMat_updated_w  = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);


for iTrainFrac = 1:NUM_TRAIN_FRACS
    trainFrac = TRAIN_FRACS(iTrainFrac)
    for run=1:RUNS_PER_FRAC
        display(run);

        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,(run-1));
        train = importdata(fname);

        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,(run-1));
        test  = importdata(fname);
        

        trainVectors = train(:,1:end-1);

        trainLabels = train(:,end);

        trainLabels = 2*trainLabels - 1;
        

        testVectors = test(:,1:end-1);

        testLabels = test(:,end);

        testLabels = 2*testLabels - 1;
	        
        trainVectors = addDummyColumn(trainVectors);
        testVectors  = addDummyColumn(testVectors);
        
     
        trainVectors = normalizeRows(trainVectors);
        testVectors  = normalizeRows(testVectors);

     
        num_of_features = size(trainVectors, 2);
        initial_w = zeros(1, num_of_features);

     
	 [trainErrorMat(run,iTrainFrac) , ...
         false_positives_ratio, ...
         w ] ...
            = perceptronAlg(initial_w, trainVectors, trainLabels);
        
        
        [testErrorMat_const_w(run,iTrainFrac), ...
         testFalsePosMat_const_w(run,iTrainFrac) ] ...
            = hyperplaneClassify(w, 0, testVectors, testLabels);
            
        
        [testErrorMat_updated_w(run,iTrainFrac), ...
         testFalsePosMat_updated_w(run,iTrainFrac), ...
         w ] ...
            = perceptronAlg(w, testVectors, testLabels);
                
    end
end

meanTrainErrorMat = mean(trainErrorMat, 1);
meanTestErrorMat_const_w = mean(testErrorMat_const_w, 1);
meanTestFalsePosMat_const_w = mean(testFalsePosMat_const_w, 1);
meanTestErrorMat_updated_w = mean(testErrorMat_updated_w, 1);
meanTestFalsePosMat_updated_w = mean(testFalsePosMat_updated_w, 1);

save perctest5.mat meanTrainErrorMat meanTestErrorMat_const_w meanTestErrorMat_updated_w meanTestFalsePosMat_const_w meanTestFalsePosMat_updated_w;

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
fname = sprintf('results_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
