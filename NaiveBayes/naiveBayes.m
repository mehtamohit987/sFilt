function [trainError, testError, testFalsePos] = ...
    naiveBayes(trainData, testData, not_biased, thresh)

global IGNORE_RARE_WORDS IGNORE_COMMON;


Y = 2*int8(trainData(:,end))-1;
testY = 2*int8(testData(:,end))-1;

X = trainData(:,1:end-1);
testX = testData(:,1:end-1);
X = (X>0);
testX = (testX>0);

spam = X(Y==+1,:);
ham  = X(Y==-1,:);
ws = mean(spam);
wh = mean(ham);


HAM_DICT_SIZE = 250000;

SPAM_DICT_SIZE = HAM_DICT_SIZE* 10;
AVG_WORDS_IN_MSG = 100;
HAM_WORD_PROB = 1 -...
    ((HAM_DICT_SIZE-1)/HAM_DICT_SIZE)^AVG_WORDS_IN_MSG;
SPAM_WORD_PROB = 1 -...
    ((SPAM_DICT_SIZE-1)/SPAM_DICT_SIZE)^AVG_WORDS_IN_MSG;
wh(wh==0) = HAM_WORD_PROB;
ws(ws==0) = SPAM_WORD_PROB;


ind = true(1,size(ws,2));

if IGNORE_COMMON>0

    common  = abs(ws./(ws+wh) - .5) > IGNORE_COMMON;
    ncommon = sum(~common);
    if ncommon>0
        sprintf('eliminated %d common (of %d features)', ncommon, sum(ind))
        ind = ind & common;
    end
end
if IGNORE_RARE_WORDS>0
    rare = (ws+wh>IGNORE_RARE_WORDS);
    nrare= sum(~rare);
    if nrare>0
        sprintf('eliminated %d rare (of %d features)', nrare, sum(ind))
        ind = ind & rare;
    end
end

ws = ws(ind);
wh = wh(ind);
X = X(:,ind);
testX = testX(:,ind);

trainSpamProp = mean(trainData(:,end));
testSpamProp  = mean(testData(:,end));
if not_biased
    testSpamProp  = .5;
    trainSpamProp = .5;
end
[trainError, dummy] = calcError(X,Y,ws,wh,trainSpamProp, thresh);
[testError, testFalsePos ] = calcError(testX,testY,ws,wh,testSpamProp,thresh);
