function [error_ratio, false_positives_ratio] = knnClassify(dataVectors, dataLabels, queryVectors, queryLabels, k, supermaj_factor)

[num_of_items, num_of_features] = size(queryVectors);

[neighborIdx, neighborDistances] = kNearestNeighbors(dataVectors, queryVectors, k);

nearestNeighborsLables = dataLabels(neighborIdx);
positives = nearestNeighborsLables==1;
sum_positives = sum(positives, 2);

classifications = sum_positives > supermaj_factor*k;

classifications = 2*classifications - 1;

isError =  classifications ~= queryLabels;
IsFalsePositive =  (classifications == 1) & (queryLabels == -1);
mistakes = sum(isError);
false_positives = sum(IsFalsePositive);

num_of_hams = sum(queryLabels == -1);
error_ratio = mistakes / num_of_items;
false_positives_ratio = false_positives / num_of_hams;
