function [bestEpsilon bestF1] = selectThreshold(truth_val, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)
	predictions = (pval < epsilon);

	matches = @(predicted, real) sum((predictions == predicted) & (truth_val == real));

	true_positives = matches(1, 1);
	false_positives = matches(1, 0);
	false_negatives = matches(0, 1);
	precision = true_positives / (true_positives + false_positives);
	recall = true_positives / (true_positives + false_negatives);
	F1 = 2 * precision * recall / (precision + recall);

	if F1 > bestF1
		bestF1 = F1;
		bestEpsilon = epsilon;
	end
end

end
