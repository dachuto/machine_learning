function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

samples = [0.01, 0.03, 0.1, 0.3, 1, 3, 10];
%samples = [0.1, 1];

C_sigma_error = [];

for C_ = samples
fprintf('C = %f sigma =', C_);
for sigma_ = samples
fprintf(' %f', sigma_)

model = svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_));
error = mean(double(svmPredict(model, Xval) ~= yval));

C_sigma_error = vertcat(C_sigma_error, [C_ sigma_ error]);

end
fprintf('\n');
end

[m, i] = min(C_sigma_error);
line = C_sigma_error(i(3), :);
C = line(1);
sigma = line(2);
end
