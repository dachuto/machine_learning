function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y); % number of training examples

[J, grad] = useful_cost_function(sigmoid(X * theta), y);
theta(1, 1) = 0;
J_add = lambda * dot(theta, theta) * 0.5 / m;
J += J_add;

grad_add = lambda * theta / m;
grad += grad_add';
grad = grad';

end
