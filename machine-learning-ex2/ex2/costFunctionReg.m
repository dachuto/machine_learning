function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

[J, grad] = costFunction(theta, X, y);
theta(1, 1) = 0;
J_add = lambda * dot(theta, theta) * 0.5 / m;
J += J_add;

grad_add = lambda * theta / m;
grad += grad_add';

% =============================================================

end
