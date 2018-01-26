function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

a1 = [ones(m, 1) X];
o1 = [ones(size(a1), 1) sigmoid(a1 * Theta1')];
o2 = sigmoid(o1 * Theta2');

[max, index] = max(o2, [], 2);

p = index;

end
