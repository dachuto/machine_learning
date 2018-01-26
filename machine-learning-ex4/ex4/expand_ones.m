function [A] = expand_ones(X)
A = [ones(size(X, 1), 1) X];
end
