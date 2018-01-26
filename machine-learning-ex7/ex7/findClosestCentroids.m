function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
K = size(centroids, 1);

m = size(X, 1);
idx = zeros(m, 1);

for i = 1:m
	distances = zeros(K, 1);
	for c = 1:K
		distances(c) = sumsq(X(i, :) - centroids(c, :));
	end
	[unused, min_index] = min(distances);
	idx(i) = min_index;
end

end

