function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

%Initialize the centroids to be assigned randomly chosen values from the dataset X

%Randomly re-order the indeces of the examples
randidx = randperm(size(X,1));

%Then take the first K samples from the randomized dataset X, then assign them as the centroids
centroids  = X(randidx(1:K), :);


% =============================================================

end

