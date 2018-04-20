function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, 1:k);
%
%Use the first K components/eigenvectors in U (i.e. the first K columns
%in U) to obtain a new variable U_reduce...
U_reduce = U(:, 1:K);
%Use U_redice to compute the projection of the data: Z
for i = 1:length(X),
  Z(i, :) = X(i, :) * U_reduce;
end


% =============================================================

end
