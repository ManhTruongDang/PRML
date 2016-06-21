function [Cluster] = GetClusterBMM(images,mu, K)
% Input : images : The input images (N x D)
%       : mu : Parameters for Bernoulli distribution for each pixel (K x D)
%       : phi: Mixing coefficients (K x 1)
%       : K: Number of mixtures 
% Output: Cluster (N x 1) The clusters most likely to be associated with
% each image

N = size(images,1);
ClusterSum = zeros(N,K);
Temp1 = mu;
Temp2 = 1 - Temp1;
for n = 1 : N
    for k = 1 : K
        % In:  http://blog.manfredas.com/expectation-maximization-tutorial/
        % They used the sum of mu's, but to be honest I don't know why
        % Anyway it only gives about 80% accuracy, while mine gives 90%
        %ClusterSum(n,k) = sum(Temp1(k,images(n,:) == 1)) + sum(Temp2(k,images(n,:) == 0));
        ClusterSum(n,k) = prod(Temp1(k,images(n,:) == 1)) * prod(Temp2(k,images(n,:) == 0));
    end
end
[~,Cluster] = max(ClusterSum,[],2);
