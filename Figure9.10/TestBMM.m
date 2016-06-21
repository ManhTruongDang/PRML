function [Correct, MisClassified] = TestBMM(X, TestX, mu, Labels, TestLabels)
% Testing phase for Bernoulli mixture model
% Written by Dang Manh Truong
% The parameters here closely resemble those in the book Pattern
% recognition and machine learning (Bishop), chapter 9
% N : Number of data points
% N': Number of test points
% K : Number of mixtures
% D : Dimension of each data points
% Input: X (N x D) Train data
%      : TestX (N' x D) Test data
%      : mu (K x D) Bernoulli parameters learned from training phase
%      : numTestImages Number of test data needed 
%      : Labels (N x 1) Train labels
%      : TestLabels (N' x 1) Test labels
% Output: Correct: The number of times the algorithm get it right
%       : MisClassified(10,10) : The misclassification matrix
% MisClassified(i,j) : The number of times that the digit 'i' is
% misclassified as digit 'j'. Of course the diagonal is zero

K = size(mu,1);
N = size(X,1);
numTestImages = size(TestX,1);
Correct = 0; 
MisClassified = zeros(10,10); 
digitsInTheSameCluster = zeros(10,numTestImages);

TrainClusters = GetClusterBMM(X,mu,K); % N x 1 
TestClusters = GetClusterBMM(TestX,mu,K);
for i = 1 : numTestImages     
    for n = 1 : N            
        if TestClusters(i) == TrainClusters(n)
            digitsInTheSameCluster(Labels(n),i) = digitsInTheSameCluster(Labels(n),i) + 1;
        end    
    end
    [~, AssignedLabel] = max(digitsInTheSameCluster(:,i));
    if AssignedLabel == TestLabels(i)
        Correct = Correct + 1;
    else
        MisClassified(TestLabels(i),AssignedLabel) = MisClassified(TestLabels(i), AssignedLabel) + 1;
    end    
end