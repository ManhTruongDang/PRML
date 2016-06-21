function [mu, phi, Res, effNum] = TrainBMM(X, mu, phi, Res, effNum)
% Training phase for Bernoulli mixture model using Expecation-Maximization
% Written by Dang Manh Truong
% The parameters here closely resemble those in the book Pattern
% recognition and machine learning (Bishop), chapter 9
% D: The dimension of each data points
% N: The number of data points
% K: The number of Bernoulli mixtures 
% Input: X (N x D) Data points to be processed (each row - a data point)
%      : mu(K x D) Bernoulli parameters for each mixture
%      : phi(K x 1) Mixing coefficients for each mixture
%      : Res(N x K) Responsibilities of each component (1 - K) given a data
%      point (1 - n)
%      : effNum(K x 1) Effective number of observations for each mixture  
% Output: The new values of mu, phi, Res and effNum
% Most of the time only mu will be used

% Size of each image. I don't want to pass these to the function
% because their only purpose is to show the images
numRows = 28; numCols = 28; 

N = size(X,1); K = size(phi,1);
iterNum = 0;
uniform = 1 / K; 
fprintf('E-M algorithm in progress. This may take a while.....\n');
while 1
    % E-step    
    
    % Equivalent unvectorized code:
%     for n = 1 : N
%         for k = 1 : K
%             Res(n,k) = 1;
%             for i = 1 : D % D = size(X,2)
%                 if X(n,i) == 1
%                     Res(n,k) = Res(n,k) * mu(k,i);
%                 else
%                     Res(n,k) = Res(n,k) * (1 - mu(k,i));
%                 end
%             end
%         end
%    end    
    % TODO: Vectorize this part ASAP!!!!
    for n = 1 : N
        for k = 1 : K
            Temp1 = mu(k,:);
            Temp2 = 1 - mu(k,:);            
            Res(n,k) = prod(Temp1(X(n,:) == 1)) * prod(Temp2(X(n,:) == 0));            
        end    
    end           
    Res = bsxfun(@times, Res, phi');    
    % Divide by the denominator
    Sum = sum(Res,2);
    Sum(Sum == 0) = uniform;
    Res = bsxfun(@rdivide, Res, Sum);
    
    % M-step    
    effNum = sum(Res,1);
    mu = Res' * X;
    mu = bsxfun(@rdivide, mu, effNum');   
    % Check for convergence
    iterNum = iterNum + 1;
    
    for k = 1 : K
        subplot(1,K,k)
        Result = reshape(mu(k,:), numRows, numCols);    
        subimage(Result)     
    end
    hold on
    pause(1)
    fprintf('Iteration %d \n',iterNum);
    
    if iterNum >= 10
        break;
    end
end
fprintf('Press any key to continue\n\n\n');
pause
close(gcf)

end