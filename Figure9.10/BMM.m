function [] = BMM()
% Bernoulli mixture model for classification of MNIST dataset
% Based on Figure 9.10 in the book Pattern recognition and machine learning
% Partly inspired by: http://blog.manfredas.com/expectation-maximization-tutorial/
% Written by Dang Manh Truong
% I was able to reproduce the 3 pictures of digits 2,3 and 4, and the
% results for classfication on the MNIST test set (2 to 4) were satisfying
% with about 90% correct, not too bad for my first project.
% But it's not a good score compared with state-of-the-art methods 

pause(1);
fprintf('Bernoulli mixture model using Expectation - Maximization\n');
fprintf('to recreate Figure 9.10, chapter 9, Pattern recognition and machine learning\n');
% Change these lines if you wish
SelectedNumbers = [2 3 4 ]; % The numbers that we care about in the dataset
numTrainImages = 600; % For the numbers from 2-4: <= 17391
numTestImages = 3024; % For those from 2-4: <= 3024
rng(0,'twister');

% Step 1: Initialization

% images: #pixels * #examples
[images, Labels, numRows, numCols] = LoadMNIST(SelectedNumbers, 1, numTrainImages);

N = numTrainImages; % N : The number of train images
K = size(SelectedNumbers,2); % K : The number of mixtures
D = numRows * numCols; % Dimension of each image
phi = ones(K,1) * 1/K; % Mixing coefficients
mu = (0.75-0.25) * rand(K,D) + 0.25  ; % Means of each components
mu = mu ./ repmat(sum(mu,2),1,D);
Res = zeros(N,K); % Res(k,n): Responsibility of component 'k' given data point X(n,:)
effNum = zeros(K,1); % Effective number of data points associated with each component
X = images'; % (N x D) Each row is an image

% Step 2: Expectation - Maximization
[mu, ~, ~, ~] = TrainBMM(X, mu, phi, Res, effNum);

% Step 3: Testing 

[TestImages, TestLabels ,~, ~] = LoadMNIST(SelectedNumbers, 2,numTestImages);
TestX = TestImages'; % Each image is in one row

[Correct, MisClassified] = TestBMM(X, TestX, mu, Labels, TestLabels);

fprintf('Correct: %f percents \n',100 * Correct / numTestImages);
fprintf('The misclassification matrix: \n');
MisClassfied = MisClassified(SelectedNumbers, SelectedNumbers)

% The MisClassified matrix when I used all 17931 train images (that are
% from 2 to 4) to test all 3024 test images (again, from 2 to 4):

%  0   130    48
% 84     0    21
% 17     1     0
% The algorithm appears to correctly labels all the digits 4, but fails 
% spectacularly at the digit 2. 
