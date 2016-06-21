function [images, Labels, numRows, numCols] = LoadMNIST(SelectedNumbers, type, numImages)
% Preprocessing code for Bernoulli mixture model
% Load a number of random images from the MNIST dataset
% All the digits are within the SelectedNumbers row array
% Written by Dang Manh Truong

% type = 1 : Train set. type = 2 : Test set
assert( (type == 1) | (type == 2), 'Type = 1 or Type = 2');

 
if type == 1
    Labels = ReadLabelsMNIST('train-labels.idx1-ubyte');
    filename = 'train-images.idx3-ubyte';
else
    Labels = ReadLabelsMNIST('t10k-labels.idx1-ubyte');
    filename = 't10k-images.idx3-ubyte';
end

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

[~] = fread(fp, 1, 'int32', 0, 'ieee-be'); 
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

% Find 'numImages' random images that are from 2 to 4
% TODO: Replace 'find' with logical indexing
% Index = find(2 <= Labels & Labels <= 4); 
Index = find(sum(bsxfun(@eq,Labels, SelectedNumbers),2) > 0);

s = RandStream('mt19937ar','Seed',0);
Permuted = randperm(s,size(Index,1));
% Permuted = randperm(size(Index,1));

Permuted = Permuted(1: numImages);
Index = sort(Index(Permuted));
Labels = Labels(Index);

images = zeros(numRows, numCols, numImages);
prev = 0;
ImageSize = numCols * numRows;
for i = 1 : numImages    
    % Ignore unneeded images 
    fread(fp, (Index(i) - prev - 1) * ImageSize , 'unsigned char');
    % Read image
    Temp = fread(fp,ImageSize, 'unsigned char');
    Temp = reshape(Temp, numRows, numCols); 
    images(:,:,i) = Temp;
    prev = Index(i);
end

fclose(fp);

images = permute(images,[2 1 3]);
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1], then binarize the images
images = double(images) / 255;
images(images < 0.5) = 0;
images(images >= 0.5) = 1;

end
