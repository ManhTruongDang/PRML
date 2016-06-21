function [Labels] = ReadLabelsMNIST(filename)
% Read the labels of the MNIST dataset
% Written by Dang Manh Truong    
    
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    Labels = fread(fp,inf, 'unsigned char');    
    assert(size(Labels,1) == numLabels, 'Mismatch in label count');
   
    fclose(fp);
end