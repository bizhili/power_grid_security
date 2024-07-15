function [noisedbranchActive]= measurementNoise(measureBranchActive, std)
    noisedbranchActive= measureBranchActive+ std*randn(size(measureBranchActive));
end