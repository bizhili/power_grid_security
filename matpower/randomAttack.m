function [attackedbranchActive]= randomAttack(measureBranchActive, std)
    attackedbranchActive= measureBranchActive+ std*randn(size(measureBranchActive));
end