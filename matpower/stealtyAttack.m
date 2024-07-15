function [attackedbranchActive]= stealtyAttack(measureBranchActive, H, std)
    dimData= length(measureBranchActive);
    shapeH= size(H);
    dimstate= shapeH(2);
    attackedbranchActive= measureBranchActive+ (H*(std*randn(dimstate, dimData))).';
end