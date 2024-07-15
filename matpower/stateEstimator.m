function [estimateBusVoltageAngle]= stateEstimator(BranchActive, H, cov)%H: measurement matrix, cov: covariance matrix
    results= lscov(H, BranchActive.', cov);
    estimateBusVoltageAngle= results.';
end