function mpc = randomize_load_once(mpc0, varargin)
% Randomize active/reactive loads with feasibility guards (fixed names).
% Usage:
%   mpc = randomize_load_once(loadcase('case118'), ...
%           'pf_min',0.9,'pf_max',0.99,'alpha',[0.6 1.2]);

    p = inputParser;
    addParameter(p, 'pf_min', 0.90);
    addParameter(p, 'pf_max', 0.99);
    addParameter(p, 'alpha',  [0.6 1.2]);      % per-bus P scaling range
    addParameter(p, 'total_P_margin', 0.95);   % cap sum(PD) vs sum(PMAX)
    addParameter(p, 'total_Q_margin', 0.95);   % cap sum(QD) vs gen Q range
    parse(p, varargin{:});
    S = p.Results;

    define_constants;   % defines PD, QD, BUS_TYPE, PMAX, QMIN, QMAX, etc.
    mpc = mpc0;

    % --- sanity checks
    if ~any(mpc.bus(:, BUS_TYPE) == 3)
        error('No slack (REF) bus found.');
    end

    % --- pick load buses (PQ safest)
    isPQ = (mpc.bus(:, BUS_TYPE) == 1);
    idx  = find(isPQ);
    if isempty(idx), idx = (1:size(mpc.bus,1))'; end

    % --- randomize P (MW)
    scale = S.alpha(1) + (S.alpha(2)-S.alpha(1)) * rand(numel(idx),1);
    P0    = mpc0.bus(idx, PD);                 % base MW at those buses
    Pload = max(0, P0 .* scale);

    % inject small load where base is zero (avoid totally empty systems)
    zero_mask = (P0 == 0);
    if any(zero_mask)
        small_ref = 0.01 * max(1, mean(mpc0.bus(:, PD)));  % MW
        Pload(zero_mask) = (S.alpha(1) + (S.alpha(2)-S.alpha(1)) * ...
                            rand(sum(zero_mask),1)) .* small_ref;
    end

    % --- random lagging power factor and compute Q
    pf   = S.pf_min + (S.pf_max - S.pf_min) * rand(numel(idx),1);
    phi  = acos(pf);
    Qload = Pload .* tan(phi);                 % inductive (+QD)

    % --- cap total P to available gen capability
    on   = mpc.gen(:, GEN_STATUS) > 0;
    Pcap = S.total_P_margin * sum(mpc.gen(on, PMAX));
    Psum = sum(Pload);
    if Psum > Pcap && Psum > 0
        s = Pcap / Psum;
        Pload = Pload * s;
        Qload = Qload * s;
    end

    % --- cap total Q to aggregate gen Q capability
    Qmin = sum(mpc.gen(on, QMIN));
    Qmax = sum(mpc.gen(on, QMAX));
    Qsum = sum(Qload);
    if Qsum > S.total_Q_margin * Qmax && Qsum > 0
        s = (S.total_Q_margin * Qmax) / Qsum;  Qload = Qload * s;
    elseif Qsum < S.total_Q_margin * Qmin && Qsum < 0
        s = (S.total_Q_margin * Qmin) / Qsum;  Qload = Qload * s;
    end

    % --- write back
    mpc.bus(:, PD) = 0;      mpc.bus(:, QD) = 0;
    mpc.bus(idx, PD) = Pload;
    mpc.bus(idx, QD) = Qload;

    % --- options to improve AC PF/OPF robustness
    mpc.mpopt = mpoption('pf.enforce_q_lims', 1, 'verbose', 0, 'out.all', 0);
end
