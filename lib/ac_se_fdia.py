
"""
ac_se_fdia.py
--------------
AC State Estimation and FDIA residual checks using pandapower's WLS estimator.

Requirements:
    pip install pandapower pandas numpy

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-10-08

Overview
--------
This module helps you run AC state estimation (WLS) using pandapower when you have:
  1) Full bus injection measurements (P, Q at buses), and
  2) Branch flow measurements (P, Q at line "from" and/or "to" ends).

It wraps pandapower's state_estimation API and exposes convenience functions to:
  - Add measurements to a pandapower net
  - Run AC state estimation
  - Extract the estimated state (voltage magnitudes/angles)
  - Compute raw and normalized residuals for FDIA detection
  - Flag suspicious measurements by a configurable sigma threshold

Angles:
  - pandapower uses degrees for voltage angles (va_degree).
Units:
  - Active/reactive powers in MW/Mvar by default for pandapower.
  - Voltage magnitudes are per-unit (pu).

Typical Use
-----------
    import pandapower.networks as nw
    import ac_se_fdia as se

    # 1) Create or load a pandapower network
    net = nw.case14()  # or build your own 'net'

    # 2) Provide measurements (numpy arrays or lists).
    #    bus_inj shape: [n_bus_meas, 3] columns = [bus_idx, P(MW), Q(Mvar)]
    #    branch_flows shape: [n_line_meas, 5] columns = [line_idx, P_from, Q_from, P_to, Q_to]
    #    You can pass None for any value that is not measured.
    import numpy as np
    bus_inj = np.array([
        [0,  1.50,  0.30],   # at bus 0
        [1, -0.40, -0.10],   # at bus 1
    ], dtype=float)

    branch_flows = np.array([
        [0,  0.80,  0.10,  None, None],  # line 0, only from-end P,Q measured
        [3,  None, None,   0.60, 0.05],  # line 3, only to-end P,Q measured
    ], dtype=object)  # object to allow None

    # 3) Add measurements (choose realistic std devs for your metering)
    se.add_measurements_to_net(
        net,
        bus_injections=bus_inj,
        branch_flows=branch_flows,
        std_p=0.02,  # MW
        std_q=0.02   # Mvar
    )

    # 4) Run state estimation
    result = se.run_ac_state_estimation(net)

    # 5) Compute residuals and simple FDIA flags
    res_df, suspicious = se.compute_residuals_and_flags(net, sigma_threshold=3.0)

    # 6) Access the estimated state
    Vmag, Vang = result["vm_pu"], result["va_degree"]

"""

from typing import Optional, Tuple, Dict, Any, Iterable, Union
import numpy as np
import pandas as pd

try:
    import pandapower as pp
    import pandapower.topology as top
    from pandapower.estimation import estimate as pp_run_se
    from pandapower.control import ConstControl  # not strictly needed, but handy for extensions
except Exception as e:
    raise ImportError(
        "This module requires 'pandapower'. Install it via 'pip install pandapower'. "
        f"Original import error: {e}"
    )


# ---------------------------------------------------------------------------
# Helpers for adding measurements
# ---------------------------------------------------------------------------

def _as_iter(a: Optional[Union[np.ndarray, Iterable]]) -> Iterable:
    if a is None:
        return []
    if isinstance(a, np.ndarray):
        return a.tolist()
    return a


def add_measurements_to_net(
    net: "pp.pandapowerNet",
    bus_injections: Optional[Union[np.ndarray, Iterable]] = None,
    branch_flows: Optional[Union[np.ndarray, Iterable]] = None,
    transformers: Optional[Union[np.ndarray, Iterable]] = None,
    std_p: float = 0.02,
    std_q: float = 0.02,
) -> None:
    """
    Add P/Q injection and/or branch flow measurements to a pandapower net.

    Parameters
    ----------
    net : pandapowerNet
        Network to attach measurements to.
    bus_injections : array-like or None
        Shape [n, 3] with columns [bus_idx, P(MW), Q(Mvar)].
        Use None in P or Q columns if the value is not measured.
    branch_flows : array-like or None
        Shape [m, 5] with columns [line_idx, P_from, Q_from, P_to, Q_to].
        Use None if a given end (from/to) or component is not measured.
    std_p : float
        Standard deviation for active power measurements (MW).
    std_q : float
        Standard deviation for reactive power measurements (Mvar).

    Notes
    -----
    - For lines, pandapower requires specifying the measurement 'side' as 'from' or 'to'.
    - If you have transformer flows, you can extend similarly with element_type='trafo'.
    """
    # Clean existing measurements if any (optional, comment out if you want to append)
    if hasattr(net, "measurement") and len(net.measurement):
        # Append instead of drop if you prefer
        net.measurement.drop(net.measurement.index, inplace=True)

    # Ensure measurement table exists
    if not hasattr(net, "measurement"):
        pp.create_empty_tables(net)

    # Add bus injections
    for row in _as_iter(bus_injections):
        if row is None:
            continue
        bus, P, Q = row
        bus = int(bus)
        if P is not None and not np.isnan(P):
            pp.create_measurement(net, "p", "bus", float(P), std_p, bus)
        if Q is not None and not np.isnan(Q):
            pp.create_measurement(net, "q", "bus", float(Q), std_q, bus)

    # Add branch flows
    for row in _as_iter(branch_flows):
        if row is None:
            continue
        line_idx, Pfrom, Qfrom, Pto, Qto = row
        line_idx = int(line_idx)
        if Pfrom is not None and not (isinstance(Pfrom, float) and np.isnan(Pfrom)):
            pp.create_measurement(net, "p", "line", float(Pfrom), std_p, line_idx, side="from")
        if Qfrom is not None and not (isinstance(Qfrom, float) and np.isnan(Qfrom)):
            pp.create_measurement(net, "q", "line", float(Qfrom), std_q, line_idx, side="from")
        if Pto is not None and not (isinstance(Pto, float) and np.isnan(Pto)):
            pp.create_measurement(net, "p", "line", float(Pto), std_p, line_idx, side="to")
        if Qto is not None and not (isinstance(Qto, float) and np.isnan(Qto)):
            pp.create_measurement(net, "q", "line", float(Qto), std_q, line_idx, side="to")
    for row in _as_iter(transformers):
        if row is None:
            continue
        line_idx, Pfrom, Qfrom, Pto, Qto = row
        line_idx = int(line_idx)
        if Pfrom is not None and not (isinstance(Pfrom, float) and np.isnan(Pfrom)):
            pp.create_measurement(net, "p", "trafo", float(Pfrom), std_p, line_idx, side="hv")
        if Qfrom is not None and not (isinstance(Qfrom, float) and np.isnan(Qfrom)):
            pp.create_measurement(net, "q", "trafo", float(Qfrom), std_q, line_idx, side="hv")
        if Pto is not None and not (isinstance(Pto, float) and np.isnan(Pto)):
            pp.create_measurement(net, "p", "trafo", float(Pto), std_p, line_idx, side="lv")
        if Qto is not None and not (isinstance(Qto, float) and np.isnan(Qto)):
            pp.create_measurement(net, "q", "trafo", float(Qto), std_q, line_idx, side="lv")


# ---------------------------------------------------------------------------
# State Estimation
# ---------------------------------------------------------------------------

def run_ac_state_estimation(
    net: "pp.pandapowerNet",
    init: str = "flat",
    enforce_convergence: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Runs pandapower AC WLS state estimation and returns the estimated state.

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network with measurements already added (see add_measurements_to_net).
    init : {"flat", "results"}
        Initialization mode for SE. "flat" sets 1.0 pu magnitudes and 0 deg angles to start.
        "results" uses existing net.res_bus as a warm start (if available).
    enforce_convergence : bool
        If True, raises a RuntimeError if SE fails to converge.
    kwargs :
        Forwarded to pandapower.state_estimation.estimate (e.g., tolerance, max_iterations).

    Returns
    -------
    dict:
        {
            "vm_pu": np.ndarray of shape [nbus],
            "va_degree": np.ndarray of shape [nbus],
            "converged": bool,
            "iterations": int (if available),
        }
    """
    # Ensure necessary result tables exist
    if not hasattr(net, "res_bus"):
        pp.create_empty_tables(net)

    # Run SE
    pp_run_se(net, init=init, **kwargs)

    converged = bool(getattr(net, "se_converged", True))  # pandapower sets .se_converged in some versions
    if enforce_convergence and not converged:
        raise RuntimeError("State Estimation did not converge. Consider checking measurements and std devs.")

    vm = net.res_bus_est.vm_pu.to_numpy(copy=True)
    va = net.res_bus_est.va_degree.to_numpy(copy=True)

    # iterations are not always exposed; try best-effort
    iterations = int(getattr(net, "se_iterations", -1)) if hasattr(net, "se_iterations") else -1

    return {
        "vm_pu": vm,
        "va_degree": va,
        "converged": converged,
        "iterations": iterations
    }


# ---------------------------------------------------------------------------
# Residuals & FDIA flags
# ---------------------------------------------------------------------------

def compute_residuals_dataframe(net: "pp.pandapowerNet") -> pd.DataFrame:
    """
    Returns a DataFrame of measurement residuals after SE.

    The DataFrame includes (if available from pandapower version):
        - 'residual': z - h(x_hat)
        - 'std_dev': the measurement standard deviation used
        - 'value': the raw measurement value z
        - 'meas_type', 'element_type', 'element', and 'side' metadata

    Returns
    -------
    pd.DataFrame
    """
    if not hasattr(net, "res_measurement") or net.res_measurement is None or len(net.res_measurement) == 0:
        # Fallback: build from net.measurement if residuals are not computed (older versions).
        if hasattr(net, "measurement") and len(net.measurement):
            df = net.measurement.copy()
            # placeholders
            df["residual"] = np.nan
            df["std_dev"] = df.get("std_dev", np.nan)
            return df.rename(columns={
                "type": "meas_type",
                "element_type": "element_type",
                "element": "element"
            }).reset_index(drop=True)
        raise RuntimeError("No measurement residuals found; ensure SE ran and measurements exist.")

    # Make a friendly copy
    df = net.res_measurement.copy().reset_index(drop=True)
    # Different pandapower versions may use different column names; normalize them
    rename_map = {}
    if "type" in df.columns and "meas_type" not in df.columns:
        rename_map["type"] = "meas_type"
    if "element_type" not in df.columns and "elem_type" in df.columns:
        rename_map["elem_type"] = "element_type"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure std_dev present (some versions store it in net.measurement)
    if "std_dev" not in df.columns and hasattr(net, "measurement") and "std_dev" in net.measurement.columns:
        df = df.merge(
            net.measurement[["type", "element_type", "element", "side", "std_dev"]]
            .rename(columns={"type": "meas_type"}),
            on=["meas_type", "element_type", "element", "side"],
            how="left"
        )

    # Compute normalized residual if possible
    if "residual" in df.columns and "std_dev" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["normalized_residual"] = df["residual"] / df["std_dev"]

    return df


def compute_residuals_and_flags(
    net: "pp.pandapowerNet",
    sigma_threshold: float = 3.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute residuals and flag suspicious measurements (simple sigma rule).

    Parameters
    ----------
    net : pandapowerNet
        Network after running SE.
    sigma_threshold : float
        Threshold for |normalized_residual| above which a measurement is flagged.

    Returns
    -------
    (residual_df, suspicious_flags)
        residual_df : pd.DataFrame with metadata and residual columns
        suspicious_flags : pd.Series[bool] aligned with residual_df index
    """
    df = compute_residuals_dataframe(net)
    if "normalized_residual" not in df.columns:
        # Fallback: if std_dev missing, use raw residuals with a warning-like note
        df["normalized_residual"] = np.nan

    suspicious = pd.Series(False, index=df.index)
    if df["normalized_residual"].notna().any():
        suspicious = df["normalized_residual"].abs() > sigma_threshold
    elif "residual" in df.columns:
        # If we only have raw residuals, use a heuristic threshold: 3 * median absolute deviation
        med = np.nanmedian(np.abs(df["residual"].to_numpy(dtype=float)))
        if med == 0 or np.isnan(med):
            med = 1e-6
        suspicious = np.abs(df["residual"].to_numpy(dtype=float)) > (3.0 * med)

    return df, suspicious


# ---------------------------------------------------------------------------
# Convenience one-shot wrapper
# ---------------------------------------------------------------------------

def estimate_state(
    net: "pp.pandapowerNet",
    bus_injections: Optional[Union[np.ndarray, Iterable]] = None,
    branch_flows: Optional[Union[np.ndarray, Iterable]] = None,
    transformers: Optional[Union[np.ndarray, Iterable]] = None,
    std_p: float = 0.02,
    std_q: float = 0.02,
    init: str = "flat",
    **se_kwargs
    ) -> Dict[str, Any]:
    """
    One-shot pipeline: add measurements -> run SE
    Returns
    -------
    dict with keys:
        - "state": {"vm_pu": ..., "va_degree": ..., "converged": bool, "iterations": int}
    """
    add_measurements_to_net(
        net,
        bus_injections=bus_injections,
        branch_flows=branch_flows,
        transformers= transformers,
        std_p=std_p,
        std_q=std_q
    )
    state = run_ac_state_estimation(net, init=init, **se_kwargs)
    return state

def estimate_state_and_residuals(
    net: "pp.pandapowerNet",
    bus_injections: Optional[Union[np.ndarray, Iterable]] = None,
    branch_flows: Optional[Union[np.ndarray, Iterable]] = None,
    std_p: float = 0.02,
    std_q: float = 0.02,
    sigma_threshold: float = 3.0,
    init: str = "flat",
    **se_kwargs
) -> Dict[str, Any]:
    """
    One-shot pipeline: add measurements -> run SE -> compute residuals & flags.

    Returns
    -------
    dict with keys:
        - "state": {"vm_pu": ..., "va_degree": ..., "converged": bool, "iterations": int}
        - "residuals": pd.DataFrame
        - "flags": pd.Series[bool]
    """
    add_measurements_to_net(
        net,
        bus_injections=bus_injections,
        branch_flows=branch_flows,
        std_p=std_p,
        std_q=std_q
    )
    state = run_ac_state_estimation(net, init=init, **se_kwargs)
    residuals, flags = compute_residuals_and_flags(net, sigma_threshold=sigma_threshold)
    return {"state": state, "residuals": residuals, "flags": flags}

def map_branches_to_pandapower(net, original_connections):
    """
    Maps a list of original branch connections to their corresponding element type
    (line or trafo) and index in a pandapower network.
    Args:
        net (pandapowerNet): The pandapower network object.
        original_connections (list of tuples): A list where each tuple represents a
                                               connection, e.g., [(from_bus, to_bus), ...].
    Returns:
        dict: A dictionary mapping each original connection tuple to a dictionary
              containing the pandapower element 'type' and 'index'.
              Returns 'not_found' if a connection doesn't exist in the network.
    """
    # --- Step 1: Create efficient lookups for lines and transformers ---
    # We use sorted tuples as keys to handle bidirectional connections (e.g., (1, 2) is the same as (2, 1)).
    # Create a lookup dictionary for lines: { (sorted_bus_tuple): line_index }
    line_lookup = {
        tuple(sorted((row.from_bus, row.to_bus))): idx
        for idx, row in net.line.iterrows()
    }
    # Create a lookup dictionary for transformers: { (sorted_bus_tuple): trafo_index }
    trafo_lookup = {
        tuple(sorted((row.hv_bus, row.lv_bus))): idx
        for idx, row in net.trafo.iterrows()
    }
    # --- Step 2: Iterate through the original connections and build the mapping ---
    mapping = {}
    for i, (from_bus, to_bus) in enumerate(original_connections):
        # Create a sorted key for the current connection to check against our lookups
        key = tuple(sorted((from_bus, to_bus)))
        # original_tuple = (from_bus, to_bus)

        if key in line_lookup:
            mapping[i] = {
                'type': 'line',
                'index': line_lookup[key]
            }
        elif key in trafo_lookup:
            mapping[i] = {
                'type': 'trafo',
                'index': trafo_lookup[key]
            }
        else:
            mapping[i] = {
                'type': 'not_found',
                'index': None
            }
            print("link pair not found")
            
    return mapping

# ---------------------------------------------------------------------------
# Optional: small self-test / demo
# ---------------------------------------------------------------------------

def _demo():
    """Run a tiny demo on IEEE 14 to show usage."""
    import pandapower.networks as nw

    net = nw.case30()

    # Build some synthetic measurements from a power flow result for demo
    pp.runpp(net)
    # Use a subset as "measurements"
    bus_inj = []
    for b in list(range(14)):
        p = float(net.res_bus.p_mw.loc[b]) if "p_mw" in net.res_bus else None
        q = float(net.res_bus.q_mvar.loc[b]) if "q_mvar" in net.res_bus else None
        bus_inj.append([b, p, q])

    branch = []
    # For a few lines, take from-end flows
    print(net.res_bus)
    for line_idx in list(range(41)):
        p_from = float(net.res_line.p_from_mw.loc[line_idx])
        q_from = float(net.res_line.q_from_mvar.loc[line_idx])
        # no to-end provided in this quick demo
        branch.append([line_idx, p_from, q_from, None, None])

    result = estimate_state_and_residuals(
        net,
        bus_injections=np.array(bus_inj, dtype=float),
        branch_flows=np.array(branch, dtype=object),
        std_p=0.02,
        std_q=0.02,
        sigma_threshold=3.0,
        init="flat"
    )

    print("Converged:", result["state"]["converged"])
    print("Iterations:", result["state"]["iterations"])
    print("First 5 Vm, Va:")
    print(result["state"]["vm_pu"])
    print(result["state"]["va_degree"])
    print("\nResiduals head:")
    print(result["residuals"].head())
    print("\nSuspicious flags count:", result["flags"].sum())




if __name__ == "__main__":
    _demo()
