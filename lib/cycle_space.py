from scipy.linalg import qr
import numpy as np
import networkx as nx
import lib.parasLearn as parasLearn
import matplotlib.pyplot as plt


def select_rows_rrqr(H_e, rank=13):
    _, _, pivot = qr(H_e, pivoting=True)
    selected_rows = pivot[:rank]
    return selected_rows.tolist()

def linksDependent(covM):
    Val, Vt = np.linalg.eig(covM)
    Val= np.abs(np.real(Val))
    minIdx= np.argmin(Val)
    nullSpace= Vt[:, minIdx:minIdx+1].T
    nullMagni= np.linalg.norm(nullSpace, axis= 0)
    return nullMagni

def _rms_normalize(dataZ):
    """Equalize relative branch noise without discarding the DC mean."""
    dataZ = np.asarray(dataZ, dtype=float)
    if dataZ.ndim != 2 or not np.all(np.isfinite(dataZ)):
        raise ValueError("dataZ must be a finite two-dimensional matrix")
    scale = np.sqrt(np.mean(dataZ**2, axis=0))
    if not len(scale) or np.max(scale) == 0 or np.any(
        scale <= np.max(scale) * np.finfo(float).eps * max(dataZ.shape)
    ):
        raise ValueError("Every measured branch must have nonzero RMS")
    return dataZ / scale, scale


def get_spanning_tree(dataZ, rankH, m, method="raw"):
    """Select measurement coordinates using raw or rank-truncated RRQR."""
    dataZ = np.asarray(dataZ)
    if dataZ.ndim != 2 or dataZ.shape[1] != m or not 1 <= rankH <= m:
        raise ValueError("Invalid dataZ, rankH, or m")
    method = method.lower()
    if method == "raw":
        spaningTreeList = select_rows_rrqr(dataZ, rank=rankH)
    elif method in {"robust", "svd"}:
        normalized, _ = _rms_normalize(dataZ)
        if rankH > min(normalized.shape):
            raise ValueError("Not enough measurements for the requested rank")
        signalSpace = np.linalg.svd(
            normalized, full_matrices=False
        )[2][:rankH].T
        spaningTreeList = select_rows_rrqr(signalSpace.T, rank=rankH)
    else:
        raise ValueError("method must be 'raw' or 'robust'")
    leftLinksSet= set([i for i in range(m)])-set(spaningTreeList)
    return spaningTreeList, leftLinksSet


def check_spanning_tree(A, spaningTreeList, rankH, verbose=True):
    """Return whether the selected links form a spanning tree."""
    links = list(spaningTreeList)
    valid_ids = (
        len(links) == len(set(links))
        and all(isinstance(i, (int, np.integer)) and 0 <= i < A.shape[0] for i in links)
    )
    correct = (
        valid_ids
        and np.linalg.matrix_rank(A) == rankH
        and len(links) == rankH
        and np.linalg.matrix_rank(A[links, :]) == rankH
    )
    if verbose:
        print("spaningTreeList correct:", correct)
    return correct

def get_cycle_and_tree(spaningTreeList, leftLinksSet, wholeRight, m):
    cycleList= []
    leftLinksList= list(leftLinksSet)
    treeLinks= set([i for i in range(m)])# links in a tree structure, not belong to any cycle
    Sss= []
    logSss= []
    critias= []
    for i, link in  enumerate(leftLinksList):
        tmpLinks= spaningTreeList+[link]
        covM= wholeRight[tmpLinks, :][:, tmpLinks]
        nullMagni= linksDependent(covM)
        rankedIdx= np.argsort(nullMagni)
        convRankedIdx= [tmpLinks[i] for i in reversed(rankedIdx)]
        Ss= np.zeros(len(convRankedIdx))# used to be -1
        for j in range(1, len(convRankedIdx)+1): # used to be +0
            squareRight= wholeRight[:, convRankedIdx[:j]][convRankedIdx[:j], :]# right covariance matrix
            singularVs = np.sqrt(np.abs(np.linalg.eigvals(squareRight)))# transform eigenvalue to sigular value 
            Ss[j-1]= np.min(singularVs)
        Sss.append(Ss)
        logSs= np.log(Ss+1e-9)
        logSss.append(logSs)
        critia= (logSs[:-2]+logSs[2:]-2*logSs[1:-1])/(Ss[1:-1]+1e-9)
        maxId= max(np.argmax(critia)+2, convRankedIdx.index(link)+1)
        # print(convRankedIdx)
        # print(convRankedIdx.index(link)+1, maxId)
        cycleTmp= convRankedIdx[:maxId]
        treeLinks= treeLinks-set(cycleTmp)
        cycleList.append(set(cycleTmp))
        critias.append(Ss)
    return cycleList, treeLinks, critias


def _current_tree_orderings(Z, TG, chords):
    """Build the old method's null-magnitude tree-edge orderings."""
    wholeRight = Z.dot(Z.T)
    orderings = {}
    for chord in chords:
        links = list(TG) + [chord]
        covM = wholeRight[np.ix_(links, links)]
        ranked = np.argsort(linksDependent(covM))[::-1]
        orderings[chord] = [links[i] for i in ranked if links[i] != chord]
    return orderings


def recover_residual_criterion_cycles(Z, TG, E=None):
    """Choose the elbow of each ordered parameter-residual curve."""
    Z = np.asarray(Z)
    treeList = list(TG)
    treeSet = set(treeList)
    if Z.ndim != 2:
        raise ValueError("Z must be a two-dimensional edge-by-time matrix")

    m, numPoints = Z.shape
    n = len(treeList) + 1
    edgeSet = set(range(m)) if E is None else set(E)
    validEdges = all(
        isinstance(i, (int, np.integer)) and 0 <= i < m
        for i in edgeSet | treeSet
    )
    if (
        len(treeList) != len(treeSet)
        or n < 3
        or numPoints < n
        or not treeSet <= edgeSet
        or not validEdges
    ):
        raise ValueError("Invalid Z, TG, or E")

    chords = sorted(edgeSet - treeSet)
    if not chords:
        return [], []
    orderings = _current_tree_orderings(Z, treeList, chords)
    cycleList, critias = [], []
    for chord in chords:
        candidates, residuals = [], []
        for k in range(2, n + 1):
            links = [chord] + orderings[chord][:k - 1]
            left = np.linalg.svd(Z[links, :], full_matrices=False)[0][:, -1]
            parameter = np.zeros(m)
            parameter[links] = left
            candidates.append(set(links))
            residuals.append(np.linalg.norm(Z.T.dot(parameter)))
        residuals = np.asarray(residuals)
        logResiduals = np.log(residuals + 1e-9)
        criteria = (
            logResiduals[:-2] + logResiduals[2:] - 2 * logResiduals[1:-1]
        ) / (residuals[1:-1] + 1e-9)
        best = np.argmax(criteria) + 1 if len(criteria) else len(candidates) - 1
        cycleList.append(candidates[best])
        critias.append(criteria)
    return cycleList, critias


def get_residual_criterion_cycle_and_tree(dataZ, n, spaningTreeList):
    """Adapt the training-residual criterion to time-by-edge data."""
    if len(spaningTreeList) != n - 1:
        raise ValueError("spaningTreeList must contain n-1 edges")
    cycleList, critias = recover_residual_criterion_cycles(
        dataZ.T, spaningTreeList, E=range(dataZ.shape[1])
    )
    cycleEdges = set().union(*cycleList) if cycleList else set()
    treeLinks = set(range(dataZ.shape[1])) - cycleEdges
    return cycleList, treeLinks, critias


def get_adaptive_cycle_and_tree(
    dataZ, n, spaningTreeList, supportThreshold=1e-2
):
    """Recover BIC cycles, prune weak fitted links, and refit downstream."""
    from sklearn.linear_model import LassoLarsIC

    dataZ = np.asarray(dataZ, dtype=float)
    tree = sorted(spaningTreeList)
    m = dataZ.shape[1] if dataZ.ndim == 2 else 0
    if (
        dataZ.ndim != 2
        or len(tree) != n - 1
        or len(tree) != len(set(tree))
        or dataZ.shape[0] <= len(tree) + 1
        or any(i < 0 or i >= m for i in tree)
    ):
        raise ValueError("Invalid dataZ or spaningTreeList")

    normalized, _ = _rms_normalize(dataZ)
    predictors = normalized[:, tree]
    cycles, critias = [], []
    for chord in sorted(set(range(m)) - set(tree)):
        model = LassoLarsIC(
            criterion="bic", fit_intercept=False
        ).fit(predictors, normalized[:, chord])
        selected = np.flatnonzero(model.coef_)
        if len(selected) < 2:
            coefficients = np.linalg.lstsq(
                predictors, normalized[:, chord], rcond=None
            )[0]
            selected = np.argsort(
                -np.abs(coefficients), kind="stable"
            )[:2]

        treeEdges = [tree[i] for i in selected]
        full = [chord] + treeEdges
        weights = np.linalg.svd(
            normalized[:, full], full_matrices=False
        )[2][-1, 1:]
        order = np.lexsort((np.asarray(treeEdges), -np.abs(weights)))

        candidates = []
        for count in range(2, len(treeEdges) + 1):
            links = [chord] + [treeEdges[i] for i in order[:count]]
            residual = np.linalg.svd(
                normalized[:, links], compute_uv=False
            )[-1] ** 2
            bic = (
                len(dataZ) * np.log(
                    max(residual / len(dataZ), np.finfo(float).tiny)
                )
                + len(links) * np.log(len(dataZ))
            )
            candidates.append((bic, tuple(sorted(links)), links))

        _, _, best = min(candidates, key=lambda item: item[:2])
        cycles.append(set(best))
        scores = np.asarray([item[0] for item in candidates])
        critias.append(scores - scores.min() + 1.0)

    if supportThreshold is not None:
        cycles = refine_cycle_list(dataZ, cycles, supportThreshold)
    cycleEdges = set().union(*cycles) if cycles else set()
    return cycles, set(range(m)) - cycleEdges, critias


def get_cycle_basis(dataZ, n, spaningTreeList, method="new"):
    """Select a fundamental-cycle recovery method."""
    method = method.lower()
    if method == "old":
        m = dataZ.shape[1]
        leftLinksSet = set(range(m)) - set(spaningTreeList)
        wholeRight = dataZ.T.dot(dataZ)
        return get_cycle_and_tree(
            spaningTreeList, leftLinksSet, wholeRight, m
        )
    if method in {"new", "adaptive", "bic"}:
        return get_adaptive_cycle_and_tree(
            dataZ, n, spaningTreeList
        )
    if method == "residual":
        return get_residual_criterion_cycle_and_tree(
            dataZ, n, spaningTreeList
        )
    raise ValueError(
        "method must be 'old', 'new'/'bic', or 'residual'"
    )


def recover_cycle_list(
    dataZ, n, treeMethod="robust", cycleMethod="new"
):
    """Recover a tree and cycle list from branch measurements and n only."""
    dataZ = np.asarray(dataZ)
    if dataZ.ndim != 2 or n < 2:
        raise ValueError("Invalid dataZ or n")
    tree, _ = get_spanning_tree(
        dataZ, n - 1, dataZ.shape[1], method=treeMethod
    )
    cycles, treeLinks, critias = get_cycle_basis(
        dataZ, n, tree, method=cycleMethod
    )
    return tree, cycles, treeLinks, critias


def rank_gf2(matrix):
    """Compute matrix rank over GF(2)."""
    work = (np.asarray(matrix, dtype=np.uint8) % 2).copy()
    if work.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rank = 0
    for col in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + pivots[0]
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rank + 1, work.shape[0]):
            if work[row, col]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def minimum_cycle_basis_from_fundamental(BF, w=None):
    """Return the exact weighted MCB in the span of a fundamental basis."""
    rawBF = np.asarray(BF)
    if rawBF.ndim != 2 or not np.all((rawBF == 0) | (rawBF == 1)):
        raise ValueError("BF must be a binary two-dimensional matrix")
    BF = rawBF.astype(np.uint8)
    m, mu = BF.shape
    weights = np.ones(m) if w is None else np.asarray(w, dtype=float)
    if (
        weights.shape != (m,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
        or rank_gf2(BF) != mu
    ):
        raise ValueError("w must be positive and BF must have full GF(2) rank")

    candidates = []
    for mask in range(1, 1 << mu):
        a = np.array([(mask >> i) & 1 for i in range(mu)], dtype=np.uint8)
        cycle = (BF @ a) % 2
        candidates.append((
            float(weights @ cycle), int(cycle.sum()), tuple(a), a, cycle
        ))
    candidates.sort(key=lambda item: item[:3])

    transform = np.empty((mu, 0), dtype=np.uint8)
    minimumBasis = np.empty((m, 0), dtype=np.uint8)
    for _, _, _, coefficient, cycle in candidates:
        trial = np.column_stack((transform, coefficient))
        if rank_gf2(trial) > transform.shape[1]:
            transform = trial
            minimumBasis = np.column_stack((minimumBasis, cycle))
            if transform.shape[1] == mu:
                break
    return minimumBasis, transform


def cycle_list_to_matrix(cycleList, m):
    """Convert edge-index sets to an m-by-mu binary cycle matrix."""
    basis = np.zeros((m, len(cycleList)), dtype=np.uint8)
    for column, cycle in enumerate(cycleList):
        links = list(cycle)
        if (
            len(links) != len(set(links))
            or not all(
                isinstance(link, (int, np.integer)) and 0 <= link < m
                for link in links
            )
        ):
            raise ValueError("cycleList contains an invalid edge index")
        basis[links, column] = 1
    return basis


def cycle_matrix_to_list(basis):
    """Convert an m-by-mu binary cycle matrix to edge-index sets."""
    basis = np.asarray(basis)
    if basis.ndim != 2 or not np.all((basis == 0) | (basis == 1)):
        raise ValueError("basis must be a binary two-dimensional matrix")
    return [
        set(np.flatnonzero(basis[:, column]))
        for column in range(basis.shape[1])
    ]


def minimum_topology_cycle_basis(A):
    """Return an unweighted MCB from incidence rows, preserving parallel edges."""
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    groups = {}
    for edge, row in enumerate(A):
        nodes = np.flatnonzero(row)
        if len(nodes) != 2:
            raise ValueError("Each row of A must contain two edge endpoints")
        pair = tuple(sorted(map(int, nodes)))
        groups.setdefault(pair, []).append(edge)

    G = nx.Graph()
    G.add_nodes_from(range(A.shape[1]))
    cycles = []
    for pair, edges in groups.items():
        G.add_edge(*pair, label=edges[0], weight=1)
        cycles.extend({edges[0], edge} for edge in edges[1:])

    for nodeSet in nx.minimum_cycle_basis(G, weight="weight"):
        start = min(nodeSet)

        def order(path, left):
            if not left:
                return path if G.has_edge(path[-1], start) else None
            for node in sorted(set(G[path[-1]]) & left):
                result = order(path + [node], left - {node})
                if result is not None:
                    return result

        nodes = order([start], set(nodeSet) - {start})
        if nodes is None:
            raise ValueError("Could not order a minimum-basis cycle")
        cycles.append({
            G[nodes[i - 1]][nodes[i]]["label"]
            for i in range(len(nodes))
        })

    expected = A.shape[0] - np.linalg.matrix_rank(A)
    basis = cycle_list_to_matrix(cycles, A.shape[0])
    if len(cycles) != expected or rank_gf2(basis.T) != expected:
        raise ValueError("Topology cycle basis has incorrect rank")
    return cycles


def get_bridge_links(A):
    """Return links that belong to no cycle (graph bridges)."""
    rankA = np.linalg.matrix_rank(A)
    return {
        i for i in range(A.shape[0])
        if np.linalg.matrix_rank(np.delete(A, i, axis=0)) < rankA
    }


def check_tree_links(A, rankH, treeLinks, verbose=True):
    """Return whether treeLinks contains exactly all graph bridges."""
    expected = get_bridge_links(A)
    correct = np.linalg.matrix_rank(A) == rankH and set(treeLinks) == expected
    if verbose:
        print("treeLinks correct:", correct)
    if verbose and not correct:
        print("missing:", sorted(expected - set(treeLinks)))
        print("extra:", sorted(set(treeLinks) - expected))
    return correct


def fit_cycle_constraints(dataZ, cycleList):
    """Fit normalized TLS weights and map them to raw branch coordinates."""
    normalized, scale = _rms_normalize(dataZ)
    m = normalized.shape[1]
    constraints = []
    for cycle in cycleList:
        links = sorted(cycle)
        if (
            len(links) < 2
            or len(links) != len(set(links))
            or any(
                not isinstance(i, (int, np.integer)) or i < 0 or i >= m
                for i in links
            )
            or len(links) > len(normalized)
        ):
            raise ValueError("cycleList contains an invalid cycle")
        weights = np.linalg.svd(
            normalized[:, links], full_matrices=False
        )[2][-1]
        vector = np.zeros(m)
        vector[links] = weights / scale[links]
        vector /= np.linalg.norm(vector)
        if vector[links[0]] < 0:
            vector *= -1
        constraints.append(vector)
    return (
        np.stack(constraints, axis=1)
        if constraints else np.empty((m, 0))
    )


def refine_cycle_list(dataZ, cycleList, threshold=1e-2):
    """Threshold fitted cycle weights, then validate the refitted basis."""
    if (
        not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold) or threshold <= 0
    ):
        raise ValueError("threshold must be positive and finite")
    initial = fit_cycle_constraints(dataZ, cycleList)
    refined = [
        set(np.flatnonzero(np.abs(initial[:, cycle]) > threshold))
        for cycle in range(initial.shape[1])
    ]
    final = fit_cycle_constraints(dataZ, refined)
    if refined and np.linalg.matrix_rank(final) != len(refined):
        raise ValueError("Refined cycle constraints are rank deficient")
    return refined


def weighted_cycle_accuracies(H, dataZ, cycleList):
    """Return each learned cycle vector's squared accuracy in null(H.T)."""
    H, dataZ = np.asarray(H), np.asarray(dataZ)
    if H.ndim != 2 or dataZ.ndim != 2 or dataZ.shape[1] != H.shape[0]:
        raise ValueError("H and dataZ have incompatible shapes")
    rankH = np.linalg.matrix_rank(H)
    trueSpace = np.linalg.svd(H, full_matrices=True)[0][:, rankH:]
    constraints = fit_cycle_constraints(dataZ, cycleList)
    accuracies = np.sum((trueSpace.T @ constraints) ** 2, axis=0)
    return np.clip(accuracies, 0.0, 1.0)


def check_cycle_list(A, cycleList, critias, verbose=True, H=None, dataZ=None):
    """Check that every inferred link set is exactly one graph cycle."""
    incorrect = []
    for cycle_id, cycle in enumerate(cycleList):
        links = list(cycle)
        valid_ids = (
            len(links) == len(set(links))
            and all(isinstance(i, (int, np.integer)) and 0 <= i < A.shape[0] for i in links)
        )
        correct = valid_ids and len(links) > 1
        if correct:
            cycle_rank = np.linalg.matrix_rank(A[links, :])
            correct = (
                cycle_rank == len(links) - 1
                and all(
                    np.linalg.matrix_rank(A[links[:i] + links[i + 1:], :]) == cycle_rank
                    for i in range(len(links))
                )
            )
        if not correct:
            incorrect.append((cycle_id, sorted(links)))

    if verbose:
        print("cycleList correct:", not incorrect)
        correct_count = len(cycleList) - len(incorrect)
        accuracy = correct_count / len(cycleList) if cycleList else 0.0
        print(
            f"cycleList accuracy: {correct_count}/{len(cycleList)} "
            f"({100 * accuracy:.1f}%)"
        )
        if H is not None or dataZ is not None:
            if H is None or dataZ is None:
                raise ValueError("Both H and dataZ are required for weighted accuracy")
            for cycle_id, accuracy in enumerate(
                weighted_cycle_accuracies(H, dataZ, cycleList)
            ):
                print(
                    f"weighted cycleList[{cycle_id}] accuracy: "
                    f"{100 * accuracy:.2f}%"
                )
        for cycle_id, links in incorrect:
            print(f"incorrect cycleList[{cycle_id}] links:", links)
            # plt.plot(critias[cycle_id], label= f"{cycle_id}")
        if incorrect:
            # plt.legend()
            # plt.yscale("log")
            pass
    return not incorrect


def statistical_cycle_test(
    dataZ, A, rankH, numTrain, noiseStd, numTests=100
):
    """Run repeated noisy cycle-learning tests and print success rates."""
    if numTests < 1:
        raise ValueError("numTests must be positive")

    m = A.shape[0]
    trainData = dataZ[:numTrain, :]
    topologyRankCorrect = np.linalg.matrix_rank(A) == rankH
    expectedTreeLinks = get_bridge_links(A)
    success = {"spaningTreeList": 0, "treeLinks": 0, "cycleList": 0}

    for _ in range(numTests):
        noisyTrain = trainData + np.random.randn(*trainData.shape) * noiseStd
        spaningTreeList, _ = get_spanning_tree(
            noisyTrain, rankH, m, method="robust"
        )
        cycleList, treeLinks, critias = (
            get_adaptive_cycle_and_tree(
                noisyTrain, rankH + 1, spaningTreeList
            )
        )
        success["spaningTreeList"] += check_spanning_tree(
            A, spaningTreeList, rankH, verbose=False
        )
        success["treeLinks"] += topologyRankCorrect and set(treeLinks) == expectedTreeLinks
        success["cycleList"] += check_cycle_list(
            A, cycleList, critias, verbose=False
        )

    rates = {name: count / numTests for name, count in success.items()}
    for name, count in success.items():
        print(f"{name} success rate: {count}/{numTests} ({100 * rates[name]:.1f}%)")
    return rates


def compare_cycle_recovery_statistics(
    dataZ, A, rankH, numTrain, noiseStd, numTests=100
):
    """Compare the complete old and robust pipelines on identical noise."""
    if numTests < 1:
        raise ValueError("numTests must be positive")

    m = A.shape[0]
    trainData = dataZ[:numTrain, :]
    topologyRankCorrect = np.linalg.matrix_rank(A) == rankH
    expectedTreeLinks = get_bridge_links(A)
    success = {
        "Old": {"spaningTreeList": 0, "treeLinks": 0, "cycleList": 0},
        "New robust": {
            "spaningTreeList": 0, "treeLinks": 0, "cycleList": 0
        },
    }

    for _ in range(numTests):
        noisyTrain = trainData + np.random.randn(*trainData.shape) * noiseStd
        wholeRight = noisyTrain.T.dot(noisyTrain)
        oldTree, leftLinksSet = get_spanning_tree(
            noisyTrain, rankH, m, method="raw"
        )
        oldCycles, oldTreeLinks, oldCritias = get_cycle_and_tree(
            oldTree, leftLinksSet, wholeRight, m
        )
        newTree, _ = get_spanning_tree(
            noisyTrain, rankH, m, method="robust"
        )
        newCycles, newTreeLinks, newCritias = (
            get_adaptive_cycle_and_tree(
                noisyTrain, rankH + 1, newTree
            )
        )

        for method, tree, cycleList, treeLinks, critias in (
            ("Old", oldTree, oldCycles, oldTreeLinks, oldCritias),
            (
                "New robust",
                newTree,
                newCycles,
                newTreeLinks,
                newCritias,
            ),
        ):
            success[method]["spaningTreeList"] += check_spanning_tree(
                A, tree, rankH, verbose=False
            )
            success[method]["treeLinks"] += (
                topologyRankCorrect and set(treeLinks) == expectedTreeLinks
            )
            success[method]["cycleList"] += check_cycle_list(
                A, cycleList, critias, verbose=False
            )

    rates = {}
    for method, counts in success.items():
        print(f"{method}:")
        rates[method] = {
            name: count / numTests for name, count in counts.items()
        }
        for name, count in counts.items():
            print(
                f"  {name} success rate: "
                f"{count}/{numTests} ({100 * rates[method][name]:.1f}%)"
            )
    return rates


def compare_ranked_subspace_statistics(
    dataZ, H, noiseRatio=0.1, sampleMultiples=(2, 3, 4),
    numTests=100, seed=0
):
    """Evaluate recovery only; H is used solely for accuracy scoring."""
    dataZ, H = np.asarray(dataZ, dtype=float), np.asarray(H, dtype=float)
    m, n = H.shape
    rankH = np.linalg.matrix_rank(H)
    if (
        dataZ.ndim != 2 or dataZ.shape[1] != m or rankH != n - 1
        or noiseRatio <= 0 or numTests < 1
    ):
        raise ValueError("Invalid dataZ, H, noiseRatio, or numTests")
    trueSpace = np.linalg.svd(H, full_matrices=False)[0][:, :rankH]

    def accuracy(basis):
        basis = np.linalg.svd(
            basis, full_matrices=False
        )[0][:, :rankH]
        return np.linalg.norm(trueSpace.T @ basis, "fro") ** 2 / rankH

    results = {}
    for multiple in sampleMultiples:
        count = int(multiple * m)
        if multiple != int(multiple) or count > len(dataZ):
            raise ValueError("Invalid sampleMultiples")
        clean = dataZ[:count]
        rms = np.sqrt(np.mean(clean**2, axis=0))
        noiseStd = noiseRatio * np.maximum(
            rms, np.max(rms) * np.finfo(float).eps
        )
        cycleScores, pcaScores = [], []
        rankSuccess = 0
        for test in range(numTests):
            rng = np.random.default_rng(seed + test)
            noisy = clean + rng.normal(scale=noiseStd, size=clean.shape)
            tree, _ = get_spanning_tree(
                noisy, rankH, m, method="robust"
            )
            cycles, _, _ = get_cycle_basis(
                noisy, n, tree, method="new"
            )
            constraints = fit_cycle_constraints(noisy, cycles)
            constraintRank = np.linalg.matrix_rank(constraints)
            rankSuccess += constraintRank == m - rankH
            cycleBasis = np.linalg.svd(
                constraints.T, full_matrices=True
            )[2][constraintRank:].T
            cycleScores.append(accuracy(cycleBasis))

            scale = np.std(noisy, axis=0, ddof=1)
            normalized = (noisy - noisy.mean(axis=0)) / scale
            pcaBasis = scale[:, None] * np.linalg.svd(
                normalized.T, full_matrices=False
            )[0][:, :rankH]
            pcaScores.append(accuracy(pcaBasis))

        cycleScores, pcaScores = map(
            np.asarray, (cycleScores, pcaScores)
        )
        margins = cycleScores - pcaScores
        results[int(multiple)] = {
            "cycleList": cycleScores,
            "PCA": pcaScores,
            "wins": int(np.sum(margins > 0)),
            "minMargin": float(np.min(margins)),
            "rankSuccess": int(rankSuccess),
        }
        print(
            f"{int(multiple)}m: cycleList={100 * cycleScores.mean():.4f}%, "
            f"PCA={100 * pcaScores.mean():.4f}%, H known=100.0000%, "
            f"cycleList>PCA {np.sum(margins > 0)}/{numTests}, "
            f"min margin={100 * np.min(margins):.4f} pp, "
            f"rank {rankSuccess}/{numTests}"
        )
    return results


def find_components(sets):
    from collections import defaultdict, deque
    
    # Build the adjacency list based on intersections
    nCyc = len(sets)
    adj = defaultdict(set)
    
    for i in range(nCyc):
        for j in range(i+1, nCyc):
            if sets[i] & sets[j]:  # intersection is non-empty
                adj[i].add(j)
                adj[j].add(i)
    
    visited = set()
    components = []
    
    # BFS to find connected components
    for i in range(nCyc):
        if i not in visited:
            component = set()
            queue = deque([i])
            visited.add(i)
            while queue:
                cycle = queue.popleft()
                component.add(cycle)
                for neighbor in adj[cycle]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    # Return the actual sets grouped by components
    grouped_components = [[list(sets[i]) for i in component] for component in components]
    return grouped_components


def get_cycle_space(newDataNp, rankH, m, cycleList= None):
    wholeRight= newDataNp.T.dot(newDataNp)
    if cycleList== None:
        spaningTreeList, _ = get_spanning_tree(
            newDataNp, rankH, m, method="robust"
        )
        cycleList, _, _ = get_adaptive_cycle_and_tree(
            newDataNp, rankH + 1, spaningTreeList
        )
    biComponents= find_components(cycleList.copy())
    predParas, _= parasLearn.paras_learning(biComponents, wholeRight)
    cSpace= fit_cycle_constraints(newDataNp, cycleList)

    # print(cycleList)
    return predParas, cSpace, wholeRight
