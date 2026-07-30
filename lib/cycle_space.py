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

def get_spanning_tree(dataZ, rankH, m):
    spaningTreeList= select_rows_rrqr(dataZ, rank= rankH)
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


def get_cycle_basis(dataZ, n, spaningTreeList, method="new"):
    """Select the old or new fundamental-cycle recovery method."""
    method = method.lower()
    if method == "old":
        m = dataZ.shape[1]
        leftLinksSet = set(range(m)) - set(spaningTreeList)
        wholeRight = dataZ.T.dot(dataZ)
        return get_cycle_and_tree(
            spaningTreeList, leftLinksSet, wholeRight, m
        )
    if method == "new":
        return get_residual_criterion_cycle_and_tree(
            dataZ, n, spaningTreeList
        )
    raise ValueError("method must be 'old' or 'new'")


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


def weighted_cycle_accuracies(H, dataZ, cycleList):
    """Return each learned cycle vector's squared accuracy in null(H.T)."""
    H, dataZ = np.asarray(H), np.asarray(dataZ)
    if H.ndim != 2 or dataZ.ndim != 2 or dataZ.shape[1] != H.shape[0]:
        raise ValueError("H and dataZ have incompatible shapes")
    rankH = np.linalg.matrix_rank(H)
    trueSpace = np.linalg.svd(H, full_matrices=True)[0][:, rankH:]
    covariance = dataZ.T @ dataZ
    accuracies = []
    for cycle in cycleList:
        links = list(cycle)
        if not links or any(i < 0 or i >= H.shape[0] for i in links):
            accuracies.append(np.nan)
            continue
        weights = np.linalg.svd(
            covariance[np.ix_(links, links)], full_matrices=False
        )[2][-1]
        vector = np.zeros(H.shape[0])
        vector[links] = weights
        accuracy = np.linalg.norm(trueSpace.T @ vector) ** 2 / np.dot(vector, vector)
        accuracies.append(np.clip(accuracy, 0.0, 1.0))
    return np.asarray(accuracies)


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
            plt.plot(critias[cycle_id], label= f"{cycle_id}")
        plt.legend()
        plt.yscale("log")
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
        spaningTreeList, _ = get_spanning_tree(noisyTrain, rankH, m)
        cycleList, treeLinks, critias = (
            get_residual_criterion_cycle_and_tree(
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
    """Compare old and new cycle recovery using identical noisy trees."""
    if numTests < 1:
        raise ValueError("numTests must be positive")

    m = A.shape[0]
    trainData = dataZ[:numTrain, :]
    topologyRankCorrect = np.linalg.matrix_rank(A) == rankH
    expectedTreeLinks = get_bridge_links(A)
    spanningSuccess = 0
    success = {
        "Old curvature heuristic": {"treeLinks": 0, "cycleList": 0},
        "New residual-curvature criterion": {"treeLinks": 0, "cycleList": 0},
    }

    for _ in range(numTests):
        noisyTrain = trainData + np.random.randn(*trainData.shape) * noiseStd
        wholeRight = noisyTrain.T.dot(noisyTrain)
        spaningTreeList, leftLinksSet = get_spanning_tree(
            noisyTrain, rankH, m
        )
        oldCycles, oldTreeLinks, oldCritias = get_cycle_and_tree(
            spaningTreeList, leftLinksSet, wholeRight, m
        )
        newCycles, newTreeLinks, newCritias = (
            get_residual_criterion_cycle_and_tree(
                noisyTrain, rankH + 1, spaningTreeList
            )
        )

        spanningSuccess += check_spanning_tree(
            A, spaningTreeList, rankH, verbose=False
        )
        for method, cycleList, treeLinks, critias in (
            ("Old curvature heuristic", oldCycles, oldTreeLinks, oldCritias),
            (
                "New residual-curvature criterion",
                newCycles,
                newTreeLinks,
                newCritias,
            ),
        ):
            success[method]["treeLinks"] += (
                topologyRankCorrect and set(treeLinks) == expectedTreeLinks
            )
            success[method]["cycleList"] += check_cycle_list(
                A, cycleList, critias, verbose=False
            )

    spanningRate = spanningSuccess / numTests
    print(
        f"spaningTreeList success rate: "
        f"{spanningSuccess}/{numTests} ({100 * spanningRate:.1f}%)"
    )
    rates = {"spaningTreeList": spanningRate}
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
        spaningTreeList, leftLinksSet = get_spanning_tree(newDataNp, rankH, m)
        cycleList, _, _ = get_cycle_and_tree(
            spaningTreeList, leftLinksSet, wholeRight, m
        )
    biComponents= find_components(cycleList.copy())
    predParas, cSpace= parasLearn.paras_learning(biComponents, wholeRight)

    # print(cycleList)
    return predParas, cSpace, wholeRight
