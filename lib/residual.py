import numpy as np
import lib.cycle_space as csp

def cycle_residual(trainDataNp, rankH, m, testData, cycleList= None, sigma= None):
    predParas, cSpace, wholeRight= csp.get_cycle_space(trainDataNp, rankH, m, cycleList= cycleList)
    # cProjection= cSpace@np.linalg.inv(cSpace.T@cSpace)@cSpace.T #cSpace@np.linalg.inv(cSpace.T@cSpace)@cSpace.T; cSpace.T
    cProjection= cSpace.T
    ErrorC= cProjection@(testData.T)
    if sigma is not None:
        pass
    return ErrorC, cSpace, wholeRight

def svd_residual(trainDataNp, rankH, testData, wholeRight= None):
    pfDatanor2= testData#/np.linalg.norm(testData, 2, axis=1, keepdims= True)
    pfDataSdr2= pfDatanor2 #- np.mean(pfDatanor2, 0, keepdims= True)
    if wholeRight is None:
        wholeRight= trainDataNp.T.dot(trainDataNp)
    m= wholeRight.shape[0]
    eigvals, eigvecs = np.linalg.eigh(wholeRight)
    nullH = eigvecs[:, :m-rankH]
    # hProjection= nullH@np.linalg.inv(nullH.T@nullH)@nullH.T
    ErrorH= nullH.T@(pfDataSdr2.T)
    residualH= np.linalg.norm(ErrorH, axis= 0)
    return residualH, nullH

def true_residual(H, m, testData):
    TPinv= np.linalg.pinv(H)
    testT= np.eye(m)-H@TPinv
    ErrorT= testT@(testData.T)
    residualT= np.linalg.norm(ErrorT, axis= 0)
    return residualT


def covariance_fdia_basis(trainData, targetRank):
    """Top-r eigenvectors of the centered, unscaled sample covariance."""
    data = np.asarray(trainData, dtype=float)
    if (
        data.ndim != 2 or data.shape[0] < 2
        or int(targetRank) != targetRank
        or not 1 <= targetRank <= min(data.shape[1], data.shape[0] - 1)
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("Invalid covariance-basis input")
    centered = data - data.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(data) - 1)
    eigenvectors = np.linalg.eigh(covariance)[1]
    return eigenvectors[:, -int(targetRank):][:, ::-1]


def linear_autoencoder_weights(trainData, latentDim):
    """Solve Zc -> Zc @ encoder @ decoder with weights only."""
    data = np.asarray(trainData, dtype=float)
    if (
        data.ndim != 2 or data.shape[0] < 2
        or int(latentDim) != latentDim
        or not 1 <= latentDim <= min(data.shape[1], data.shape[0] - 1)
        or not np.all(np.isfinite(data))
    ):
        raise ValueError("Invalid linear-autoencoder input")
    centered = data - data.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered) <= np.finfo(float).eps:
        raise ValueError("Autoencoder requires nonconstant training data")
    decoder = np.linalg.svd(centered, full_matrices=False)[2][:int(latentDim)]
    encoder = decoder.T.copy()
    return encoder, decoder


def noise_sensitivity_accuracy(
    H, cleanTrain, cycleListTrue, noiseLevels=None, numTrials=10,
    seed=0, treeMethod="robust", cycleMethod="new", autoencoderRank=None,
    verbose=False
):
    """Monte Carlo overlap with the true detector space Null(H.T)."""
    H, data = np.asarray(H, dtype=float), np.asarray(cleanTrain, dtype=float)
    levels = np.logspace(-3, -1, 9) if noiseLevels is None else np.asarray(
        noiseLevels, dtype=float
    )
    m = H.shape[0] if H.ndim == 2 else 0
    rankH = np.linalg.matrix_rank(H) if H.ndim == 2 else 0
    autoencoderRank = H.shape[1]-1 if autoencoderRank is None and H.ndim == 2 else autoencoderRank
    if (
        H.ndim != 2 or data.ndim != 2 or data.shape[1] != m
        or data.shape[0] < 2 or not np.all(np.isfinite(H))
        or not np.all(np.isfinite(data)) or levels.ndim != 1
        or not len(levels) or not np.all(np.isfinite(levels))
        or np.any(levels <= 0) or int(numTrials) != numTrials
        or numTrials < 1 or rankH < 1 or rankH >= m
        or int(autoencoderRank) != autoencoderRank
        or not 1 <= autoencoderRank <= min(m, data.shape[0] - 1)
        or rankH > min(m, data.shape[0] - 1)
    ):
        raise ValueError("Invalid noise-sensitivity input")

    trueNull = np.linalg.svd(H, full_matrices=True)[0][:, rankH:]

    def accuracy(nullBasis):
        basisRank = np.linalg.matrix_rank(nullBasis)
        if basisRank < 1:
            return 0.0
        q = np.linalg.svd(
            nullBasis, full_matrices=False
        )[0][:, :basisRank]
        overlap = np.linalg.norm(trueNull.T @ q, "fro") ** 2
        return float(np.clip(overlap / trueNull.shape[1], 0.0, 1.0))

    def cycleNull(noisy, cycles):
        constraints = csp.fit_cycle_constraints(noisy, cycles)
        if np.linalg.matrix_rank(constraints) != trueNull.shape[1]:
            raise ValueError("Cycle constraints have an unexpected rank")
        return constraints

    def signalNull(basis):
        basisRank = np.linalg.matrix_rank(basis)
        return np.linalg.svd(
            basis, full_matrices=True
        )[0][:, basisRank:]

    names = (
        "H known", "cycleListTrue", "cycleList", "Covariance eigenspace",
        "PCA", "Linear autoencoder"
    )
    scores = {
        name: np.empty((len(levels), int(numTrials))) for name in names
    }
    scores["H known"].fill(1.0)
    rms = np.maximum(np.sqrt(np.mean(data**2, axis=0)), 1e-8)
    standardNoise = np.random.default_rng(seed).standard_normal(
        (int(numTrials), *data.shape)
    )

    for levelId, level in enumerate(levels):
        for trial in range(int(numTrials)):
            noisy = data + level * rms * standardNoise[trial]
            scores["cycleListTrue"][levelId, trial] = accuracy(
                cycleNull(noisy, cycleListTrue)
            )
            try:
                tree, _ = csp.get_spanning_tree(
                    noisy, rankH, m, method=treeMethod
                )
                cycles, _, _ = csp.get_cycle_basis(
                    noisy, H.shape[1], tree, method=cycleMethod
                )
                scores["cycleList"][levelId, trial] = accuracy(
                    cycleNull(noisy, cycles)
                )
            except (ValueError, np.linalg.LinAlgError):
                scores["cycleList"][levelId, trial] = 0.0

            scores["Covariance eigenspace"][levelId, trial] = accuracy(
                signalNull(covariance_fdia_basis(noisy, rankH))
            )
            scale = np.std(noisy, axis=0, ddof=1)
            normalized = (noisy - noisy.mean(axis=0)) / scale
            pcaBasis = scale[:, None] * np.linalg.svd(
                normalized.T, full_matrices=False
            )[0][:, :rankH]
            scores["PCA"][levelId, trial] = accuracy(signalNull(pcaBasis))
            _, decoder = linear_autoencoder_weights(noisy, autoencoderRank)
            scores["Linear autoencoder"][levelId, trial] = accuracy(
                signalNull(decoder.T)
            )

        if verbose:
            summary = ", ".join(
                f"{name}={100 * scores[name][levelId].mean():.3f}%"
                for name in names
            )
            print(f"noise={level:.3g}: {summary}")
    return levels, scores


def paper_fdia_basis(trainData, extension=4, targetRank=None):
    """Yang-Wang basis; targetRank forces an equal-rank diagnostic basis."""
    data = np.asarray(trainData, dtype=float)
    if (
        data.ndim != 2 or data.shape[0] < 2 or extension < 1
        or int(extension) != extension
    ):
        raise ValueError("Invalid paper-method input")
    extension = int(extension)
    scale = np.std(data, axis=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("Paper method requires nonconstant measurements")

    z = ((data - data.mean(axis=0)) / scale).T
    m = z.shape[0]
    zExtended = z
    if extension > 1:
        zExtended = np.tile(z, (extension, 1))
        zExtended += np.eye(zExtended.shape[0], zExtended.shape[1])
    rows, columns = zExtended.shape
    c = columns / rows
    u, singular, vt = np.linalg.svd(
        zExtended / np.sqrt(rows), full_matrices=False
    )
    if targetRank is not None:
        if int(targetRank) != targetRank or not 1 <= targetRank <= m:
            raise ValueError("Invalid target rank")
        return scale[:, None] * u[:m, :int(targetRank)]

    discriminant = (singular**2 - 1 - c)**2 - 4*c
    tolerance = np.finfo(float).eps * max(rows, columns) * singular[0]**4
    keep = (singular > c**0.25) & (discriminant > tolerance)
    if not np.any(keep):
        raise ValueError("Paper method retained no singular vectors")
    lam = singular[keep]
    omega2 = (
        lam**2 - 1 - c + np.sqrt(np.maximum(discriminant[keep], 0))
    ) / 2
    omega = np.sqrt(omega2)
    gamma = (omega**4 - c) / (
        omega * np.sqrt((omega2 + c) * (omega2 + 1))
    )
    cleaned = ((u[:, keep] * gamma) @ vt[keep])[:m]
    cleanedU = np.linalg.svd(cleaned, full_matrices=False)[0]
    return scale[:, None] * cleanedU[:, :np.count_nonzero(keep)]


def chi2_attack_success_benchmark(
    H, trainData, testData, cycleListTrue, cycleList, noiseStd,
    numTrials=1000, attackStrength=8.0, numThresholds=25, seed=0,
    paperExtension=4, returnMagnitudes=False, cycleAttackRank=3,
    proposedRank=None, autoencoderRank=None
):
    """Compare attacks using the raw residual-energy BDD in paper Fig. 4."""
    H, trainData, testData = map(np.asarray, (H, trainData, testData))
    m, rankH = H.shape[0], np.linalg.matrix_rank(H)
    if autoencoderRank is None:
        autoencoderRank = H.shape[1]-1 if H.ndim == 2 else 0
    sigma = np.asarray(noiseStd, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(m, sigma)
    if (
        H.ndim != 2 or trainData.ndim != 2 or testData.ndim != 2
        or trainData.shape[0] < 2 or testData.shape[0] < 1
        or trainData.shape[1] != m or testData.shape[1] != m
        or sigma.shape != (m,) or np.any(sigma <= 0)
        or numTrials < 1 or numThresholds < 2 or attackStrength <= 0
        or int(cycleAttackRank) != cycleAttackRank
        or not 1 <= cycleAttackRank <= rankH
        or int(autoencoderRank) != autoencoderRank
        or not 1 <= autoencoderRank <= min(m, trainData.shape[0] - 1)
        or (
            proposedRank is not None
            and (
                int(proposedRank) != proposedRank
                or not 1 <= proposedRank <= rankH
            )
        )
    ):
        raise ValueError("Invalid benchmark input")

    knownBasis = np.linalg.svd(H, full_matrices=False)[0][:, :rankH]

    def cycleBasis(cycles):
        _, constraints, _ = cycle_residual(
            trainData, rankH, m, trainData, cycles
        )
        constraintRank = np.linalg.matrix_rank(constraints)
        nullBasis = np.linalg.svd(
            constraints.T, full_matrices=True
        )[2][constraintRank:].T
        if nullBasis.shape[1] != rankH:
            raise ValueError("cycle list does not define the expected subspace")
        right = np.linalg.svd(
            trainData @ nullBasis, full_matrices=False
        )[2]
        return nullBasis @ right[-int(cycleAttackRank):].T

    scale = np.std(trainData, axis=0, ddof=1)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("PCA requires nonconstant measurement rows")
    normalized = (trainData - trainData.mean(axis=0)) / scale
    pcaBasis = scale[:, None] * np.linalg.svd(
        normalized.T, full_matrices=False
    )[0][:, :rankH]
    covarianceBasis = covariance_fdia_basis(trainData, rankH)
    _, autoencoderDecoder = linear_autoencoder_weights(
        trainData, autoencoderRank
    )
    proposedBasis = paper_fdia_basis(
        trainData, paperExtension, targetRank=proposedRank
    )
    physicalBases = {
        "H known": knownBasis,
        "cycleListTrue": cycleBasis(cycleListTrue),
        "cycleList": cycleBasis(cycleList),
        "Covariance eigenspace": covarianceBasis,
        "PCA": pcaBasis,
        "Linear autoencoder": autoencoderDecoder.T,
        "Proposed (paper)": proposedBasis,
    }

    whitenedH = H / sigma[:, None]
    detectorVectors = np.linalg.svd(
        whitenedH, full_matrices=True
    )[0][:, rankH:]
    stateEstimator = np.linalg.pinv(whitenedH)

    def whitenedBasis(basis):
        basis = basis / sigma[:, None]
        basisRank = np.linalg.matrix_rank(basis)
        if basisRank < 1:
            raise ValueError("attack basis is rank deficient")
        return np.linalg.qr(basis, mode="reduced")[0][:, :basisRank]

    attackBases = {
        name: whitenedBasis(basis) for name, basis in physicalBases.items()
    }
    cleanTest = testData @ knownBasis @ knownBasis.T
    rng = np.random.default_rng(seed)
    rows = rng.integers(testData.shape[0], size=numTrials)
    clean = cleanTest[rows] / sigma
    noise = rng.standard_normal((numTrials, m))
    directions = rng.standard_normal((numTrials, m))

    statistics, magnitudes = {}, {}
    for name, basis in attackBases.items():
        coefficients = directions @ basis
        coefficients /= np.linalg.norm(coefficients, axis=1, keepdims=True)
        attack = attackStrength * coefficients @ basis.T
        stateChange = attack @ stateEstimator.T
        magnitudes[name] = {
            "rawVector": attack * sigma,
            "whitenedVector": attack,
            "raw": np.linalg.norm(attack * sigma, axis=1),
            "whitened": np.linalg.norm(attack, axis=1),
            "stateL2": np.linalg.norm(stateChange, axis=1),
            "stateMax": np.max(np.abs(stateChange), axis=1),
        }
        residual = (clean + noise + attack) @ detectorVectors
        residualRaw = residual @ detectorVectors.T * sigma
        statistics[name] = np.square(residualRaw).sum(axis=1)

    upper = np.quantile(np.concatenate(list(statistics.values())), 0.995)
    thresholds = np.linspace(0.0, upper, numThresholds)
    success = {
        name: (values[:, None] <= thresholds).mean(axis=0)
        for name, values in statistics.items()
    }
    if returnMagnitudes:
        return thresholds, success, magnitudes
    return thresholds, success
