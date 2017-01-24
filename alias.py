import random, math

def generateAlias(probs):
    """
    Build alias table for Walker's method. This implementation follows the "square histogram"
    version of Masaglia et al. (2004).
    """
    l = len(probs)
    A = [[0 ,0] for i in range(l)]
    L = [[0 ,0] for i in range(l)]
    H = [[0 ,0] for i in range(l)]
    lind = 0
    hind = 0
    aind = 0
    linv = 1.0/l
    updatedHighValue = 0

    # Create low and high probability values.
    for i in range(l):
        if probs[i] <= linv:
            L[lind][0],L[lind][1] = i, probs[i]
            lind += 1
        else:
            H[hind][0],H[hind][1] = i, probs[i]
            hind += 1

        A[i][0] = i
        A[i][1] = (i + 1) * linv
    
    lind -= 1
    hind -= 1

    while lind >= 0 and hind >= 0:
        aind       = L[lind][0]
        A[aind][0] = H[hind][0]
        A[aind][1] = (aind) * linv + L[lind][1]
        updatedHighValue = H[hind][1] - (linv - L[lind][1])
        lind -= 1
        hind -= 1

        if updatedHighValue > linv :
            hind += 1
            H[hind][0] = A[aind][0]
            H[hind][1] = updatedHighValue
        else:
            lind += 1
            L[lind][0] = A[aind][0]
            L[lind][1] = updatedHighValue
    
    return A

def debugAlias(A):
    l = len(A)
    linv = 1.0/l
    probs = [0]*l
    for i in range(l):
        probs[i] += A[i][1] - (linv * (i))
        probs[A[i][0]] += linv * (i+1) - A[i][1]
    return np.array(probs)

def sampleAlias(A):
    l = len(A)
    u = random.random()
    bin = int(math.floor(u*l))
    if u < A[bin][1]:
        return bin
    else:
        return A[bin][0]

def test():
    probs = np.array([0.1, 0.2, 0.2, 0.5])
    A = generateAlias(probs)
    print debugAlias(A), probs
