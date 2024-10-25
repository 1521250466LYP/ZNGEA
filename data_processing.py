import numpy as np
import scipy as sp

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output




def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.sparse.issparse(A):
        D = sp.sparse.diags(degrees)
    else:
        D = np.diag(degrees)
    return D