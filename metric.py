import numpy as np


def metric(y_true, y_pred, y_old, at1=10, at2=30, average=True):
    """
    new_prec@10 + new_prec@30 + 1/2 *(prec_@10 + prec@30)
    """
    scores = []
    for t, p, o in zip(y_true, y_pred, y_old):
        t = list(t)
        p = list(p)
        o = o if isinstance(o, (set, list)) else []
        
        prec1 = len(set(t[:at1]) & set(p[:at1])) / at1
        prec2 = len(set(t[:at2]) & set(p[:at2])) / at2
        new_prec1 = len((set(p[:at1]) - set(o)) & set(t[:at1])) / at1
        new_prec2 = len((set(p[:at2]) - set(o)) & set(t[:at2])) / at2

        scores.append(new_prec1 + new_prec2 + 0.5*(prec1 + prec2))

    return np.mean(scores) if average else scores


assert metric([[1]], [[3,1,0,4]], [[1]], at1=2, at2=4) == 0.375
assert metric([[1]], [[3,0,1,4]], [[1]], at1=2, at2=4) == 0.125
assert metric([[1]], [[3,2,0,1]], [[1]], at1=2, at2=4) == 0.125
assert metric([[1]], [[3,0,5,4]], [[1]], at1=2, at2=4) == 0.0
assert metric([[1]], [[3,1,0,4]], [[2]], at1=2, at2=4) == 1.125
assert metric([[1]], [[3,0,1,4]], [[2]], at1=2, at2=4) == 0.375
assert metric([[1]], [[3,2,0,1]], [[2]], at1=2, at2=4) == 0.375
assert metric([[1]], [[3,0,5,4]], [[2]], at1=2, at2=4) == 0.0
