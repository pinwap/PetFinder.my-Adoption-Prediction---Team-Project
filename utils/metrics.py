from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    return cohen_kappa_score(
        y_true,
        y_pred,
        weights='quadratic'
    )