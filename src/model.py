"""
Modeling module for the TalentMatch Learning to Rank pipeline.
Defines the HyperGradient Boosting Regressor and Classifier models used for ranking.
"""
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Model hyperparameters (as used in the notebook)
REGRESSOR_PARAMS = dict(
    max_iter=800,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
    early_stopping=False
)

CLASSIFIER_PARAMS = dict(
    max_iter=800,
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
    early_stopping=False
)

def train_regressor(X, y, **kwargs):
    """
    Train a HistGradientBoostingRegressor.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    **kwargs : dict
        Additional parameters to pass to HistGradientBoostingRegressor.
        If not provided, uses the default REGRESSOR_PARAMS.

    Returns
    -------
    HistGradientBoostingRegressor
        Fitted regressor.
    """
    params = REGRESSOR_PARAMS.copy()
    params.update(kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return model

def train_classifier(X, y, **kwargs):
    """
    Train a HistGradientBoostingClassifier.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values (integer labels).
    **kwargs : dict
        Additional parameters to pass to HistGradientBoostingClassifier.
        If not provided, uses the default CLASSIFIER_PARAMS.

    Returns
    -------
    HistGradientBoostingClassifier
        Fitted classifier.
    """
    params = CLASSIFIER_PARAMS.copy()
    params.update(kwargs)
    model = HistGradientBoostingClassifier(**params)
    model.fit(X, y)
    return model

def predict_and_blend(regressor, classifier, X):
    """
    Generate predictions from the regressor and classifier and blend them.

    The blending used in the notebook is: 0.6 * regressor_score + 0.4 * classifier_expected_relevance

    Parameters
    ----------
    regressor : HistGradientBoostingRegressor
        Fitted regressor.
    classifier : HistGradientBoostingClassifier
        Fitted classifier.
    X : array-like of shape (n_samples, n_features)
        Data to predict on.

    Returns
    -------
    numpy.ndarray
        Blended predictions of shape (n_samples,).
    """
    reg_pred = regressor.predict(X)
    class_proba = classifier.predict_proba(X)
    # Expected value of the class distribution
    classes = classifier.classes_.astype(float)
    class_pred = class_proba @ classes
    blended = 0.6 * reg_pred + 0.4 * class_pred
    return blended