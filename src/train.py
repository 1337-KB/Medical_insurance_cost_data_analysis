from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

SUPPORTED_CLASSIFIERS = ("logistic", "naive_bayes", "lda")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def train_regression_model(df_reg, test_size: float = 0.2, random_state: int = 42):
    X = df_reg.drop("charges", axis=1)
    y = df_reg["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def train_classification_model(
    df_cls,
    classifier: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
):
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            f"Choose from: {SUPPORTED_CLASSIFIERS}"
        )

    X = df_cls.drop("smoker_bin", axis=1)
    y = df_cls["smoker_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if classifier == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == "naive_bayes":
        model = GaussianNB()
    elif classifier == "lda":
        model = LinearDiscriminantAnalysis()

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
