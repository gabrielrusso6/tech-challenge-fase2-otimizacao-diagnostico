from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LogisticRegression_baseline": LogisticRegression(
            solver="liblinear", max_iter=10000, random_state=random_state, tol=1e-4
        ),
        "RandomForest_baseline": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "SVM_baseline": SVC(random_state=random_state, probability=True),
        "KNN_baseline": KNeighborsClassifier(),
        "NaiveBayes_baseline": GaussianNB(),
    }
