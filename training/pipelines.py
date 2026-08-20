from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training.config import DetectorConfig


def build_pipeline(cfg: DetectorConfig, seed: int) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=cfg.ngram,
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
    )
    classifier = LinearSVC(
        C=cfg.c,
        class_weight=cfg.class_weight or None,
        dual=True,
        random_state=seed,
    )
    return Pipeline([("tfidf", vectorizer), ("svc", classifier)])
