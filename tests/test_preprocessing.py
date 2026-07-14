from src.preprocessing import BreastCancerPreprocessor


def test_preprocessing_pipeline_shapes():
    preprocessor = BreastCancerPreprocessor()
    split = preprocessor.preprocess_pipeline("data/breast-cancer-wisconsin.csv")
    assert split.X_train.shape[0] > 0
    assert split.X_test.shape[0] > 0
    assert split.X_train.shape[1] == 30
    assert split.X_test.shape[1] == 30
    assert len(split.feature_columns) == 30
