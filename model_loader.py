import joblib


def load_assets():
    model = joblib.load("rubber_model.pkl")

    feature_columns = list(joblib.load("feature_columns.pkl"))
    target_columns = list(joblib.load("target_columns.pkl"))

    epdm_cols = [
        c for c in feature_columns
        if "EPDM" in str(c).upper()
    ]

    other_cols = [
        c for c in feature_columns
        if c not in epdm_cols
    ]

    return model, feature_columns, target_columns, epdm_cols, other_cols