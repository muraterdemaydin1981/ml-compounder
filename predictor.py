import pandas as pd


def normalize_phr_recipe(recipe, feature_columns):
    df_phr = pd.DataFrame([recipe])
    df_phr = df_phr.reindex(columns=feature_columns, fill_value=0)

    total_phr = df_phr[feature_columns].sum(axis=1).iloc[0]

    if total_phr == 0:
        df_norm = df_phr.copy()
    else:
        df_norm = df_phr / total_phr * 100

    return df_phr, df_norm, total_phr


def predict_recipe(model, recipe, feature_columns, target_columns):
    df_phr, df_norm, total_phr = normalize_phr_recipe(recipe, feature_columns)

    prediction = model.predict(df_norm)

    result_df = pd.DataFrame(prediction, columns=target_columns)

    return result_df, df_phr, df_norm, total_phr