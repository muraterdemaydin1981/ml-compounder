import streamlit as st
import pandas as pd

from model_loader import load_assets
from predictor import predict_recipe


st.set_page_config(page_title="ML Compounder", layout="wide")

model, feature_columns, target_columns, epdm_cols, other_cols = load_assets()

st.title("ML Compounder")
st.caption("AI-Powered Rubber Formulation Platform")

tab_predict, tab_opt = st.tabs([
    "📈 Property Prediction",
    "🧪 Recipe Optimization"
])

with tab_predict:
    st.header("Property Prediction")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(feature_columns):
        with cols[i % 3]:
            if feature in epdm_cols:
                default = 100.0 / len(epdm_cols)
            elif feature in ["CB1", "CB2"]:
                default = 50.0
            elif feature == "Oil":
                default = 60.0
            elif feature in ["White1", "White2"]:
                default = 30.0
            elif feature == "Activators":
                default = 5.0
            elif feature == "S+Accelerators":
                default = 3.0
            else:
                default = 0.0

            input_data[feature] = st.number_input(
                str(feature),
                value=float(default),
                step=1.0,
                key=f"pred_{feature}"
            )

    input_df = pd.DataFrame([input_data]).reindex(
        columns=feature_columns,
        fill_value=0
    )

    epdm_total = input_df[epdm_cols].sum(axis=1).iloc[0]
    total_phr = input_df[feature_columns].sum(axis=1).iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("EPDM Total", f"{epdm_total:.1f} phr")
    c2.metric("Total Compound PHR", f"{total_phr:.1f} phr")

    if st.button("Predict Properties"):
        pred_df, df_phr, df_norm, total_phr_model = predict_recipe(
            model,
            input_data,
            feature_columns,
            target_columns
        )

        st.subheader("Predicted Properties")
        st.dataframe(
            pred_df.T.rename(columns={0: "Predicted"}),
            use_container_width=True
        )


with tab_opt:
    st.header("Recipe Optimization")
    st.write("Enter target specifications. The optimizer will suggest candidate PHR formulations.")

    st.subheader("Target Specifications")

    specs = {}

    h1, h2, h3, h4 = st.columns([1, 3, 2, 2])
    h1.markdown("**Use**")
    h2.markdown("**Property**")
    h3.markdown("**Condition**")
    h4.markdown("**Target**")

    for prop in target_columns:
        c1, c2, c3, c4 = st.columns([1, 3, 2, 2])

        use = c1.checkbox("", value=False, key=f"opt_use_{prop}")
        c2.write(prop)

        condition = c3.selectbox(
            "",
            ["=", ">=", "<="],
            key=f"opt_cond_{prop}"
        )

        target = c4.number_input(
            "",
            value=0.0,
            step=1.0,
            key=f"opt_target_{prop}"
        )

        specs[prop] = {
            "use": use,
            "condition": condition,
            "target": target
        }

    st.subheader("Basic Recipe Rules")

    r1, r2, r3 = st.columns(3)

    with r1:
        cb_max = st.number_input("Max CB1 / CB2", value=150.0, step=10.0)
        oil_max = st.number_input("Max Oil", value=150.0, step=10.0)

    with r2:
        white_max = st.number_input("Max White1 / White2", value=200.0, step=10.0)
        activator_min = st.number_input("Min Activators", value=2.0, step=1.0)
        activator_max = st.number_input("Max Activators", value=10.0, step=1.0)

    with r3:
        cure_min = st.number_input("Min S+Accelerators", value=1.0, step=0.5)
        cure_max = st.number_input("Max S+Accelerators", value=10.0, step=0.5)
        max_total_phr = st.number_input("Max Total PHR", value=450.0, step=25.0)

    st.info("Optimizer generates PHR recipes. Before prediction, recipes are automatically normalized to % for the model.")

    st.subheader("Ingredient Availability")

    active = {}
    cols = st.columns(4)

    for i, feature in enumerate(feature_columns):
        with cols[i % 4]:
            active[feature] = st.checkbox(
                f"Use {feature}",
                value=True,
                key=f"opt_active_{feature}"
            )

    st.subheader("Optimization Settings")

    n_candidates = st.slider(
        "Number of candidate recipes to test",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    limits = {
        "cb_max": cb_max,
        "oil_max": oil_max,
        "white_max": white_max,
        "activator_min": activator_min,
        "activator_max": activator_max,
        "cure_min": cure_min,
        "cure_max": cure_max,
        "max_total_phr": max_total_phr
    }

    if st.button("Optimize Recipe"):
        selected_specs = [p for p, rule in specs.items() if rule["use"]]
        active_epdm = [c for c in epdm_cols if active.get(c, True)]

        if len(selected_specs) == 0:
            st.error("Select at least one target specification.")
        elif len(active_epdm) == 0:
            st.error("At least one EPDM must be active.")
        else:
            with st.spinner("Searching candidate formulations..."):
                results = optimize_random_search(
                    model=model,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    epdm_cols=epdm_cols,
                    other_cols=other_cols,
                    specs=specs,
                    active=active,
                    limits=limits,
                    n_candidates=n_candidates,
                    n_results=5
                )

            st.subheader("Suggested Formulations")

            for i, result in enumerate(results, start=1):
                score = result["score"]
                recipe = result["recipe"]
                pred = result["prediction"]
                total_phr = result["total_phr"]

                st.markdown(f"### Alternative {i}")

                m1, m2 = st.columns(2)
                m1.metric("Compatibility Score", f"{max(0, 100 - score * 100):.1f} / 100")
                m2.metric("Total PHR", f"{total_phr:.1f}")

                left, right = st.columns(2)

                with left:
                    st.write("Suggested Formula - PHR")
                    recipe_df = pd.DataFrame.from_dict(
                        recipe,
                        orient="index",
                        columns=["PHR"]
                    )
                    st.dataframe(recipe_df, use_container_width=True)

                with right:
                    st.write("Predicted Properties")

                    rows = []

                    for prop in target_columns:
                        predicted = pred[prop]

                        if specs[prop]["use"]:
                            condition = specs[prop]["condition"]
                            target = specs[prop]["target"]
                            ok = check_spec(predicted, condition, target)
                            status = "OK ✅" if ok else "NOT OK ❌"
                            spec_text = f"{condition} {target}"
                        else:
                            status = "-"
                            spec_text = "-"

                        rows.append({
                            "Property": prop,
                            "Predicted": predicted,
                            "Spec": spec_text,
                            "Status": status
                        })

                    status_df = pd.DataFrame(rows)
                    st.dataframe(status_df, use_container_width=True)
