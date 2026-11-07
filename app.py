import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "ticket_classifier.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
EVALUATION_PATH = ARTIFACT_DIR / "evaluation.json"
CONFUSION_PATH = ARTIFACT_DIR / "confusion_matrix.png"

st.set_page_config(
    page_title="Service Desk Intelligence",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path_string):
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("🛠️ IT Service Desk Intelligence")
st.caption(
    "A machine-learning prototype for classifying IT support tickets "
    "and suggesting a support destination."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Trained model files were not found. First run "
        "`python train_model.py` from the project folder."
    )
    st.stop()

model = load_model()
metadata = load_json(str(METADATA_PATH))
team_routing = metadata.get("team_routing", {})
classes = metadata.get("classes", [])

page = st.sidebar.radio(
    "Navigation",
    ["Predict a ticket", "Batch prediction", "Model insights"],
)

if page == "Predict a ticket":
    st.subheader("Classify an IT support request")
    examples = [
        "I cannot connect to the company VPN from my laptop.",
        "Please grant me access to a shared folder.",
        "Microsoft Outlook is not sending or receiving emails.",
        "A software application crashes whenever I open it.",
    ]
    selected_example = st.selectbox(
        "Try an example (optional)",
        ["Write my own ticket"] + examples,
    )
    default_text = "" if selected_example == "Write my own ticket" else selected_example
    ticket_text = st.text_area(
        "Describe the technical issue",
        value=default_text,
        height=150,
        placeholder="Example: I am unable to access the shared drive...",
    )

    if st.button("Analyse ticket", type="primary"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description first.")
        else:
            prediction = model.predict([ticket_text])[0]
            probabilities = model.predict_proba([ticket_text])[0]
            probability_by_class = dict(zip(model.classes_, probabilities))
            confidence = float(probability_by_class[prediction])
            support_team = team_routing.get(
                prediction, "Manual routing review required"
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted category", str(prediction))
            col2.metric("Suggested support team", support_team)
            col3.metric("Model score", f"{confidence:.1%}")

            st.progress(confidence)
            st.caption(metadata.get("confidence_note", ""))

            if confidence < 0.45:
                st.warning(
                    "Low-confidence result: send this ticket for human review "
                    "instead of relying on automatic routing."
                )
            else:
                st.info(
                    "Prototype suggestion only. A human support agent should "
                    "confirm the category before operational use."
                )

            probability_table = pd.DataFrame(
                {
                    "Category": list(probability_by_class.keys()),
                    "Model score": list(probability_by_class.values()),
                }
            ).sort_values("Model score", ascending=False)
            st.subheader("Scores across all categories")
            st.dataframe(
                probability_table.style.format({"Model score": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

elif page == "Batch prediction":
    st.subheader("Classify multiple tickets")
    st.write("Upload a CSV file containing a column named `text`.")
    uploaded_file = st.file_uploader("Upload ticket CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            if "text" not in input_df.columns:
                st.error("The uploaded CSV must contain a column named `text`.")
            else:
                input_df["text"] = input_df["text"].fillna("").astype(str)
                valid_mask = input_df["text"].str.strip().ne("")
                result_df = input_df.loc[valid_mask].copy()

                if result_df.empty:
                    st.warning("No non-empty ticket descriptions were found.")
                else:
                    predictions = model.predict(result_df["text"])
                    probabilities = model.predict_proba(result_df["text"])
                    result_df["predicted_category"] = predictions
                    result_df["model_score"] = probabilities.max(axis=1)
                    result_df["suggested_team"] = result_df[
                        "predicted_category"
                    ].map(team_routing).fillna("Manual review")

                    st.success(f"Processed {len(result_df)} tickets.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv_bytes,
                        file_name="ticket_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

elif page == "Model insights":
    st.subheader("Model evaluation")
    if not EVALUATION_PATH.exists():
        st.info("Evaluation file not found. Run the training script first.")
    else:
        evaluation = load_json(str(EVALUATION_PATH))
        c1, c2, c3 = st.columns(3)
        c1.metric("Holdout accuracy", f"{evaluation['test_accuracy']:.1%}")
        c2.metric("Holdout macro F1", f"{evaluation['test_macro_f1']:.1%}")
        c3.metric("Test tickets", evaluation["test_rows"])

        st.caption(evaluation["split_note"])
        st.write(
            f"Training examples: **{evaluation['train_rows']}** | "
            f"Test examples: **{evaluation['test_rows']}**"
        )

        st.markdown("#### Validation model comparison")
        comparison_df = pd.DataFrame(
            evaluation["validation_model_comparison"]
        ).T.reset_index(names="Model")
        st.dataframe(
            comparison_df.style.format(
                {
                    "validation_accuracy": "{:.3f}",
                    "validation_macro_f1": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if CONFUSION_PATH.exists():
            st.markdown("#### Confusion matrix")
            st.image(str(CONFUSION_PATH), use_container_width=True)

        st.markdown("#### Classification report")
        report_df = pd.DataFrame(evaluation["classification_report"]).T
        st.dataframe(report_df, use_container_width=True)

st.divider()
st.caption(
    "Research/portfolio prototype. Predictions and routing suggestions "
    "must be reviewed by a human before real operational use. "
    "Do not upload confidential company tickets."
)
