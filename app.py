from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
from PIL import Image
import streamlit as st

from retina_model import (
    ARTIFACTS_DIR,
    CLASS_INFO,
    DATASET_DIR,
    DIABETES_TYPES,
    LABEL_TO_FOLDER,
    dataset_summary,
    load_metrics,
    load_model,
    predict_image,
)


st.set_page_config(
    page_title="RetinaCare AI",
    page_icon="👁️",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        :root {
            --bg: #f6f8f4;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-strong: #ffffff;
            --text: #15211a;
            --muted: #5f6f66;
            --accent: #0f766e;
            --accent-soft: #dff5f1;
            --border: #d9e7df;
            --critical: #b42318;
            --high: #c2410c;
            --medium: #a16207;
            --low: #15803d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 24%),
                radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.10), transparent 22%),
                linear-gradient(180deg, #f8fbf7 0%, #eef4ef 100%);
            color: var(--text);
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            font-family: "Outfit", sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        .hero {
            padding: 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0f766e, #115e59 52%, #164e63);
            color: white;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
            margin-bottom: 1rem;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2.2rem, 4vw, 4rem);
            margin-bottom: 0.4rem;
        }

        .hero p {
            color: rgba(255, 255, 255, 0.92);
            max-width: 820px;
            line-height: 1.75;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.2rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
        }

        .metric-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem;
            min-height: 132px;
        }

        .severity-pill {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .critical { color: var(--critical); }
        .high { color: var(--high); }
        .medium { color: var(--medium); }
        .low { color: var(--low); }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .info-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem;
            min-height: 180px;
        }

        .source-list a {
            color: var(--accent);
            text-decoration: none;
            font-weight: 700;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.2rem;
        }

        .workflow-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem;
            min-height: 170px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        .workflow-card span {
            display: inline-block;
            padding: 0.32rem 0.6rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.7rem;
        }

        .workflow-card p {
            color: var(--muted);
            line-height: 1.7;
            margin: 0;
        }

        .insight-card {
            background: linear-gradient(135deg, #ffffff, #f2faf8);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem;
            min-height: 140px;
        }

        .prob-row {
            margin-bottom: 0.8rem;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }

        .stage-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem;
            margin-top: 0.5rem;
        }

        @media (max-width: 900px) {
            .info-grid {
                grid-template-columns: 1fr;
            }

            .workflow-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_dataset_summary():
    return dataset_summary()


def risk_css_class(risk_level: str) -> str:
    mapping = {
        "Low": "low",
        "Mild": "low",
        "Moderate": "medium",
        "High": "high",
        "Critical": "critical",
    }
    return mapping.get(risk_level, "medium")


def get_sample_image(label: int | None = None) -> Path:
    if label is None:
        label = random.choice(sorted(CLASS_INFO.keys()))
    folder = DATASET_DIR / LABEL_TO_FOLDER[label]
    images = sorted(folder.glob("*.png"))
    return random.choice(images)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="severity-pill">Diabetic Retinopathy Screening Project</div>
            <h1>RetinaCare AI</h1>
            <p>
                Upload a retinal fundus image and this project estimates diabetic retinopathy severity from
                class <strong>0 to 4</strong>, explains what the stage means, and shows medically safer
                next-step guidance. It also includes diabetes-type education and a clear note that retina
                images do <strong>not</strong> diagnose whether someone has type 1, type 2, or gestational diabetes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_summary(summary: dict, metrics: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    counts = summary["class_counts"]

    with col1:
        st.markdown(
            f"<div class='metric-card'><small>Dataset Size</small><h3>{summary['image_count']}</h3><p>Retina images available locally</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><small>No DR / Mild</small><h3>{counts.get('No DR', 0)} / {counts.get('Mild', 0)}</h3><p>Lower-severity groups in the dataset</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'><small>Moderate / Severe</small><h3>{counts.get('Moderate', 0)} / {counts.get('Severe', 0)}</h3><p>Mid-to-high severity groups</p></div>",
            unsafe_allow_html=True,
        )
    with col4:
        accuracy = metrics.get("accuracy")
        accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "Not trained yet"
        st.markdown(
            f"<div class='metric-card'><small>Validation Accuracy</small><h3>{accuracy_text}</h3><p>From the locally trained screening model</p></div>",
            unsafe_allow_html=True,
        )


def render_workflow() -> None:
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-card">
                <span>Step 1</span>
                <h4>Choose Your Image</h4>
                <p>Upload a retina image from your system or try one of the labeled dataset samples already included in the project.</p>
            </div>
            <div class="workflow-card">
                <span>Step 2</span>
                <h4>Screen Retinopathy Severity</h4>
                <p>The trained model predicts one of five diabetic retinopathy classes from class 0 to class 4.</p>
            </div>
            <div class="workflow-card">
                <span>Step 3</span>
                <h4>Read The Report</h4>
                <p>See severity, confidence breakdown, stage explanation, and general next-step care guidance in one place.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(result: dict) -> None:
    info = result["class_info"]
    predicted_label = result["predicted_label"]
    sorted_probs = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)

    st.markdown(
        f"""
        <div class="card">
            <div class="severity-pill">Predicted Class {predicted_label}</div>
            <h2>{info['name']}</h2>
            <p class="{risk_css_class(info['risk_level'])}"><strong>Risk level:</strong> {info['risk_level']}</p>
            <p><strong>Severity meaning:</strong> {info['severity']}</p>
            <p>{info['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_cols = st.columns(min(3, len(sorted_probs)))
    for index, (label, prob) in enumerate(sorted_probs[:3]):
        with prob_cols[index]:
            st.metric(CLASS_INFO[label]["short"], f"{prob * 100:.2f}%")

    st.markdown("### Recommended Next Steps")
    for step in info["next_steps"]:
        st.write(f"- {step}")

    st.warning(
        "This is a screening-style ML prediction, not a medical diagnosis. Please confirm any concerning result with a qualified eye specialist."
    )


def render_probability_breakdown(result: dict) -> None:
    st.markdown("### Prediction Confidence Breakdown")
    sorted_probs = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    for label, prob in sorted_probs:
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-meta">
                    <strong>{CLASS_INFO[label]['short']}</strong>
                    <span>{prob * 100:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(prob))


def render_stage_explorer() -> None:
    st.markdown("## Severity Explorer")
    selected_label = st.select_slider(
        "Move through the retinopathy stages",
        options=list(CLASS_INFO.keys()),
        key="severity_explorer_label",
        format_func=lambda value: f"Class {value}: {CLASS_INFO[value]['short']}",
    )
    info = CLASS_INFO[selected_label]
    st.markdown(
        f"""
        <div class="stage-card">
            <div class="severity-pill">Class {selected_label}</div>
            <h3>{info['name']}</h3>
            <p class="{risk_css_class(info['risk_level'])}"><strong>Risk level:</strong> {info['risk_level']}</p>
            <p><strong>What it means:</strong> {info['severity']}</p>
            <p>{info['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_care_guidance() -> None:
    st.markdown("## Care And Follow-up Guide")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Prevention Focus</h4>
                <p>Control blood sugar, blood pressure, cholesterol, regular exercise, and follow scheduled dilated eye exams.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Urgent Warning Signs</h4>
                <p>Rapid blur, dark floaters, bleeding appearance, sudden blind spots, or fast vision changes need urgent specialist care.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Common Treatments</h4>
                <p>Depending on stage, doctors may recommend monitoring, anti-VEGF injections, laser treatment, or vitrectomy surgery.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Why this app cannot tell the diabetes type from only an eye image"):
        st.write(
            "Retina images can help screen diabetic retinopathy, but type 1, type 2, gestational diabetes, and prediabetes are diagnosed using medical history, examination, and lab tests like blood glucose and A1C."
        )

    with st.expander("What this project is best used for"):
        st.write(
            "This project is best presented as an AI-assisted diabetic retinopathy severity screening system for retinal images, with educational guidance and class-based severity reporting."
        )


def render_diabetes_education() -> None:
    st.markdown("## Diabetes Types")
    st.info(
        "Important: this retina-image model screens diabetic retinopathy severity. It does not determine whether a person has type 1, type 2, gestational diabetes, or prediabetes."
    )
    col1, col2 = st.columns(2)
    for index, item in enumerate(DIABETES_TYPES):
        target_col = col1 if index % 2 == 0 else col2
        with target_col:
            st.markdown(
                f"""
                <div class='info-card'>
                    <h4>{item['name']}</h4>
                    <p>{item['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_stage_reference() -> None:
    st.markdown("## Diabetic Retinopathy Stage Reference")
    stage_rows = []
    for label, info in CLASS_INFO.items():
        stage_rows.append(
            {
                "Class": label,
                "Stage": info["name"],
                "Risk": info["risk_level"],
                "Meaning": info["severity"],
            }
        )
    st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)


def render_sources() -> None:
    st.markdown(
        """
        <div class="card source-list">
            <h3>Medical Reference Sources Used In This Project</h3>
            <p><a href="https://www.cdc.gov/diabetes/about/index.html" target="_blank">CDC: Diabetes Basics</a></p>
            <p><a href="https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/diabetic-retinopathy" target="_blank">NEI: Diabetic Retinopathy</a></p>
            <p><a href="https://www.nei.nih.gov/sites/default/files/health-pdfs/Diabetic_Retinopathy_What_You_Should_Know.pdf" target="_blank">NEI: Diabetic Retinopathy Stages PDF</a></p>
            <p><a href="https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/diabetic-eye-disease" target="_blank">NIDDK: Diabetic Eye Disease Treatment and Prevention</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    render_hero()

    metrics = load_metrics()
    summary = get_dataset_summary()
    render_project_summary(summary, metrics)
    render_workflow()

    if "sample_refresh_count" not in st.session_state:
        st.session_state.sample_refresh_count = 0
    if "severity_explorer_label" not in st.session_state:
        st.session_state.severity_explorer_label = 2
    if "last_prediction_signature" not in st.session_state:
        st.session_state.last_prediction_signature = None

    with st.sidebar:
        st.header("Screening Controls")
        st.caption("Use a sample retina image or upload your own for prediction.")
        use_sample = st.toggle("Use sample dataset image", value=True)
        selected_sample_label = st.selectbox(
            "Sample class",
            options=list(CLASS_INFO.keys()),
            format_func=lambda value: f"Class {value} - {CLASS_INFO[value]['short']}",
        )
        if st.button("Load another sample", use_container_width=True):
            st.session_state.sample_refresh_count += 1
            st.rerun()
        st.markdown("---")
        st.markdown("**Included screening classes**")
        for label, info in CLASS_INFO.items():
            st.write(f"`{label}` - {info['short']}")

    st.markdown("## Upload Or Test With A Sample")
    left, right = st.columns([1.1, 0.9], gap="large")

    uploaded_file = left.file_uploader(
        "Upload a retina image",
        type=["png", "jpg", "jpeg"],
    )

    image_to_predict = None
    image_caption = ""

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        image_caption = uploaded_file.name
    elif use_sample:
        sample_path = get_sample_image(selected_sample_label)
        image_to_predict = Image.open(sample_path)
        image_caption = f"Sample: {sample_path.name}"

    with left:
        if image_to_predict is not None:
            st.image(image_to_predict, caption=image_caption, use_container_width=True)
        else:
            st.info("Upload a retinal image or enable sample mode from the sidebar.")

    with right:
        if not ARTIFACTS_DIR.joinpath("dr_model.joblib").exists():
            st.error(
                "Model artifact not found yet. Run `python3 train_model.py` once to train the classifier before using the app."
            )
        elif image_to_predict is not None:
            model = get_model()
            result = predict_image(model, image_to_predict)
            prediction_signature = f"{image_caption}|{result['predicted_label']}"
            if st.session_state.last_prediction_signature != prediction_signature:
                st.session_state.severity_explorer_label = result["predicted_label"]
                st.session_state.last_prediction_signature = prediction_signature
            render_prediction(result)
            render_probability_breakdown(result)
        else:
            st.markdown(
                """
                <div class="card">
                    <h3>Ready For Screening</h3>
                    <p>
                        This app predicts one of five diabetic retinopathy classes:
                        No DR, Mild, Moderate, Severe, or Proliferative DR.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_stage_explorer()
    render_stage_reference()
    render_care_guidance()
    render_diabetes_education()
    render_sources()


if __name__ == "__main__":
    main()
