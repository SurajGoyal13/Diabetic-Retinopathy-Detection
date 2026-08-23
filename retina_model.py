from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "gaussian_filtered_images" / "gaussian_filtered_images"
CSV_PATH = BASE_DIR / "train.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "dr_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "training_metrics.json"

LABEL_TO_FOLDER = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

FOLDER_TO_LABEL = {value: key for key, value in LABEL_TO_FOLDER.items()}

CLASS_INFO = {
    0: {
        "name": "No Diabetic Retinopathy",
        "short": "No DR",
        "severity": "No visible diabetic retinopathy in this screening result.",
        "risk_level": "Low",
        "summary": "The uploaded retinal image looks closest to the no-retinopathy class.",
        "next_steps": [
            "Keep regular diabetes follow-up visits and routine dilated eye exams.",
            "Maintain blood sugar, blood pressure, and cholesterol control.",
            "Return earlier if you notice blurred vision, floaters, or sudden vision changes.",
        ],
    },
    1: {
        "name": "Mild Nonproliferative Diabetic Retinopathy",
        "short": "Mild",
        "severity": "Early diabetic retinopathy changes may be present.",
        "risk_level": "Mild",
        "summary": "This stage is the earliest visible form of diabetic retinopathy.",
        "next_steps": [
            "Book an ophthalmology or retina check-up for formal confirmation.",
            "Work on stable blood sugar, blood pressure, and cholesterol targets.",
            "Do not wait for symptoms because early disease may not cause vision complaints.",
        ],
    },
    2: {
        "name": "Moderate Nonproliferative Diabetic Retinopathy",
        "short": "Moderate",
        "severity": "More retinal blood-vessel changes may be present.",
        "risk_level": "Moderate",
        "summary": "Moderate disease can progress and should be reviewed by an eye specialist.",
        "next_steps": [
            "Arrange an eye specialist appointment soon for detailed retinal examination.",
            "Review A1C, blood pressure, lipids, smoking status, and current diabetes treatment plan.",
            "Seek faster care if vision becomes blurry or distorted.",
        ],
    },
    3: {
        "name": "Severe Nonproliferative Diabetic Retinopathy",
        "short": "Severe",
        "severity": "Advanced retinal damage may be present before proliferative disease.",
        "risk_level": "High",
        "summary": "This stage carries a higher risk of progression toward sight-threatening disease.",
        "next_steps": [
            "Get prompt ophthalmology review, ideally with a retina specialist.",
            "Further testing may be needed to look for macular edema or worsening ischemia.",
            "Urgent follow-up is important even if symptoms are still mild.",
        ],
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy",
        "short": "Proliferative",
        "severity": "Sight-threatening proliferative changes may be present.",
        "risk_level": "Critical",
        "summary": "This is the most advanced class in the screening model and needs urgent specialist care.",
        "next_steps": [
            "Seek urgent retina-specialist evaluation as soon as possible.",
            "Treatment may include anti-VEGF injections, laser therapy, surgery, or a combination depending on examination findings.",
            "Go urgently if you have sudden floaters, bleeding, dark spots, or rapid vision loss.",
        ],
    },
}

DIABETES_TYPES = [
    {
        "name": "Type 1 Diabetes",
        "summary": "An autoimmune condition where the body stops making insulin. It needs medical diagnosis and ongoing insulin treatment.",
    },
    {
        "name": "Type 2 Diabetes",
        "summary": "The body does not use insulin well and blood sugar rises over time. It is the most common type.",
    },
    {
        "name": "Gestational Diabetes",
        "summary": "Diabetes that develops during pregnancy and needs proper medical follow-up.",
    },
    {
        "name": "Prediabetes",
        "summary": "Blood sugar is higher than normal but not yet in the diabetes range. Lifestyle changes can help lower progression risk.",
    },
]


@dataclass
class TrainingResult:
    model_path: Path
    metrics_path: Path
    accuracy: float
    test_size: int
    train_size: int


def list_dataset_records() -> list[tuple[Path, int]]:
    df = pd.read_csv(CSV_PATH)
    records: list[tuple[Path, int]] = []

    for row in df.itertuples(index=False):
        label = int(row.diagnosis)
        folder = LABEL_TO_FOLDER[label]
        image_path = DATASET_DIR / folder / f"{row.id_code}.png"
        if image_path.exists():
            records.append((image_path, label))

    return records


def preprocess_image(image: Image.Image, size: int = 32) -> np.ndarray:
    image = image.convert("RGB")
    image = ImageOps.autocontrast(image)
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0

    channel_means = array.mean(axis=(0, 1))
    channel_stds = array.std(axis=(0, 1))
    flat = array.flatten()
    return np.concatenate([flat, channel_means, channel_stds], axis=0)


def build_feature_matrix(records: Iterable[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for image_path, label in records:
        with Image.open(image_path) as image:
            features.append(preprocess_image(image))
        labels.append(label)

    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save_model() -> TrainingResult:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = list_dataset_records()
    x, y = build_feature_matrix(records)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = create_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True)

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "labels": LABEL_TO_FOLDER,
                "classification_report": report,
            },
            indent=2,
        )
    )

    return TrainingResult(
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        accuracy=accuracy,
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
    )


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def predict_image(model: Pipeline, image: Image.Image) -> dict:
    feature_vector = preprocess_image(image).reshape(1, -1)
    probabilities = model.predict_proba(feature_vector)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = int(model.classes_[predicted_index])

    probability_map = {
        int(label): float(probabilities[index])
        for index, label in enumerate(model.classes_)
    }

    return {
        "predicted_label": predicted_label,
        "probabilities": probability_map,
        "class_info": CLASS_INFO[predicted_label],
    }


def dataset_summary() -> dict:
    df = pd.read_csv(CSV_PATH)
    counts = df["diagnosis"].value_counts().sort_index().to_dict()
    return {
        "image_count": int(len(df)),
        "class_counts": {CLASS_INFO[key]["short"]: int(value) for key, value in counts.items()},
    }
