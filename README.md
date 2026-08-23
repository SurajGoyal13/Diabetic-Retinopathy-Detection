# RetinaCare AI

This project screens diabetic retinopathy severity from retinal fundus images.

## Features

- Upload a retina image and predict diabetic retinopathy class `0-4`
- Show severity level and class explanation
- Show next-step care guidance
- Explain the main types of diabetes
- Show dataset statistics and model validation accuracy
- Use either uploaded images or local sample images from the dataset

## Class Mapping

- `0` = No Diabetic Retinopathy
- `1` = Mild Nonproliferative DR
- `2` = Moderate Nonproliferative DR
- `3` = Severe Nonproliferative DR
- `4` = Proliferative Diabetic Retinopathy

## Important Note

This project screens diabetic retinopathy severity from an eye image. It does not diagnose whether a patient has type 1, type 2, gestational diabetes, or prediabetes. Those require clinical evaluation and lab testing.

## Setup

```bash
cd Diabetic-Retinopathy-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train_model.py
streamlit run app.py
```

## Files

- `app.py` : Streamlit UI
- `retina_model.py` : preprocessing, training, prediction, metadata
- `train_model.py` : trains and saves the model
- `step1.py` : quick dataset validation script
- `train.csv` : label file
- `gaussian_filtered_images/` : local training dataset (not included in GitHub repository)

## Medical Sources Used

- CDC Diabetes Basics
- National Eye Institute diabetic retinopathy guidance
- NIDDK diabetic eye disease treatment and prevention
