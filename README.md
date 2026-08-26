# 👁️ RetinaCare AI

An AI-powered diabetic retinopathy screening application built with **Python, Streamlit, and machine learning**. The application analyzes retinal fundus images and predicts one of five diabetic retinopathy severity classes from **0 to 4**, then provides a severity explanation, probability breakdown, and general next-step care guidance.

The project is designed as an **AI-assisted screening and educational tool**, not as a medical diagnostic system.

## ✨ Key Features

- 🖼️ Upload retinal fundus images for screening
- 🤖 Machine-learning-based diabetic retinopathy classification
- 🔢 Prediction across five diabetic retinopathy classes (`0–4`)
- 📊 Prediction probability breakdown for all classes
- 🩺 Severity and risk-level explanation
- 💡 General next-step care guidance
- 📚 Diabetic retinopathy stage reference
- 🧪 Dataset statistics
- 📈 Hold-out test accuracy display
- 🖼️ Support for local labeled sample images from the dataset
- 📖 Educational information about Type 1, Type 2, gestational diabetes, and prediabetes
- ⚠️ Medical-safety messaging distinguishing screening from diagnosis
- 🌐 Interactive Streamlit web interface

## 🧠 Classification Classes

| Class | Condition |
|------:|-----------|
| `0` | No Diabetic Retinopathy |
| `1` | Mild Nonproliferative Diabetic Retinopathy |
| `2` | Moderate Nonproliferative Diabetic Retinopathy |
| `3` | Severe Nonproliferative Diabetic Retinopathy |
| `4` | Proliferative Diabetic Retinopathy |

The application also provides a description and risk level associated with each predicted class.

## 🔄 How It Works

The application follows this pipeline:

    Retinal Fundus Image
            ↓
    Image Preprocessing
            ↓
    Feature Extraction
            ↓
    Standardization
            ↓
    Logistic Regression Model
            ↓
    Class Probabilities
            ↓
    Predicted DR Class (0–4)
            ↓
    Severity + Risk Explanation
            ↓
    Next-Step Care Guidance

### Image Preprocessing

Each image is:

1. Converted to RGB
2. Automatically contrast-adjusted
3. Center-cropped to a square
4. Resized to **32 × 32**
5. Normalized to the `[0, 1]` range

Additional image statistics, including per-channel means and standard deviations, are included in the feature representation.

### Model

The project uses a scikit-learn pipeline consisting of:

- **StandardScaler** for feature standardization
- **Logistic Regression** for five-class classification
- Balanced class weighting to help account for class imbalance

The model produces class probabilities using `predict_proba()`, which are displayed in the application as a prediction-probability breakdown.

## 🏋️ Model Training

The training pipeline:

1. Reads labels from `train.csv`
2. Maps each label to its corresponding retinal-image directory
3. Loads available dataset images
4. Applies the same preprocessing used during prediction
5. Builds the feature matrix
6. Splits the data into training and hold-out test sets using an **80/20 stratified split**
7. Trains the Logistic Regression pipeline
8. Evaluates predictions on the hold-out test set
9. Saves the trained model
10. Saves training metrics and the classification report

The training split uses a fixed random state of `42` for reproducibility.

## 📊 Model Evaluation

The application displays the model's accuracy on the hold-out test set.

The training process also generates a classification report containing class-level evaluation metrics.

The displayed accuracy should be interpreted as an evaluation result for this particular dataset split, **not as clinical diagnostic accuracy**.

## 📁 Dataset

The project uses a labeled retinal fundus image dataset referenced through `train.csv`.

The expected local dataset structure is:

    gaussian_filtered_images/
    └── gaussian_filtered_images/
        ├── No_DR/
        ├── Mild/
        ├── Moderate/
        ├── Severe/
        └── Proliferate_DR/

The large retinal-image dataset is **not included in this GitHub repository**.

`train.csv` provides the image identifiers and corresponding diagnosis labels used by the training pipeline.

## 🖥️ Application Workflow

The Streamlit application provides three main stages:

### 1. Choose an Image

Upload a retinal fundus image or select a labeled sample image from the local dataset.

### 2. Screen Retinopathy Severity

The trained model predicts one of the five diabetic retinopathy classes.

### 3. Review the Result

The application displays:

- Predicted class
- Severity level
- Risk level
- Class explanation
- Prediction probability breakdown
- General next-step care guidance

The application also provides a severity explorer and educational information about diabetes types.

## 🛠️ Technology Stack

### Programming

- **Python**

### Machine Learning

- **scikit-learn**
- **Logistic Regression**
- **StandardScaler**
- **Joblib**

### Data & Image Processing

- **NumPy**
- **Pandas**
- **Pillow**

### Application

- **Streamlit**

## 📁 Project Structure

    Diabetic-Retinopathy-Detection/
    │
    ├── app.py                    # Streamlit application and user interface
    ├── retina_model.py           # Preprocessing, model training, prediction, and metadata
    ├── train_model.py            # Training entry point
    ├── step1.py                  # Dataset validation utility
    ├── train.csv                 # Dataset labels and image identifiers
    ├── artifacts/
    │   ├── dr_model.joblib       # Trained model artifact
    │   └── training_metrics.json # Training and evaluation metrics
    ├── .gitignore
    └── README.md

The retinal image dataset itself is not stored in the repository.

## 🚀 Installation

### 1. Clone the Repository

    git clone https://github.com/SurajGoyal13/Diabetic-Retinopathy-Detection.git
    cd Diabetic-Retinopathy-Detection

### 2. Create a Virtual Environment

#### Windows

    python -m venv .venv
    .venv\Scripts\activate

#### macOS / Linux

    python3 -m venv .venv
    source .venv/bin/activate

### 3. Install Dependencies

    pip install -r requirements.txt

### 4. Add the Dataset

Place the retinal image dataset in the expected directory:

    gaussian_filtered_images/
    └── gaussian_filtered_images/
        ├── No_DR/
        ├── Mild/
        ├── Moderate/
        ├── Severe/
        └── Proliferate_DR/

Make sure the image filenames correspond to the identifiers in `train.csv`.

### 5. Train the Model

    python train_model.py

This creates or updates:

    artifacts/dr_model.joblib
    artifacts/training_metrics.json

### 6. Run the Streamlit Application

    streamlit run app.py

The application will open in your browser.

## ▶️ Usage

1. Start the Streamlit application.
2. Upload a retinal fundus image or choose a local sample image.
3. Run the screening.
4. Review the predicted diabetic retinopathy class.
5. Examine the probability breakdown.
6. Read the severity and risk explanation.
7. Review the general next-step care guidance.

## 📚 Diabetes Education

The application also provides educational information about:

- Type 1 Diabetes
- Type 2 Diabetes
- Gestational Diabetes
- Prediabetes

This educational section is separate from the retinal-image classification model.

The retinal-image model **does not determine whether a person has Type 1, Type 2, gestational diabetes, or prediabetes**.

## ⚠️ Medical Disclaimer

This project is an **AI-assisted diabetic retinopathy screening prototype**.

It does **not** provide a medical diagnosis and should not be used as a replacement for evaluation by a qualified healthcare professional.

A model prediction may be incorrect, and a concerning result should be confirmed through appropriate clinical examination by an eye-care professional.

The probability values shown by the model are model outputs and should not be interpreted as clinical certainty.

## ⚠️ Limitations

- The model uses engineered image features and Logistic Regression rather than a deep neural network or end-to-end retinal image model.
- Model performance depends on the training dataset and preprocessing pipeline.
- The hold-out accuracy does not establish clinical diagnostic performance.
- The project does not perform clinical validation.
- The retinal-image model does not determine a person's diabetes type.
- The application requires the retinal-image dataset locally for retraining and sample-image functionality.
- Predictions should not be used as a standalone medical decision-making tool.

## 🔮 Future Improvements

- 🧠 Experiment with CNN and transfer-learning architectures
- 🔬 Evaluate additional image preprocessing techniques
- 📊 Add precision, recall, F1-score, and confusion-matrix visualization
- 🧪 Add cross-validation and more robust model evaluation
- 🖼️ Improve retinal-image quality checks
- 🎯 Explore class-imbalance handling techniques
- 🔍 Add model interpretability methods such as saliency or Grad-CAM for deep-learning models
- 📈 Compare multiple machine-learning approaches
- 🌐 Deploy the application for broader demonstration
- 🏥 Explore clinical validation with appropriate medical datasets and expert evaluation

## 🎯 Project Objective

The goal of this project is to demonstrate the practical application of **machine learning to retinal fundus image screening** through an interactive Streamlit application.

The project combines:

- Image preprocessing
- Feature engineering
- Multiclass machine learning
- Model evaluation
- Probability-based prediction
- Dataset analysis
- Interactive web application development
- Medical-safety-aware result presentation

## 📚 Medical Sources Used

The application includes educational information and care guidance informed by:

- **CDC Diabetes Basics**
- **National Eye Institute diabetic retinopathy guidance**
- **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)** diabetic eye disease information

These sources are used for educational and general guidance content within the application and do not replace professional medical advice.

## 👨‍💻 Author

**Suraj Goyal**

Computer Science Student · Python · AI/ML · Web Development · DSA

---

⭐ **If you find this project useful, consider starring the repository.**
