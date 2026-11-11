# Plant Guardian AI 🌱

A beginner-friendly plant-leaf image classification project using TensorFlow and Streamlit.

## What is included
- `app.py`: web app for uploading a leaf photo and viewing the model prediction.
- `train_model.py`: trains a MobileNetV2 transfer-learning classifier.
- `dataset/`: put your image folders here, one folder per class.

## Important
The project does **not** include a trained disease model. You must train it on a suitable, correctly labelled dataset first. Model predictions are only screening hints, not confirmed diagnoses. Dataset quality and the plant species represented determine how reliable the model can be.

## 1. Prepare the dataset
Create folders like this (example only):

```text
dataset/
  Tomato_healthy/
  Tomato_Early_blight/
  Tomato_Late_blight/
  Potato_healthy/
  Potato_Early_blight/
```

Put labelled JPG/PNG leaf photos in the matching folders. Use a reputable dataset and check its licence and labels. Keep classes reasonably balanced and do not mix plant species/classes accidentally.

## 2. Install
Use Python 3.10 or 3.11 in a virtual environment:

```bash
python -m venv .venv
```

Activate it, then:

```bash
pip install -r requirements.txt
```

## 3. Train
```bash
python train_model.py
```

This creates `plant_guardian_model.keras` and `class_names.txt`.

## 4. Run the app
```bash
streamlit run app.py
```

Open the local address shown in the terminal.

## Google Colab note
You can upload this project to Google Drive, open a Colab notebook, install the requirements, and mount Drive. Training is compute-intensive; a GPU runtime may help. Keep the model and class-name file together with `app.py`.

## Limitations
- The classifier only recognizes classes it was trained on.
- A leaf photo may not show the full cause of a plant problem.
- Lighting, camera quality, plant variety, and dataset bias can affect predictions.
- Do not apply pesticides or other treatments based only on this app; verify with a qualified local agriculture expert.
