# DeepTrace — Project Guide

> **Project:** Explainable Deepfake Image Detection Using Facial Landmark Geometry and Symmetry Analysis  
> **Status:** In development  
> **Platform:** Windows  
>
> This is the main project guide. Update it whenever a new development stage is completed.

---

## 1. Project Overview

DeepTrace is designed to detect whether a facial image is **Real** or **Deepfake** using facial landmark geometry and facial symmetry analysis.

Instead of relying only on a black-box image classifier, the project extracts interpretable measurements from a detected face, including:

- distances between facial landmarks
- facial proportions
- ratios
- angles
- left/right facial symmetry

These features will later be provided to a machine-learning classifier.

The final system is intended to provide:

- **REAL / DEEPFAKE** prediction
- confidence score
- explanation based on relevant features
- facial landmark visualization

### Important design principle

Natural human faces are not perfectly symmetrical. Therefore, symmetry is treated as **one feature group among several**, not as a rule that says "more symmetry = fake."

---

## 2. Project Objective

The objective is to build a lightweight and explainable deepfake-image detection system based on:

1. Face detection
2. Facial landmark detection
3. Geometry feature extraction
4. Facial symmetry analysis
5. Machine-learning classification
6. Explainable prediction
7. Web application
8. Prediction storage using PostgreSQL

---

## 3. System Pipeline

```text
Input Image
     |
     v
Face Detection
     |
     v
Facial Landmarks
     |
     v
Geometry Features + Symmetry Features
     |
     v
Machine Learning Classifier
     |
     v
REAL / DEEPFAKE
     |
     v
Explanation + Confidence
     |
     v
Store Prediction in PostgreSQL
```

---

## 4. Technology Stack

### Machine Learning / Computer Vision

- Python
- MediaPipe Face Landmarker
- OpenCV
- NumPy
- scikit-learn
- Hugging Face Datasets

### Frontend

- React.js
- JavaScript

### Backend

- Node.js
- Express.js
- REST API

### Database

- PostgreSQL

### Development Tools

- VS Code
- Jupyter Notebook
- Git
- GitHub
- Windows Command Prompt

---

## 5. Project Folder Structure

Current structure is approximately:

```text
deepfake-detection/
|
├── backend/
|   ├── db.js
|   ├── server.js
|   ├── package.json
|   ├── package-lock.json
|   └── .env
|
├── frontend/
|   ├── src/
|   ├── package.json
|   └── ...
|
├── dataset/
|   ├── test_landmarks.py
|   ├── visualize_landmarks.py
|   ├── extract_geometry.py
|   ├── extract_features.py
|   └── samples/
|       ├── fake_landmarks.jpg
|       └── real_landmarks.jpg
|
├── models/
|   └── face_landmarker.task
|
├── docs/
|   └── PROJECT_GUIDE.md
|
├── .gitignore
└── ...
```

The structure may change as development continues.

---

## 6. Prerequisites

Check Python:

```cmd
py --version
```

Check Node.js:

```cmd
node --version
```

Check npm:

```cmd
npm --version
```

Check PostgreSQL:

```cmd
psql --version
```

If `psql` is not recognized, add the PostgreSQL `bin` directory to the Windows System PATH.

---

## 7. Open the Project

Current project directory:

```text
C:\Users\piush\OneDrive - presidencyuniversity.in\Sem 3\MINI Proj\deepfake-detection
```

Open Command Prompt:

```cmd
cd "C:\Users\piush\OneDrive - presidencyuniversity.in\Sem 3\MINI Proj\deepfake-detection"
```

Check files:

```cmd
dir
```

---

## 8. Dataset

Current dataset:

```text
Hemg/deepfake-and-real-images
```

It is loaded using Hugging Face `datasets`.

Current dataset information:

```text
Fake: 95134
Real: 95201
Total: 190335
```

The labels are:

```text
0 -> Fake
1 -> Real
```

The dataset is loaded from the Hugging Face cache rather than duplicating all 190,335 images inside the Git repository.

### Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("Hemg/deepfake-and-real-images")
train = ds["train"]
```

Check labels:

```python
print(train.features["label"].names)
```

Expected:

```text
['Fake', 'Real']
```

---

## 9. MediaPipe Face Landmarker

MediaPipe is used to detect facial landmarks.

The project uses the newer MediaPipe Tasks API.

Model:

```text
models/face_landmarker.task
```

The tested setup detects **478 facial landmarks** for a face.

---

## 10. MediaPipe Installation

Run:

```cmd
py -m pip install mediapipe opencv-python
```

Check the version:

```cmd
py -c "import mediapipe as mp; print(mp.__version__)"
```

The currently tested version is:

```text
1.0.1
```

---

## 11. Face Landmark Detection Test

File:

```text
dataset/test_landmarks.py
```

This test loads one Fake image and one Real image and checks whether MediaPipe detects their faces.

Run:

```cmd
cd dataset
py test_landmarks.py
```

Expected important output:

```text
Testing Fake image...
Fake: Face detected!
Number of landmarks: 478

Testing Real image...
Real: Face detected!
Number of landmarks: 478

Landmark test completed.
```

This verifies:

- dataset loading
- Fake/Real image access
- MediaPipe model loading
- face detection
- landmark extraction

---

## 12. Landmark Visualization

File:

```text
dataset/visualize_landmarks.py
```

This draws the detected landmarks over one Fake image and one Real image.

Run:

```cmd
py visualize_landmarks.py
```

Expected output:

```text
Processing Fake image...
Fake: 478 landmarks detected.
Saved to: samples\fake_landmarks.jpg

Processing Real image...
Real: 478 landmarks detected.
Saved to: samples\real_landmarks.jpg

Landmark visualization completed.
```

Generated files:

```text
dataset/samples/fake_landmarks.jpg
dataset/samples/real_landmarks.jpg
```

These are generated verification images and are not part of the original dataset.

---

## 13. Initial Geometry Feature Test

File:

```text
dataset/extract_geometry.py
```

This script tests extraction of geometric measurements such as:

- left eye width
- right eye width
- eye distance
- mouth width
- mouth height
- face width
- face height
- nose-to-mouth distance
- eye width ratio
- mouth aspect ratio
- face aspect ratio
- nose-to-mouth ratio

Run:

```cmd
py extract_geometry.py
```

### Important

The initial test revealed that the first mouth aspect-ratio calculation could become extremely large because the selected mouth-height landmarks were very close together.

Therefore, this initial feature set should **not** be treated as the final ML feature set. It was a validation step.

---

## 14. Geometry + Symmetry Feature Extraction

Current development stage:

**Geometry + Symmetry Feature Engineering**

File:

```text
dataset/extract_features.py
```

The current experiment calculates feature groups including:

### Geometry

- eye widths
- eye heights
- eye distance
- mouth width
- mouth height
- face width
- face height
- nose width
- nose-to-mouth distance
- eyebrow distance

### Ratios

- left eye aspect ratio
- right eye aspect ratio
- mouth aspect ratio
- face aspect ratio
- nose width ratio
- nose-mouth ratio

### Symmetry

- eye width symmetry
- eye height symmetry
- eye aspect-ratio symmetry
- eyebrow symmetry

### Angle

- nose angle

Run from the `dataset` directory:

```cmd
py extract_features.py
```

At the current stage this is a small test on one Fake image and one Real image.

---

## 15. What Happens After Feature Testing?

Once the feature calculations are validated, the next stage is:

```text
Many Images
     |
     v
478 Landmarks
     |
     v
Geometry Features
     +
Symmetry Features
     |
     v
Feature Dataset
     |
     v
Machine Learning
```

The feature dataset will eventually contain rows conceptually like:

```text
image_id | label | eye_distance | face_width | ... | symmetry
```

The final feature list will be finalized after validation.

---

## 16. Machine Learning Stage

The ML stage is not yet completed.

Planned models include:

1. Random Forest
2. Support Vector Machine (SVM)
3. Logistic Regression baseline

The model will receive extracted numerical features rather than raw images.

Evaluation will include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Do not train the final model until the feature extraction pipeline has been validated.

---

## 17. Explainability Stage

Explainability is a major part of DeepTrace.

The final system should answer both:

> "Is this image fake?"

and:

> "Which measurable facial characteristics contributed to this prediction?"

Possible methods include:

- feature importance
- SHAP
- LIME
- landmark visualization

A final result may conceptually look like:

```text
Prediction: DEEPFAKE
Confidence: 91%

Important contributing features:
- Eye symmetry deviation
- Facial proportion deviation
- Nose-to-mouth ratio
- Mouth geometry
```

The exact explanation will depend on the final trained model and validated features.

---

## 18. Backend

Backend technologies:

```text
Node.js
Express.js
PostgreSQL
```

Current backend files:

```text
backend/
├── db.js
├── server.js
├── package.json
├── package-lock.json
└── .env
```

Install dependencies:

```cmd
cd backend
npm install
```

Current packages include:

```text
express
pg
dotenv
cors
```

---

## 19. Environment Variables

The backend uses a `.env` file.

Structure:

```env
PORT=5000
DB_USER=postgres
DB_HOST=localhost
DB_NAME=deeptrace
DB_PASSWORD=YOUR_POSTGRES_PASSWORD
DB_PORT=5432
```

### Security

Never commit `.env` to GitHub.

Never put the actual PostgreSQL password in:

- source code
- Git commits
- public documentation
- screenshots

The `.env` file is ignored by Git.

---

## 20. PostgreSQL

Database:

```text
deeptrace
```

Local PostgreSQL:

```text
localhost:5432
```

The Node.js backend connects using the `pg` package.

PostgreSQL is intended to store application/prediction information such as:

- prediction
- confidence
- image metadata
- extracted features
- explanation
- model version
- timestamp

Large datasets and model files should not be stored directly in PostgreSQL.

---

## 21. Test Backend

From `backend`:

```cmd
node server.js
```

Expected:

```text
DeepTrace backend running on http://localhost:5000
```

Test the root endpoint in a browser:

```text
http://localhost:5000/
```

Expected:

```json
{
  "message": "DeepTrace backend is running"
}
```

Database test:

```text
http://localhost:5000/api/db-test
```

Expected response contains:

```json
{
  "message": "Database connected successfully!",
  "database": "deeptrace"
}
```

---

## 22. Frontend

The frontend uses React.

From the frontend directory:

```cmd
cd frontend
npm install
```

Start development server:

```cmd
npm run dev
```

Vite will display the local URL in the terminal.

The frontend communicates with the backend through HTTP/REST.

---

## 23. Application Architecture

The intended final architecture is:

```text
                 USER
                  |
                  v
          React Frontend
                  |
             HTTP / REST
                  |
                  v
        Node.js + Express
             /          \
            /            \
           v              v
   Python/ML Pipeline   PostgreSQL
           |
           v
 Geometry + Symmetry
           |
           v
     ML Classifier
           |
           v
  REAL / DEEPFAKE
           |
           v
 Explanation + Confidence
```

React does **not** connect directly to PostgreSQL.

Correct flow:

```text
React
  |
  v
Node.js / Express
  |
  v
PostgreSQL
```

---

## 24. Running the Complete Project — Planned

The complete system is still under development.

The intended startup sequence will eventually be:

### Terminal 1 — Backend

```cmd
cd "C:\Users\piush\OneDrive - presidencyuniversity.in\Sem 3\MINI Proj\deepfake-detection\backend"
node server.js
```

### Terminal 2 — Frontend

```cmd
cd "C:\Users\piush\OneDrive - presidencyuniversity.in\Sem 3\MINI Proj\deepfake-detection\frontend"
npm run dev
```

### ML pipeline

The final ML prediction service/process will be added to this section after it is implemented.

> **Do not treat this as the final complete-project run procedure yet.**

---

## 25. Git and GitHub Workflow

Check status:

```cmd
git status
```

Stage changes:

```cmd
git add .
```

Check staged files:

```cmd
git status
```

Commit:

```cmd
git commit -m "Describe the change"
```

Push:

```cmd
git push origin main
```

Example:

```cmd
git commit -m "Add facial landmark visualization and feature extraction"
```

---

## 26. Files That Should NOT Be Committed

Normally keep these out of Git:

```text
.env
node_modules/
dataset/raw/
dataset/processed/
generated datasets
large model files
temporary files
```

Always check before pushing:

```cmd
git status
```

Make sure passwords, `node_modules`, large datasets, and other generated files are not accidentally staged.

---

## 27. Common Warnings

### Hugging Face authentication warning

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is a warning about authentication/rate limits. It does not mean the dataset failed.

### TensorFlow oneDNN

```text
oneDNN custom operations are on
```

Informational, not an error.

### MediaPipe XNNPACK

```text
XNNPACK delegate for CPU
```

Normal CPU acceleration information.

### MediaPipe feedback manager

```text
Feedback manager requires a model with a single signature inference.
```

This warning does not prevent the current Face Landmarker setup from working.

If the script reports:

```text
Face detected!
Number of landmarks: 478
```

the landmark stage is working.

---

## 28. Troubleshooting

### Python command not working

Try:

```cmd
py --version
```

instead of:

```cmd
python --version
```

### MediaPipe `solutions` error

The project uses the newer Tasks API:

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

Do not switch the current implementation to the old:

```python
mp.solutions.face_mesh
```

API.

### Face Landmarker model not found

Verify:

```text
models/face_landmarker.task
```

exists.

When scripts are run from `dataset/`, they currently reference:

```text
../models/face_landmarker.task
```

### PostgreSQL command not recognized

If:

```cmd
psql --version
```

is not recognized, add the PostgreSQL `bin` directory to Windows PATH.

### Backend database connection fails

Check:

1. PostgreSQL is running.
2. Database name is `deeptrace`.
3. Username is correct.
4. Password in `.env` is correct.
5. Port is `5432`.
6. `.env` is inside `backend/`.

### React dependency problems

From `frontend/`:

```cmd
rmdir /s /q node_modules
del package-lock.json
npm install
npm run dev
```

---

## 29. Current Project Status

### Completed

- [x] Project repository created
- [x] Frontend/backend structure started
- [x] Node.js backend setup
- [x] PostgreSQL installed
- [x] `deeptrace` database created
- [x] Node.js → PostgreSQL connection tested
- [x] Hugging Face dataset selected
- [x] Dataset loaded successfully
- [x] Fake/Real labels verified
- [x] MediaPipe installed
- [x] Face Landmarker model downloaded
- [x] Face detection tested
- [x] 478 facial landmarks detected
- [x] Landmark visualization completed
- [x] Initial geometry extraction tested
- [x] Geometry + symmetry feature extraction started

### Current Stage

**Geometry + Symmetry Feature Engineering**

The next task is to validate the current feature set and then process a controlled portion of the dataset.

### Not Yet Completed

- [ ] Final geometry feature set
- [ ] Final symmetry feature set
- [ ] Larger-scale feature extraction
- [ ] Feature dataset creation
- [ ] Train/validation/test split
- [ ] ML model training
- [ ] Model comparison
- [ ] Model evaluation
- [ ] Explainable AI implementation
- [ ] Prediction API
- [ ] PostgreSQL prediction-history tables
- [ ] React upload interface
- [ ] React results interface
- [ ] Complete frontend/backend/ML integration
- [ ] End-to-end testing
- [ ] Final deployment

---

## 30. Development Roadmap

```text
1. Geometry feature validation
          |
          v
2. Symmetry feature validation
          |
          v
3. Process a controlled dataset sample
          |
          v
4. Create feature dataset
          |
          v
5. Train ML models
          |
          v
6. Evaluate and select model
          |
          v
7. Implement explainability
          |
          v
8. Build prediction API
          |
          v
9. Create PostgreSQL prediction tables
          |
          v
10. Build React upload/dashboard
          |
          v
11. Connect React → Backend → ML → PostgreSQL
          |
          v
12. End-to-end testing
          |
          v
13. Documentation and final presentation
```

---

## 31. Development Rule

For every new project stage:

1. Implement the code.
2. Test it on a small sample.
3. Check the output.
4. Fix problems.
5. Only then scale it to the larger dataset.
6. Update this guide.
7. Commit the working changes to Git.

This prevents errors from propagating into the full dataset and final ML model.

---

## 32. Quick Start — Current Development Stage

If the environment is already configured:

```cmd
cd "C:\Users\piush\OneDrive - presidencyuniversity.in\Sem 3\MINI Proj\deepfake-detection"
```

Then:

```cmd
cd dataset
```

Landmark test:

```cmd
py test_landmarks.py
```

Landmark visualization:

```cmd
py visualize_landmarks.py
```

Initial geometry test:

```cmd
py extract_geometry.py
```

Current geometry + symmetry test:

```cmd
py extract_features.py
```

---

## 33. Project Principle

DeepTrace is being developed incrementally.

> **Test small → verify → improve → scale → train → integrate.**
