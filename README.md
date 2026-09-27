# 💻 Laptop Price Predictor (End-to-End Machine Learning Web App)

An end-to-end Machine Learning web application built with **Flask**, **Scikit-Learn**, and **Python** to estimate laptop market prices based on hardware specifications, display quality, and operating systems.

---

## 🌟 Key Features

* **Intelligent Price Estimation**: Powered by a trained Scikit-Learn regression pipeline trained on comprehensive laptop market data.
* **100% Dataset-Driven Dropdowns**: All input fields (RAM, Storage, Weight, Screen Size, Resolution, etc.) are restricted to verified dataset values, preventing invalid inputs, negative numbers, or server crashes.
* **Dynamic Dependent Filtering**: Interactive frontend logic automatically links Brand to valid Operating Systems (e.g., selecting **Apple** dynamically locks the OS to **Mac**, while selecting PC brands displays Windows/Linux).
* **Smart "Not Sure / Standard" Auto-Defaults**: Everyday non-technical users who don't know their exact laptop weight or resolution can pick *"Not Sure / Standard"*, and the backend will automatically map to the median values for that laptop category.
* **Feature Engineering (PPI Calculation)**: Automatically computes screen pixel density ($\text{PPI}$) from screen resolution and physical display dimensions:
  $$\text{PPI} = \frac{\sqrt{X_{\text{res}}^2 + Y_{\text{res}}^2}}{\text{Screen Size}}$$
* **Responsive & Modern UI**: Clean 2-column responsive layout with real-time feedback, error alerts, and formatted currency outputs (e.g., `₹ 73,886`).

---

## 🛠️ Technology Stack

* **Backend**: Python 3.11, Flask
* **Machine Learning & Data**: Scikit-Learn, Pandas, NumPy, Pickle
* **Frontend**: HTML5, Vanilla CSS3 (Custom Responsive Grid), JavaScript (Dynamic Form Validation)

---

## 📁 Project Structure

```
Laptop-Price-Prediction/
├── app.py                           # Flask application, routing & ML inference logic
├── templates/
│   └── index.html                   # Responsive frontend interface & dynamic JS
├── pipe.pkl                         # Serialized Scikit-Learn ML pipeline
├── df.pkl                           # Serialized dataset metadata for dropdown options
├── laptop_data.csv                  # Raw laptop specification dataset
├── requirements.txt                 # Project dependencies
├── TESTING_AND_DEPLOYMENT_GUIDE.md  # Complete test cases & cloud deployment guide
└── README.md                        # Project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sAkhil2027/Laptop-Price-Prediction.git
cd Laptop-Price-Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🧪 Testing & Pre-Deployment Verification

A complete pre-deployment test suite with boundary checks, extreme value tests, and monotonicity checks is provided in:
👉 **[TESTING_AND_DEPLOYMENT_GUIDE.md](TESTING_AND_DEPLOYMENT_GUIDE.md)**

---

## 👤 Author
* **Akhil** - [sAkhil2027](https://github.com/sAkhil2027)
