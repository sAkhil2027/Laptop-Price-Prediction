# 💻 Laptop Price Prediction using Machine Learning

A complete end-to-end Machine Learning project that predicts the price of a laptop based on its specifications. The project includes **data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and model serialization** using Scikit-learn.

The model is trained using a **Random Forest Regressor** and saved for future deployment in web applications such as Streamlit or Flask.

---

# 🚀 Features

- 📊 Complete Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- ⚙️ Advanced Feature Engineering
- 🤖 Random Forest Regression Model
- 📈 Model Performance Evaluation
- 💾 Model Serialization using Pickle
- 🔥 Ready for Deployment
- 📓 Well-structured Jupyter Notebook

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Pickle | Model Serialization |
| Jupyter Notebook | Development Environment |

---

# 📁 Project Structure

```
.
├── Laptop_price_pre.ipynb      # Complete notebook
├── laptop_price.csv            # Dataset
├── pipe.pkl                    # Trained ML Pipeline
├── df.pkl                      # Processed dataset
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/laptop-price-prediction.git

cd laptop-price-prediction
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset

The notebook uses the dataset

```
laptop_price.csv
```

which contains laptop specifications such as:

- Company
- Product
- Type Name
- RAM
- Weight
- Touchscreen
- IPS Display
- Screen Resolution
- CPU
- GPU
- Memory
- Operating System
- Screen Size
- Price (Target Variable)

---

# 🔄 Project Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Train-Test Split
    │
    ▼
Random Forest Model
    │
    ▼
Performance Evaluation
    │
    ▼
Save Model
```

---

# 🧹 Data Preprocessing

The notebook performs several preprocessing steps:

- Remove unwanted columns
- Convert RAM from string to integer
- Convert Weight into numeric values
- Handle missing values
- Remove unsupported GPU brands
- Convert target variable using Log Transformation
- Prepare dataset for model training

---

# ⚙️ Feature Engineering

Several new features are created to improve prediction accuracy.

### Display Features

- Touchscreen Detection
- IPS Display Detection
- Screen Resolution
- Pixels Per Inch (PPI)

---

### CPU Features

CPU names are grouped into categories:

- Intel Core i3
- Intel Core i5
- Intel Core i7
- Other Intel Processor
- AMD Processor

---

### Storage Features

The storage column is decomposed into multiple features:

- HDD Capacity
- SSD Capacity
- Hybrid Storage
- Flash Storage

---

### GPU Features

Only GPU Brand is retained.

Examples:

- Intel
- NVIDIA
- AMD

---

### Operating System

Operating systems are grouped into:

- Windows
- Mac
- Others / Linux / No OS

---

# 📊 Exploratory Data Analysis

The notebook includes visualizations such as:

- Company vs Price
- Laptop Type vs Price
- RAM vs Price
- CPU Brand vs Price
- GPU Brand vs Price
- Operating System vs Price
- Touchscreen Analysis
- IPS Display Analysis
- Correlation Heatmap
- Scatter Plots
- Distribution Analysis

---

# 🤖 Machine Learning Model

The notebook uses a **Random Forest Regressor** for price prediction.

Model Parameters:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)
```

---

# 🔧 Machine Learning Pipeline

The project builds a complete Scikit-learn Pipeline.

Pipeline Components:

### Step 1

Column Transformer

- One-Hot Encoding
- Categorical Feature Transformation

↓

### Step 2

Random Forest Regressor

↓

Final Prediction

---

# 🏗 Architecture

```
Laptop Specifications
         │
         ▼
Data Cleaning
         │
         ▼
Feature Engineering
         │
         ▼
Column Transformer
         │
         ▼
One-Hot Encoding
         │
         ▼
Random Forest Regressor
         │
         ▼
Predicted Laptop Price
```

---

# 📈 Model Evaluation

The notebook evaluates the trained model using:

- R² Score
- Mean Absolute Error (MAE)

These metrics help measure prediction accuracy and overall model performance.

---

# 💾 Model Serialization

After training, the notebook saves:

```python
pipe.pkl
```

Trained Machine Learning Pipeline

and

```python
df.pkl
```

Processed Dataset

These files can be directly used in deployment projects.

---

# ▶️ Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open

```
Laptop_price_pre.ipynb
```

Run all cells sequentially.

---

# 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle
jupyter
```

Install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

---

# 🌟 Future Improvements

- 🚀 Deploy using Streamlit
- 🌐 Build a Flask/FastAPI backend
- 📱 Responsive web interface
- ☁️ Cloud deployment
- 🔄 Hyperparameter tuning
- 📊 Feature importance visualization
- 🧠 Compare multiple regression algorithms
- 📈 Cross-validation and model optimization
- 🐳 Docker support

---

# 💡 Applications

- E-commerce laptop price estimation
- Laptop recommendation systems
- Inventory pricing
- Second-hand laptop valuation
- Market price analysis
- Retail analytics

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push the branch

```bash
git push origin feature/new-feature
```

5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Akhil Vikram Singh**

If you found this project useful, consider giving it a ⭐ on GitHub!
