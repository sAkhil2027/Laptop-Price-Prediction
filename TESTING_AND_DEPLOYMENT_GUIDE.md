# Laptop Price Prediction - Pre-Deployment & Testing Guide

This document contains a comprehensive testing plan, test cases, sanity checks, and deployment instructions to verify and prepare your application before going live.

---

## 1. Pre-Deployment Test Suite

### Test Suite 1: Boundary & Sanity Checks (Extreme Hardware Specs)

| Test ID | Test Scenario | Inputs | Expected Behavior & Price Range | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | **Ultra-Budget / Minimum Specs** | Brand: `Acer`, Type: `Notebook`, RAM: `2 GB`, Weight: `1.5 kg`, Screen: `11.6"`, Res: `1366x768`, CPU: `Other Intel Processor`, HDD: `500 GB`, SSD: `0 GB`, GPU: `Intel`, OS: `Windows` | Should output an entry-level price between **₹ 15,000 – ₹ 28,000**. | [ ] |
| **TC-02** | **Mid-Range Standard Workstation** | Brand: `Dell`, Type: `Notebook`, RAM: `8 GB`, Weight: `2.0 kg`, Screen: `15.6"`, Res: `1920x1080`, CPU: `Intel Core i5`, HDD: `1000 GB`, SSD: `256 GB`, GPU: `Intel`, OS: `Windows` | Should output a standard mainstream price between **₹ 45,000 – ₹ 65,000**. | [ ] |
| **TC-03** | **Flagship / High-End Gaming** | Brand: `Razer` / `MSI`, Type: `Gaming`, RAM: `32 GB` or `64 GB`, Weight: `2.8 kg`, Screen: `17.3"`, Res: `3840x2160`, CPU: `Intel Core i7`, HDD: `0 GB`, SSD: `1024 GB`, GPU: `Nvidia`, OS: `Windows` | Should output a premium price between **₹ 1,50,000 – ₹ 3,00,000+**. | [ ] |
| **TC-04** | **Apple Ultrabook Baseline** | Brand: `Apple`, Type: `Ultrabook`, RAM: `8 GB`, Weight: `1.37 kg`, Screen: `13.3"`, Res: `2560x1600`, CPU: `Intel Core i5`, HDD: `0 GB`, SSD: `128 GB` / `256 GB`, GPU: `Intel`, OS: `Mac` | Should output realistic Apple pricing between **₹ 70,000 – ₹ 1,10,000**. | [ ] |

---

### Test Suite 2: Monotonicity & Component Upgrade Checks

Verify that upgrading individual components logically increases the estimated price:

* **RAM Upgrade Test**:
  * Keep all specs identical (e.g., HP Notebook, i5, 256GB SSD, 1080p).
  * Test: `4 GB` $\rightarrow$ `8 GB` $\rightarrow$ `16 GB` $\rightarrow$ `32 GB`.
  * *Check*: Price must steadily increase as RAM increases.
* **GPU Upgrade Test**:
  * Keep all specs identical and switch GPU from `Intel (Integrated)` $\rightarrow$ `Nvidia (Dedicated)`.
  * *Check*: Price should noticeably rise for dedicated graphics.
* **Display Quality Test**:
  * Switch Resolution from `1366x768` $\rightarrow$ `1920x1080` $\rightarrow$ `3840x2160 (4K)`.
  * *Check*: Higher PPI and 4K screen should produce higher estimated valuation.
* **Storage Speed Test**:
  * Compare `1000 GB HDD (0 SSD)` vs. `256 GB / 512 GB SSD (0 HDD)`.
  * *Check*: SSD configurations should reflect modern market valuation over spinning HDD.

---

### Test Suite 3: Domain Logic & Dependent Dropdown Filtering

* **Apple Brand Restriction**:
  * Select `Brand = Apple`.
  * *Check*: OS dropdown must automatically lock and only show **`Mac`**.
* **PC Brand OS Validation**:
  * Select `Brand = Dell`, `HP`, or `Lenovo`.
  * *Check*: OS dropdown must only show **`Windows`** and **`Others/No OS/Linux`** (hiding `Mac`).
* **Chromebook Brand Validation**:
  * Select `Brand = Google`.
  * *Check*: OS dropdown must set to **`Others/No OS/Linux`**.

---

### Test Suite 4: Calculation & Boundary Edge Cases

* **"Not Sure / Standard" Auto Defaults (Approach B)**:
  * Select `Weight = Not Sure / Standard` and `Resolution = Not Sure / Standard`.
  * *Check*: App automatically maps them to realistic dataset medians (e.g., Notebook weight $\approx 2.1$ kg, Ultrabook $\approx 1.3$ kg, Resolution $= 1920\times 1080$) and predicts accurately without errors.
* **Zero Storage (0 HDD + 0 SSD)**:
  * Select `HDD = 0 GB (No HDD / Not Sure)` and `SSD = 0 GB`.
  * *Check*: App must not throw `ZeroDivisionError` or `ValueError` and should compute a baseline price.
* **High PPI Display**:
  * Screen Size = `11.6"` + Resolution = `3840x2160`.
  * *Check*: App calculates PPI correctly without float overflow.
* **Low PPI Display**:
  * Screen Size = `18.4"` + Resolution = `1366x768`.
  * *Check*: App calculates PPI correctly without underflow.

---

### Test Suite 5: UX & Server Robustness

1. **Form Persistence**: After submitting the form, ensure that all selected dropdowns remain populated on your chosen values.
2. **Double / Rapid Click**: Click "Predict Estimated Price" rapidly multiple times. Ensure the server answers smoothly without hanging.
3. **Mobile Layout**: Shrink browser width to under 580px and verify that the 2-column grid stacks into a clean 1-column layout.

---

## 2. Deployment Readiness Checklist

Before publishing to cloud platforms (Render, Railway, PythonAnywhere, AWS):

- [x] **Input Restriction**: All inputs are dropdowns (`<select>`) derived from dataset values.
- [x] **Error Handling**: `try...except` block wrapped around model inference in `app.py`.
- [x] **Dependent UI Filtering**: Client-side JavaScript links Brand to valid OS.
- [x] **Backend OS Tamper Protection**: Server enforces valid OS for brand even if manipulated.
- [ ] **Clean Production `requirements.txt`**: Minimal UTF-8 dependency list without Windows-only or Jupyter dependencies.
- [ ] **Procfile / WSGI Server**: Configured with `gunicorn` for production cloud deployment.
- [ ] **Dynamic Port Binding**: `app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))`.

---

## 3. Recommended Cloud Deployment Configuration

### Minimal Production `requirements.txt`:
```txt
Flask==3.0.0
gunicorn==21.2.0
scikit-learn==1.8.0
numpy==2.4.0
pandas==2.3.3
Jinja2==3.1.6
```

### `Procfile` (For Render / Heroku / Railway):
```
web: gunicorn app:app
```
