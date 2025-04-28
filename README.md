# 💼 Customer Churn Predictor

This project is a **Streamlit** web app that predicts the probability of a customer churning (leaving a service) based on their features like geography, age, balance, credit score, and more.  
It uses a trained **TensorFlow** model and preprocessing tools like **OneHotEncoder**, **LabelEncoder**, and **StandardScaler**.

---

## 🚀 Demo

The app allows users to input customer data through an interactive sidebar and outputs:
- The predicted **churn probability**,
- A visual **progress bar** showing the risk,
- A clear message indicating if the customer is likely to churn or not.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure you have the following files in the project directory:**
   - `model.h5`
   - `onehot_encoder_geo.pkl`
   - `label_encoder_gender.pkl`
   - `scaler.pkl`

---

## 📦 Required Libraries

- `streamlit`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `pickle`

(They should all install automatically via `requirements.txt`.)

---

## 🧠 How It Works

- **Model Loading**: The app loads a trained TensorFlow model (`model.h5`) and preprocessing objects (`.pkl` files) at startup.
- **Input Collection**: Sidebar widgets collect user input.
- **Preprocessing**: Inputs are encoded and scaled to match the model's training format.
- **Prediction**: Model predicts the churn probability.
- **Display**: Results are shown with metrics, progress bar, and status messages.

---

## ▶️ Run the App Locally

Once everything is installed and set up:

```bash
streamlit run app.py
```

(Assuming your Python script is named `app.py`.)

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure (Typical)

```
📂 your-project/
├── app.py
├── model.h5
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## ✨ Features

- Interactive input sidebar.
- Real-time prediction and probability display.
- Friendly UI with progress bars and success/error indicators.
- Expandable section to show raw model output.

---

## 📝 Notes

- Make sure that the `model.h5` and `.pkl` files match the expected input format.
- If you retrain the model, also regenerate and save updated encoders and scaler files.

---

## 📜 License

This project is for educational and demonstration purposes.  
Feel free to customize or extend it!
