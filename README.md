# Bank Customer Churn Prediction  
A Streamlit-based web application that predicts whether a bank customer is likely to **stay** or **leave (churn)** using a trained Deep Learning model (TensorFlow) and Scikit-learn preprocessing.

---

## 🚀 Project Features
- Predicts bank customer churn using a **trained neural network model**.
- Uses:
  - Label Encoding for **Gender**
  - One-Hot Encoding for **Geography**
  - Standard Scaling for numerical features
- Simple & interactive **Streamlit UI**.
- Loads pretrained:
  - `model.h5` (TensorFlow model)
  - `label_encoder_gender.pkl`
  - `onehot_encoder_geo.pkl`
  - `scaler.pkl`

---

## 🧠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Pandas / NumPy**
- **Scikit-Learn**
- **Streamlit**
- **Pickle**

---
## 📂 Folder Structure
project/
│── model.h5
│── label_encoder_gender.pkl
│── onehot_encoder_geo.pkl
│── scaler.pkl
│── app.py
│── requirements.txt
│── .gitignore
│── README.md


---

## ▶️ How to Run the App

### 1️⃣ Install dependencies

pip install -r requirements.txt


### 2️⃣ Run Streamlit app


Your web app will open in the browser.

---

## 📝 Input Features Used

| Feature           | Type        |
|------------------|-------------|
| Geography        | Categorical (One-hot encoded) |
| Gender           | Categorical (Label encoded) |
| Age              | Numeric |
| Credit Score     | Numeric |
| Balance          | Numeric |
| Tenure           | Numeric |
| Number of Products | Numeric |
| Has Credit Card  | Binary |
| Is Active Member | Binary |
| Estimated Salary | Numeric |

---

## 🧮 Model Output

The model returns a **churn probability** between `0 to 1`.

- If probability > 0.5 → **Customer will leave**
- Else → **Customer will stay**

---

## 🎯 Example Prediction Output
Customer is likely to stay.
Churn Probability: 0.23


---

## 📊 Model Training (Summary)

The model was trained using:
- A Deep Neural Network built in **TensorFlow**
- Scaled numerical features (StandardScaler)
- Encoded categorical features (LabelEncoder + OneHotEncoder)
- Binary classification (Churn vs Not Churn)

Loss Function: **Binary Crossentropy**  
Optimizer: **Adam**  
Metrics: **Accuracy**

---
## 🖥️ How to Run This Project on Any System

# Follow these steps exactly — works on any Windows/Mac/Linux PC.

# 1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Bank_churn_prediction.git
cd Bank_churn_prediction

# 2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# 4️⃣ Train the Model
python src/train.py

# This will generate:

churn_model.h5

label_encoder_gender.pkl

label_encoder_geo.pkl

scaler.pkl

# All stored inside the model/ folder.

# 5️⃣ Run Prediction
python src/predict.py

# You will be prompted for input such as:
Geography
Gender
Age
Balance
Tenure
Credit Score
Estimated Salary

# Output Example:

Customer will NOT churn. (0)

## 📄 License
This project is open-source and free to use.

---

## ✨ Author
Developed by **Harsh Gupta**  
Feel free to star ⭐ the repository!



