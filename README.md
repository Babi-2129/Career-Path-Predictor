# 🎓 Career Path Predictor – For High School Students

# Link:- https://career-path-predictor-by-babi-pepakayala.streamlit.app/

## 🎯 Project Overview  
This project aims to help high school students discover suitable career fields based on their academic performance, skills, and experiences. A user-friendly web application, built with Streamlit, takes various inputs from the student and utilizes a trained Machine Learning model to predict the most aligned career field. The application is designed for free deployment, making it accessible to anyone.

---

## ✨ Features  
- **Interactive Input Form:** Students can easily enter their GPA, extracurricular activities, internships, project details, leadership roles, specific course information, research experience, and various skill ratings (coding, communication, problem-solving, teamwork, analytical, presentation, networking, industry certifications).  
- **Machine Learning Prediction:** Uses a Random Forest Classifier trained on a diverse dataset to predict the most suitable career field (e.g., Engineering, Law, Biology).  
- **User-Friendly Interface:** Organized input sections, clear visual separators, and a responsive layout.  
- **Developer Information Tab:** A dedicated section to display developer contact details, LinkedIn profile, and portfolio.  
- **Cloud Deployment Ready:** Designed to be easily deployed on platforms like Streamlit Cloud.

---

## 🚀 Technologies Used  
- **Machine Learning:** Python, Scikit-learn (RandomForestClassifier, LabelEncoder)  
- **Data Handling:** Pandas  
- **Web Application:** Streamlit  
- **Model Persistence:** Joblib  
- **File Downloads:** Requests  
- **Version Control:** Git, GitHub  
- **Deployment:** Streamlit Cloud (with large model file hosted on Google Drive)

---

## 🗺️ Development Journey & Phases Followed  

### **Phase 1: Project Setup & Objective Definition**  
- Defined scope, final goal (free web app deployment), and target audience (high school students).

### **Phase 2: Data Collection & Cleaning**  
- Used a cleaned dataset (`career_path_in_all_field.csv`) with 9000 samples and 17 columns.  
- Ensured all features were in a numerical format suitable for ML.

### **Phase 3: Exploratory Data Analysis (EDA)**  
- Analyzed data shape, types, and distributions.  
- Visualized the `Field` target variable distribution.  
- Grouped analysis by career field for deeper insights.

### **Phase 4: Model Building & Evaluation**  
- Selected **Random Forest Classifier**.  
- Encoded the target variable with LabelEncoder.  
- Performed 80/20 train-test split.  
- Evaluated with accuracy, classification report, and confusion matrix.  
- Saved trained model and label encoder as `.pkl` files.

### **Phase 5: Web Application Development (Front-end)**  
- Built using **Streamlit (`web.py`)**.  
- Implemented input widgets for all 15 features.  
- Used visual separators and tabs for better UX.  
- Added a button linking to the developer’s LinkedIn.

### **Phase 6: Deployment Preparation & Execution**  
- Addressed Streamlit Cloud's 25MB upload limit by hosting `.pkl` files on **Google Drive**.  
- Implemented runtime download of models using `requests` and `st.cache_resource`.  
- Created `requirements.txt` and organized files for GitHub.

---

## 📂 Project Structure & File Formats  

```
career_predictor_app/
├── web.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── career_path_in_all_field.csv  # Dataset
├── Model Training/
│   ├── best_career_path_predictor_model.pkl  # Trained model
│   └── field_label_encoder.pkl              # Label encoder
└── README.md                     # This file
```

---

## 🏃‍♀️ How to Run Locally

### 1. Clone the Repository
```bash
git clone YOUR_GITHUB_REPO_URL_HERE
cd career_predictor_app
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
.env\Scriptsctivate   # For Windows
source venv/bin/activate # For macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model & Encoder  
Manually download the following:
- `best_career_path_predictor_model.pkl` from [Google Drive Link]
- `field_label_encoder.pkl` from [Google Drive Link]

Place both files inside the `Model Training/` folder.

### 5. Run the App
```bash
streamlit run web.py
```
Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deployment (Streamlit Cloud)

1. Push your code to a public GitHub repo (excluding large `.pkl` files).  
2. Visit [Streamlit Cloud](https://share.streamlit.io).  
3. Connect your GitHub.  
4. Click **"New app"**, select your repo, set `web.py` as the main file, and hit **Deploy**.  
5. Your app will be publicly available with a shareable URL.

---

## 👨‍💻 Author  
**Babi Pepakayala**  
📧 Email: babipepakayala162129@gmail.com  
🔗 LinkedIn: [Babi Pepakayala](https://www.linkedin.com/in/babi-pepakayala/)
🌐 Portfolio: [View Me 🥰](https://babi-2129.github.io/portfolio-website/)
