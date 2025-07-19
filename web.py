import streamlit as st
import pandas as pd
import joblib
import os
import requests # NEW: Import requests library for downloading

# Define the base directory for your project
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, '..', 'Model Training')

# Construct the full path to the label encoder file (it's on GitHub now)
label_encoder_path = os.path.join(model_dir, 'field_label_encoder.pkl')

# --- NEW: Your Model's Public Google Drive Download URL ---
# IMPORTANT: This is the DIRECT download link.
# Converted from: https://drive.google.com/file/d/1TCSOmaOnfcBjDFYf_57GQK0-Y7EmtvDC/view?usp=drive_link
MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1TCSOmaOnfcBjDFYf_57GQK0-Y7EmtvDC"
LOCAL_MODEL_FILENAME = "best_career_path_predictor_model.pkl" # This is where the downloaded model will be saved

# NEW: Function to download model from URL
@st.cache_resource # Cache the model so it's only downloaded once across reruns
def load_model_from_url(url, local_filename):
    st.info(f"Downloading model...") # Removed URL from message for conciseness
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (e.g., 404)
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")
        return joblib.load(local_filename)
    except Exception as e:
        st.error(f"Failed to download or load model: {e}")
        st.stop()


# Load your trained model and label encoder
try:
    model = load_model_from_url(MODEL_DOWNLOAD_URL, LOCAL_MODEL_FILENAME) # Load model from URL
    label_encoder = joblib.load(label_encoder_path) # Load label encoder locally from GitHub
except FileNotFoundError:
    st.error(f"Error: Label encoder file not found at: {label_encoder_path}. Please ensure it's in Model Training folder on GitHub.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during model/encoder loading: {e}")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Career Path Predictor", layout="centered")

# Custom CSS for styling and addressing number color
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Default text color */
    }
    /* Profile Image Styling */
    .profile-img {
        width: 260px; /* Big size */
        height: 260px; /* Big size */
        border-radius: 50%; /* Rounded */
        object-fit: cover; /* Ensures image covers the area without distortion */
        background-color: #007bff; /* Using the primary blue color from other elements */
        padding: 10px;
        box-shadow: 0 0 20px rgba(0, 123, 255, 0.5); /* Blue glow shadow */
        margin-bottom: 20px; /* Space below image */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 12px 20px 0 rgba(0,0,0,0.24), 0 10px 30px 0 rgba(0,0,0,0.19);
    }
    .stSuccess {
        background-color: #e6ffe6; /* Light green for success */
        color: #006600; /* Dark green text */
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
    }
    .stInfo {
        background-color: #e0f2f7; /* Light blue for info */
        color: #0056b3; /* Dark blue text */
        border-radius: 10px;
        padding: 10px;
        border-left: 5px solid #007bff;
    }
    h1 {
        color: #2c3e50; /* Dark blue-gray for main title */
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    h3 {
        color: #34495e; /* Slightly lighter blue-gray for subtitles */
        text-align: center;
        font-size: 1.8em;
        margin-top: 0;
    }
    h2 {
        color: #2980b9; /* A nice blue for section headers */
        border-bottom: 2px solid #2980b9;
        padding-bottom: 5px;
        margin-top: 2em;
    }
    /* --- CSS FOR NUMBERS (targeting more broadly) --- */
    input[data-testid="stNumberInput-StyledInput"] {
        color: #333333 !important;
    }
    .stSlider .st-bh span, .stSlider .st-bd span {
        color: #333333 !important; /* Force dark gray */
    }
    .stApp > header + div > div > div > div > div > div > div > div > span {
        color: #333333 !important;
    }

    .stSlider > div > div > div:nth-child(1) {
        background-color: #3498db;
    }
    .stSlider > div > div > div:nth-child(2) {
        background-color: #2980b9;
    }
    /* --- END CSS --- */

    .developer-section {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
        margin-top: 1em;
        border: 1px solid #dee2e6;
        text-align: center;
        color: #495057;
    }
    .developer-section a {
        color: #007bff;
        text-decoration: none;
    }
    .developer-section a:hover {
        text-decoration: underline;
    }
    /* Styling for tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stIcon"] {
        margin-right: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 10px 20px;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        border: 1px solid #dee2e6;
        border-bottom: none;
        font-size: 1.1em;
        color: #495057;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #007bff;
        color: #007bff;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff;
        padding: 20px;
        border: 1px solid #dee2e6;
        border-radius: 0 0 8px 8px;
        margin-top: -1px;
    }
</style>
""", unsafe_allow_html=True)


st.title("🧑‍🎓 Career Path Predictor")
st.markdown("### <span style='color:#34495e;'>Discover Your Ideal Career Field Based on Your Strengths!</span>", unsafe_allow_html=True)
st.write("Please provide your academic and skill details below. Most inputs are on a scale of 0-10 unless specified.")

# --- Define Expected Features (from your X dataframe) ---
feature_columns = [
    'GPA', 'Extracurricular_Activities', 'Internships', 'Projects',
    'Leadership_Positions', 'Field_Specific_Courses', 'Research_Experience',
    'Coding_Skills', 'Communication_Skills', 'Problem_Solving_Skills',
    'Teamwork_Skills', 'Analytical_Skills', 'Presentation_Skills',
    'Networking_Skills', 'Industry_Certifications'
]


# Create tabs
tab1, tab2 = st.tabs(["📊 Prediction Tool", "👨‍💻 Developer Info"])

with tab1:
    st.header("Your Profile Details:")

    st.subheader("📊 General Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        gpa = st.number_input("GPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.01, key='gpa_input_tab1')
    with col2:
        extracurricular_activities = st.slider("Extracurricular Activities (Count)", 0, 10, 3, key='extracurricular_input_tab1')
    with col3:
        internships = st.slider("Internships (Count)", 0, 5, 1, key='internships_input_tab1')

    st.markdown("---") # Visual separator between categories

    st.subheader("💡 Projects & Leadership")
    projects = st.slider("Projects (Count)", 0, 10, 2, key='projects_input_tab1')
    leadership_positions = st.slider("Leadership Positions (Count)", 0, 5, 0, key='leadership_input_tab1')

    st.markdown("---") # Visual separator between categories

    st.subheader("📚 Academic & Research")
    field_specific_courses = st.slider("Field-Specific Courses (Count)", 0, 15, 5, key='courses_input_tab1')
    research_experience = st.slider("Research Experience (Years/Count)", 0, 5, 0, key='research_input_tab1')

    st.markdown("---") # Visual separator between categories

    st.subheader("🛠️ Core Skills (0-5 Scale)")
    coding_skills = st.slider("Coding Skills", 0, 5, 3, key='coding_input_tab1')
    communication_skills = st.slider("Communication Skills", 0, 5, 4, key='communication_input_tab1')
    problem_solving_skills = st.slider("Problem Solving Skills", 0, 5, 4, key='problem_solving_input_tab1')
    teamwork_skills = st.slider("Teamwork Skills", 0, 5, 4, key='teamwork_input_tab1')
    analytical_skills = st.slider("Analytical Skills", 0, 5, 4, key='analytical_input_tab1')
    presentation_skills = st.slider("Presentation Skills", 0, 5, 3, key='presentation_input_tab1')

    st.markdown("---") # Visual separator between categories

    st.subheader("🏆 Other Skills & Certifications")
    networking_skills = st.slider("Networking Skills", 0, 5, 2, key='networking_input_tab1')
    industry_certifications = st.slider("Industry Certifications (Count)", 0, 5, 1, key='certifications_input_tab1')


    # --- Prediction Button ---
    st.markdown("---") # Visual separator
    if st.button("🌟 Predict My Career Path", key='predict_button_tab1'):
        input_data_dict = {
            'GPA': gpa,
            'Extracurricular_Activities': extracurricular_activities,
            'Internships': internships,
            'Projects': projects,
            'Leadership_Positions': leadership_positions,
            'Field_Specific_Courses': field_specific_courses,
            'Research_Experience': research_experience,
            'Coding_Skills': coding_skills,
            'Communication_Skills': communication_skills,
            'Problem_Solving_Skills': problem_solving_skills,
            'Teamwork_Skills': teamwork_skills,
            'Analytical_Skills': analytical_skills,
            'Presentation_Skills': presentation_skills,
            'Networking_Skills': networking_skills,
            'Industry_Certifications': industry_certifications
        }

        input_df = pd.DataFrame([input_data_dict], columns=feature_columns)

        predicted_field = model.predict(input_df)[0]

        st.balloons()
        st.success(f"## Your Predicted Career Field: **{predicted_field}**")
        st.info("💡 Remember, this is a prediction based on your input and the model's training. It's a guide to explore further!")

     
    st.markdown("---") # Another visual separator before the button
    # Direct link button
    st.caption("Developed as part of the Career Path Predictor Project")
    st.markdown(f"""
<div class="developer-section">
    <h4>Developed by: Babi Pepakayala</h4>
    <p>Check developer's profile for more derails</p>
    <p style="text-align: center;">
</div>
""", unsafe_allow_html=True)


# --- Developer Section ---
with tab2:
    st.header("👨‍💻 Developer Information")
    st.markdown(f"""
    <div class="developer-section">
        <h4 style="color:#2c3e50;">Developed by: Babi Pepakayala</h4>
        <p style="text-align: center;">
            <img src="https://res.cloudinary.com/dtdqxhkm4/image/upload/linkedin_profile_edg9by.jpg" alt="Babi Pepakayala" class="profile-img">
        </p>
        <p>For any queries, contact me at: <a href="mailto:babipepakayala162129@gmail.com">babipepakayala162129@gmail.com</a></p>
        <p>LinkedIn Profile: <a href="https://www.linkedin.com/in/babi-pepakayala/" target="_blank">Babi Pepakayala</a></p>
        <p>My Portfolio: <a href="https://babi-2129.github.io/portfolio-website/  " target="_blank">View Me 🥰 </a></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True) # Add some space