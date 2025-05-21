import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdfplumber
import os
import pytesseract
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from google.api_core import exceptions
import base64
import json
from typing import Optional, Dict, List
from datetime import datetime
import re

# ------------------- Load environment variables -------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# ------------------- Constants -------------------
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
ANALYSIS_TEMPLATE = """
## 1. Key Findings
[Concise bullet points of important findings]

## 2. Potential Diagnoses
[List of possible diagnoses with confidence levels]

## 3. Medication Recommendations
[Suggested medicines with effectiveness percentages, dosages, and side effects]

## 4. Lifestyle Guidance
[Diet plans (vegetarian/non-vegetarian), exercise routines, sleep advice]

## 5. Disease Classification
[Classification into chronic, infectious, common illnesses etc.]

## 6. Next Steps
[Recommended follow-up actions and timeline]
"""

# ------------------- Animation Assets -------------------
def load_lottie(filepath: str) -> Optional[Dict]:
    """Load Lottie animation from file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# Load local Lottie files (better performance than URLs)
lottie_upload = load_lottie("assets/upload.json")
lottie_scan = load_lottie("assets/scan.json")
lottie_success = load_lottie("assets/success.json")
lottie_doctor = load_lottie("assets/doctor.json")

# ------------------- Custom CSS -------------------
def set_custom_style():
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #10b981;
            --dark: #1e293b;
            --light: #f8fafc;
            --danger: #ef4444;
            --warning: #f59e0b;
        }}
       
        .stApp {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--light);
        }}
       
        /* Main container */
        .main-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
       
        /* Cards */
        .card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }}
       
        /* Buttons */
        .stButton>button {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            width: 100%;
        }}
       
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            background: linear-gradient(135deg, var(--primary-dark) 0%, #4338ca 100%);
        }}
       
        /* File uploader */
        .stFileUploader>div>div {{
            background: rgba(30, 41, 59, 0.5) !important;
            border: 2px dashed rgba(255, 255, 255, 0.2) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
        }}
       
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
       
        .stTabs [data-baseweb="tab"] {{
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px 12px 0 0 !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s ease;
        }}
       
        .stTabs [aria-selected="true"] {{
            background: var(--primary) !important;
            color: white !important;
        }}
       
        /* Custom elements */
        .highlight-box {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.2) 100%);
            border-left: 4px solid var(--primary);
            border-radius: 0 8px 8px 0;
            padding: 1rem;
            margin: 1rem 0;
        }}
       
        .medicine-card {{
            background: rgba(30, 41, 59, 0.7);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border-left: 4px solid var(--primary);
        }}
       
        .medicine-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }}
       
        .success-animation {{
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }}
       
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .main-container {{
                padding: 1rem;
            }}
           
            .card {{
                padding: 1rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------- AI Analysis -------------------
def analyze_medical_report(content: str) -> str:
    """Analyze medical report content with Gemini AI"""
    prompt = f"""
    Analyze this medical report and provide structured output with these exact section headers:

    {ANALYSIS_TEMPLATE}

    Additional requirements:
    - Use concise, professional language
    - Include relevant emojis for readability
    - For medications, provide exact dosages when possible
    - For diagnoses, include confidence percentages
    - Format with clear markdown headers
    """
   
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(f"{prompt}\n\n{content}")
            if response and hasattr(response, 'text'):
                return response.text
            raise ValueError("Invalid response from AI")
        except (exceptions.GoogleAPIError, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report. Error: {str(e)}")
                return "AI analysis failed. Please try again later."

# ------------------- Text Extraction -------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF file"""
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_image(image) -> str:
    """Extract text from image using OCR"""
    return pytesseract.image_to_string(image)

# ------------------- Result Processing -------------------
def extract_section(content: str, section_title: str) -> str:
    """Extract specific section from analysis results"""
    if section_title not in content:
        return f"## {section_title}\nNo information found in this section."
   
    section_start = content.index(section_title)
    remaining_content = content[section_start:]
   
    # Find the next section header or end of content
    next_section_pos = remaining_content.find("## ", len(section_title))
    if next_section_pos != -1:
        return remaining_content[:next_section_pos].strip()
    return remaining_content.strip()

def parse_medications(content: str) -> List[Dict]:
    """Parse medication recommendations from analysis results"""
    medications = []
    med_section = extract_section(content, "## 3. Medication Recommendations")
    
    # Split into individual medication entries
    med_entries = re.split(r"\n\s*[-*]\s+", med_section)[1:]  # Skip first empty split
    
    for entry in med_entries:
        med_data = {}
        lines = [line.strip() for line in entry.split("\n") if line.strip()]
        
        if not lines:
            continue
            
        # First line is medication name and effectiveness
        first_line = lines[0]
        med_data["name"] = first_line.split("(")[0].split(":")[0].strip()
        
        # Extract effectiveness if available
        eff_match = re.search(r"(\d{1,3})%", first_line)
        if eff_match:
            med_data["effectiveness"] = int(eff_match.group(1))
        elif "Effectiveness:" in first_line:
            # Handle cases where effectiveness is mentioned but not as percentage
            eff_text = first_line.split("Effectiveness:")[1].strip()
            if "%" in eff_text:
                eff_match = re.search(r"(\d{1,3})%", eff_text)
                if eff_match:
                    med_data["effectiveness"] = int(eff_match.group(1))
            else:
                # Try to extract any number
                num_match = re.search(r"\d+", eff_text)
                if num_match:
                    med_data["effectiveness"] = int(num_match.group())
        
        # Process remaining lines for details
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                med_data[key] = value.strip()
            elif line.startswith("(") and line.endswith(")"):
                # Handle additional info in parentheses
                med_data["additional_info"] = line.strip("()")
        
        medications.append(med_data)
    
    return medications

# ------------------- Visualizations -------------------
def plot_medicine_effectiveness(medications: List[Dict]):
    """Create a bar chart of medication effectiveness"""
    if not medications:
        st.info("No medication data available for visualization")
        return
    
    # Prepare data - extract effectiveness values
    names = []
    effectiveness = []
    
    for med in medications:
        name = med.get("name", "Unknown Medication")
        eff = med.get("effectiveness", 0)
        
        # Handle different effectiveness formats
        if isinstance(eff, str):
            # Extract first number from strings like "70-80%"
            numbers = re.findall(r'\d+', eff)
            if numbers:
                eff = int(numbers[0])
            else:
                eff = 0  # Default if no number found
        
        names.append(name)
        effectiveness.append(eff)
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gradient bars
    colors = plt.cm.viridis([eff/100 for eff in effectiveness])
    bars = ax.barh(names, effectiveness, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width}%',
                ha='left', va='center',
                color='white', fontsize=10)
    
    # Styling
    ax.set_xlim(0, 100)
    ax.set_xlabel('Effectiveness (%)')
    ax.set_title('Medication Effectiveness Comparison', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.spines['bottom'].set_color('#64748b')
    ax.tick_params(colors='#94a3b8', which='both')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    
    st.pyplot(fig)

def create_diagnosis_chart(content: str):
    """Create a pie chart of disease classification"""
    labels = ['Chronic', 'Infectious', 'Common', 'Other']
    sizes = [25, 25, 25, 25]  # Default distribution
    
    if "## 5. Disease Classification" in content:
        classification = extract_section(content, "## 5. Disease Classification").lower()
        if "chronic" in classification:
            sizes = [40, 20, 20, 20]
        if "infectious" in classification:
            sizes = [20, 40, 20, 20]
        if "common" in classification:
            sizes = [20, 20, 40, 20]
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Custom colors
    colors = ['#6366f1', '#10b981', '#f59e0b', '#64748b']
    
    # Explode the dominant slice
    max_index = sizes.index(max(sizes))
    explode = [0.1 if i == max_index else 0 for i in range(len(sizes))]
    
    # Plot
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, shadow=True,
        wedgeprops={'linewidth': 1, 'edgecolor': '#1e293b'},
        textprops={'color': 'white', 'fontsize': 10}
    )
    
    # Style
    ax.set_title('Disease Classification', color='white', pad=20)
    plt.setp(autotexts, size=10, weight="bold", color="white")
    fig.patch.set_facecolor('none')
    
    st.pyplot(fig)

# ------------------- UI Components -------------------
def render_upload_section():
    """Render the file upload section"""
    with st.container():
        st.markdown("""
        <div class="card">
            <h3 style="color: #f8fafc; margin-bottom: 1.5rem;">Upload Your Medical Report</h3>
        """, unsafe_allow_html=True)
       
        col1, col2 = st.columns([1, 2])
        with col1:
            file_type = st.radio("File Type:", ("PDF", "Image"), horizontal=True)
        with col2:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
       
        st.markdown("</div>", unsafe_allow_html=True)
       
        return file_type, uploaded_file

def render_upload_success():
    """Show upload success animation"""
    if lottie_success:
        with st.container():
            st.markdown("""
            <div class="success-animation">
            """, unsafe_allow_html=True)
            st_lottie(lottie_success, height=150, key="upload-success")
            st.markdown("""
            </div>
            <h4 style="text-align: center; color: #10b981; margin-top: -1rem;">
                File uploaded successfully!
            </h4>
            """, unsafe_allow_html=True)

def render_medication_cards(medications: List[Dict]):
    """Render medication cards with details"""
    if not medications:
        st.info("No specific medication recommendations found")
        return
    
    for med in medications:
        effectiveness = med.get("effectiveness", "N/A")
        dosage = med.get("dosage", "Not specified")
        side_effects = med.get("side_effects", "Not specified")
        
        # Handle both string and integer effectiveness values
        if isinstance(effectiveness, str) and "%" in effectiveness:
            try:
                effectiveness = int(effectiveness.replace("%", "").strip())
            except ValueError:
                pass
        
        st.markdown(f"""
        <div class="medicine-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #f8fafc;">{med.get('name', 'Unnamed Medication')}</h4>
                <div style="background: {'#10b981' if isinstance(effectiveness, int) and effectiveness > 70 else '#f59e0b' if isinstance(effectiveness, int) and effectiveness > 40 else '#ef4444'};
                            color: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">
                    {effectiveness if isinstance(effectiveness, int) else effectiveness}% effective
                </div>
            </div>
            <div style="margin-top: 0.5rem;">
                <p style="margin: 0.25rem 0; color: #94a3b8;"><strong>Dosage:</strong> {dosage}</p>
                <p style="margin: 0.25rem 0; color: #94a3b8;"><strong>Side Effects:</strong> {side_effects}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_analysis_tabs(analysis_result: str):
    """Render the analysis results in tabs with improved layout"""
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💊 Medications", "🧘 Lifestyle", "📊 Insights"])
    
    with tab1:
        st.markdown("""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">📋</span>
                <h3 style="color: #f8fafc; margin: 0;">Key Findings</h3>
            </div>
        """, unsafe_allow_html=True)
       
        key_findings = extract_section(analysis_result, "## 1. Key Findings")
        st.markdown(key_findings.replace("## 1. Key Findings", "").strip(), unsafe_allow_html=True)
       
        st.markdown("""
            <div class="card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">🩺</span>
                <h3 style="color: #f8fafc; margin: 0;">Potential Diagnoses</h3>
            </div>
        """, unsafe_allow_html=True)
       
        diagnoses = extract_section(analysis_result, "## 2. Potential Diagnoses")
        st.markdown(diagnoses.replace("## 2. Potential Diagnoses", "").strip(), unsafe_allow_html=True)
       
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">💊</span>
                <h3 style="color: #f8fafc; margin: 0;">Medication Recommendations</h3>
            </div>
        """, unsafe_allow_html=True)
       
        medications = parse_medications(analysis_result)
       
        if not medications:
            st.markdown("""
            <div class="highlight-box" style="margin: 1rem 0;">
                <p style="margin: 0; color: #94a3b8;">
                    ⚠️ No specific medication recommendations were found in the analysis.
                    Please consult with your healthcare provider for personalized advice.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_medication_cards(medications)
            
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">📈</span>
                    <h3 style="color: #f8fafc; margin: 0;">Effectiveness Comparison</h3>
                </div>
            """, unsafe_allow_html=True)
            
            plot_medicine_effectiveness(medications)
       
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">🧘</span>
                <h3 style="color: #f8fafc; margin: 0;">Lifestyle Recommendations</h3>
            </div>
        """, unsafe_allow_html=True)
       
        lifestyle = extract_section(analysis_result, "## 4. Lifestyle Guidance")
        st.markdown(lifestyle.replace("## 4. Lifestyle Guidance", "").strip(), unsafe_allow_html=True)
       
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">📊</span>
                <h3 style="color: #f8fafc; margin: 0;">Health Insights</h3>
            </div>
        """, unsafe_allow_html=True)
       
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🏷️</span>
                    <h4 style="color: #f8fafc; margin: 0;">Disease Classification</h4>
                </div>
            """, unsafe_allow_html=True)
            create_diagnosis_chart(analysis_result)
       
        with col2:
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🔄</span>
                    <h4 style="color: #f8fafc; margin: 0;">Next Steps</h4>
                </div>
            """, unsafe_allow_html=True)
            next_steps = extract_section(analysis_result, "## 6. Next Steps")
            st.markdown(next_steps.replace("## 6. Next Steps", "").strip(), unsafe_allow_html=True)
       
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Main App -------------------
def main():
    # Configure page
    st.set_page_config(
        page_title="AI Medical Report Analyzer",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styles
    set_custom_style()
    
    # Initialize session state
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    
    # Main container
    with st.container():
        st.markdown("""
        <div class="main-container">
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="color: #f8fafc; margin-bottom: 0.5rem;">AI Medical Report Analyzer</h1>
                <p style="color: #94a3b8; font-size: 1.1rem;">
                    Advanced analysis of your medical reports using Gemini AI
                </p>
            </div>
        """, unsafe_allow_html=True)
       
        # Upload section
        file_type, uploaded_file = render_upload_section()
       
        # Show file preview if uploaded
        if uploaded_file and not st.session_state.analysis_done:
            try:
                if file_type == "Image":
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Medical Report", use_column_width=True)
                render_upload_success()
                st.session_state.file_uploaded = True
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
       
        # Analyze button
        if uploaded_file and not st.session_state.analysis_done:
            if st.button("Analyze Report with AI", type="primary"):
                with st.spinner("Analyzing your report..."):
                    if lottie_scan:
                        st_lottie(lottie_scan, height=150, key="scan")
                   
                    try:
                        # Extract text based on file type
                        if file_type == "PDF":
                            extracted_text = extract_text_from_pdf(uploaded_file)
                        else:
                            extracted_text = extract_text_from_image(image)
                       
                        if not extracted_text.strip():
                            st.error("No readable text found in the document.")
                            return
                           
                        # Analyze with AI
                        st.session_state.analysis_result = analyze_medical_report(extracted_text)
                        st.session_state.analysis_done = True
                       
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.session_state.analysis_done = False
       
        # Show results if analysis is complete
        if st.session_state.analysis_done and st.session_state.analysis_result:
            st.markdown("""
            <div class="card" style="margin-top: 2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="color: #f8fafc; margin-bottom: 0;">Analysis Results</h2>
                    <div style="display: flex; gap: 1rem;">
                        <button onclick="window.print()" style="background: #334155; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; cursor: pointer;">Print</button>
                    </div>
                </div>
                <p style="color: #94a3b8; margin-top: 0.5rem;">
                    Generated on {date}
                </p>
            """.format(date=datetime.now().strftime("%B %d, %Y at %H:%M")), unsafe_allow_html=True)
            
            render_analysis_tabs(st.session_state.analysis_result)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close results card
            
            # Action buttons
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    "📥 Download Full Report",
                    st.session_state.analysis_result,
                    f"medical_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    help="Download the complete analysis report"
                )
            with col2:
                st.markdown("""
                <div class="highlight-box">
                    <p style="margin: 0; color: #94a3b8;">
                        ⚠️ <strong>Disclaimer:</strong> This AI analysis is for informational purposes only
                        and should not replace professional medical advice. Always consult with a healthcare
                        provider for medical decisions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Analyze Another Report", type="primary"):
                st.session_state.analysis_done = False
                st.session_state.analysis_result = ""
                st.session_state.file_uploaded = False
                st.rerun()  # Fixed: Changed from st.experimental_rerun() to st.rerun()
       
        st.markdown("</div>", unsafe_allow_html=True)  # Close main-container

if __name__ == "__main__":
    main()