🩺 AI Medical Report Analyzer
This project is an AI-powered web application designed to analyze medical reports and provide meaningful insights using the Gemini AI API and Streamlit. Users can upload medical reports in PDF or image format, and the app automatically extracts text using OCR or PDF parsing, then generates:

🔍 Key findings from the report

🩺 Potential diagnoses with confidence scores

💊 Medication recommendations with dosages and effectiveness

🧘 Lifestyle guidance including diet and exercise tips

📊 Disease classification and next steps

⚙️ Features
🧠 Integrates Google's Gemini AI for medical text analysis

🖼️ Accepts both image and PDF formats

📈 Interactive visualizations of medicine effectiveness and diagnosis categories

🌐 Built with Streamlit and enhanced with custom CSS and animations

🔒 Uses environment variables for secure API key management

🚀 How It Works
The user uploads a medical report (PDF or image).

The text is extracted using pdfplumber or pytesseract.

Extracted text is analyzed using Gemini AI with a custom medical prompt.

Results are formatted and displayed across tabs for medications, diagnoses, lifestyle tips, and more.

Users can download a complete analysis report.

📦 Technologies Used
Python

Streamlit

Gemini API (via google.generativeai)

PDFPlumber

Pytesseract (OCR)

Matplotlib

Custom CSS and Lottie Animations
