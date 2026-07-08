"""
AI Disease Prediction & Health Recommendation System
Streamlit Web Application (Light Medical Theme)

This web application loads the serialized ML model, scaler, and label encoder
to predict diseases based on selected symptoms and provide educational advice.

Features:
- Beautiful light medical CSS styling with rounded cards and progress bars
- Multi-category symptom checkboxes
- Top-3 disease predictions and confidence levels via Plotly
- Comprehensive disease recommendation catalog
- Feature contribution / local explainability chart (probability drop method)
- PDF summary report generator (fpdf2) - FIXED to return bytes
- Session prediction history tracker

Author: Antigravity AI
Date: July 2026
"""

import os
import json
import time
import pickle
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
DISEASE_CATALOG = {
    "Common Cold": {
        "description": "A mild, self-limiting viral infection of the upper respiratory tract.",
        "symptoms": ["cough", "runny nose", "sneezing", "sore throat", "fatigue", "headache", "chills"],
        "causes": "Rhinoviruses, coronaviruses, adenoviruses.",
        "risk_level": "Low",
        "specialist": "General Physician",
        "home_care": "Rest, stay hydrated, use warm saline gargles, sleep with head elevated, and use saline nasal sprays.",
        "foods_to_eat": "Warm broths (chicken soup), hot tea with honey, citrus fruits (oranges, lemons), garlic, ginger.",
        "foods_to_avoid": "Ice-cold drinks, excessive dairy (which may thicken mucus), refined sugar, alcohol.",
        "lifestyle": "Prioritize 8+ hours of sleep, wash hands frequently, stay warm, and run a room humidifier.",
        "prevention": "Wash hands with soap, avoid sharing utensils, avoid close contact with infected individuals.",
        "doctor_visit": "Symptoms lasting more than 10 days, fever higher than 101.3°F (38.5°C), or severe sinus pain.",
        "emergency": "Difficulty breathing, chest tightness, blue lips, severe wheezing, or confusion.",
        "recovery": "Typically 7 to 10 days. Avoid heavy exercise; rest is your body's primary recovery tool."
    },
    "Influenza (Flu)": {
        "description": "An infectious respiratory disease causing sudden fever, muscle aches, and fatigue.",
        "symptoms": ["fever", "cough", "fatigue", "headache", "muscle ache", "chills", "sore throat"],
        "causes": "Influenza viruses (Type A and B).",
        "risk_level": "Low to Medium (higher for elderly, young children, or immunocompromised)",
        "specialist": "General Physician",
        "home_care": "Strict bed rest, use paracetamol or ibuprofen for fever/aches, drink plenty of water and electrolytes.",
        "foods_to_eat": "Clear broths, herbal teas, coconut water, bananas, applesauce, foods containing zinc.",
        "foods_to_avoid": "Alcohol, caffeinated beverages, greasy/fried foods, tough meats, processed snacks.",
        "lifestyle": "Stay isolated to prevent spreading, rest in a dark quiet room, monitor temperature.",
        "prevention": "Receive the annual flu vaccine, clean surfaces frequently, avoid crowds during flu season.",
        "doctor_visit": "Fever that does not respond to medication, ear pain, sinus congestion, or symptoms returning after recovery.",
        "emergency": "Shortness of breath, persistent chest pain, severe muscle pain, seizures, or extreme weakness.",
        "recovery": "Usually 1 to 2 weeks. Gradually resume daily tasks; do not rush back to physical activities."
    },
    "COVID-19": {
        "description": "A highly contagious respiratory illness caused by the SARS-CoV-2 coronavirus.",
        "symptoms": ["fever", "cough", "fatigue", "loss of taste/smell", "breathlessness", "muscle ache", "chills", "sore throat", "headache"],
        "causes": "SARS-CoV-2 viral infection.",
        "risk_level": "Medium (High for individuals with underlying medical conditions or advanced age)",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "home_care": "Isolate in a ventilated room, monitor oxygen saturation (SpO2) with a pulse oximeter, rest, and hydrate.",
        "foods_to_eat": "Protein-rich foods, vitamin C fruits, vitamin D-rich foods (eggs, fortified milk), turmeric, ginger, warm broths.",
        "foods_to_avoid": "Highly processed foods, excess salt, carbonated drinks, trans fats.",
        "lifestyle": "Monitor vital signs daily, practice prone positioning if breathing is slightly uncomfortable, perform gentle breathing exercises.",
        "prevention": "Keep updated with vaccinations, wear masks in crowds, sanitize hands, maintain physical distance.",
        "doctor_visit": "SpO2 level drops below 94%, persistent fever above 102°F (38.9°C) for more than 3 days, or worsening cough.",
        "emergency": "Difficulty breathing, persistent pain or pressure in the chest, new confusion, or inability to wake or stay awake.",
        "recovery": "Varies from 2 weeks to months. Rest extensively, follow up on lung function if severe, monitor for long-COVID symptoms."
    },
    "Asthma": {
        "description": "A chronic inflammatory condition of the airways causing temporary spasms and obstruction.",
        "symptoms": ["breathlessness", "wheezing", "cough", "chest pain"],
        "causes": "Bronchial hyper-responsiveness triggered by allergens, exercise, cold air, or pollutants.",
        "risk_level": "Medium",
        "specialist": "Pulmonologist / Allergist",
        "home_care": "Keep rescue inhaler close at hand, sit upright during breathing distress, avoid cold drafts, track peak flow.",
        "foods_to_eat": "Magnesium-rich foods (spinach, pumpkin seeds), wild salmon, carrots, apples, tomatoes.",
        "foods_to_avoid": "Sulfite-containing foods (dried fruits, wine, processed potatoes), foods you are allergic to.",
        "lifestyle": "Identify and eliminate indoor triggers (dust, mold, pet dander), avoid smoking, exercise indoors when pollen is high.",
        "prevention": "Use controller inhalers consistently, receive flu vaccines, monitor air quality index before going outside.",
        "doctor_visit": "Waking up at night due to wheezing, needing rescue inhaler more than twice a week, or declining peak flow scores.",
        "emergency": "Extreme breathlessness, rescue inhaler provides no relief for 15 minutes, blue color in lips/fingertips, or chest retracting.",
        "recovery": "Manageable chronic condition. Recovery from an acute flare-up takes 1-3 days once triggers are removed."
    },
    "Diabetes": {
        "description": "A metabolic disorder resulting in chronic hyperglycemia (elevated blood glucose levels).",
        "symptoms": ["increased thirst", "frequent urination", "fatigue", "headache", "nausea"],
        "causes": "Insulin resistance (Type 2) or autoimmune destruction of pancreatic beta cells (Type 1).",
        "risk_level": "Medium (High if blood sugar remains uncontrolled long-term)",
        "specialist": "Endocrinologist",
        "home_care": "Regular blood glucose monitoring, take prescribed oral meds or insulin, inspect feet daily for cuts/sores.",
        "foods_to_eat": "Leafy green vegetables, non-starchy vegetables, legumes, whole grains (oats, quinoa), healthy fats (almonds, olive oil).",
        "foods_to_avoid": "Sugary beverages, refined grains (white bread, white rice), trans fats, packaged sweet snacks.",
        "lifestyle": "Engage in 150 minutes of moderate physical activity weekly, manage stress, maintain consistent meal times.",
        "prevention": "Maintain a healthy body weight, eat a balanced fiber-rich diet, stay active, and get regular checkups.",
        "doctor_visit": "Persistent high blood sugar (>180 mg/dL), signs of skin or urinary tract infections, or numbness in hands/feet.",
        "emergency": "Confusion, rapid deep breathing, fruity breath odor, persistent vomiting, or loss of consciousness (DKA/HHS).",
        "recovery": "Lifelong manageable condition. Consistent medication compliance and lifestyle habits ensure normal life expectancy."
    },
    "Hypertension": {
        "description": "A cardiovascular condition where blood pressure against arterial walls is consistently elevated.",
        "symptoms": ["headache", "fatigue", "chest pain", "breathlessness"],
        "causes": "High sodium intake, physical inactivity, obesity, genetics, chronic stress, tobacco use.",
        "risk_level": "Medium (High risk for stroke or heart attack if untreated)",
        "specialist": "Cardiologist",
        "home_care": "Monitor blood pressure at home, reduce daily salt consumption, avoid stimulants like nicotine and excess caffeine.",
        "foods_to_eat": "Bananas, sweet potatoes, spinach, berries, oatmeal, low-fat dairy (DASH diet framework).",
        "foods_to_avoid": "Processed meats (bacon, cold cuts), canned soups with high sodium, table salt, pickles, soy sauce.",
        "lifestyle": "Incorporate 30 minutes of aerobic exercise daily, practice mindfulness/meditation, limit alcohol consumption.",
        "prevention": "Eat low-sodium foods, exercise, manage weight, avoid smoking, limit stress.",
        "doctor_visit": "Blood pressure measurements consistently exceeding 140/90 mmHg, or frequent morning headaches.",
        "emergency": "Chest pain, severe headache with blurred vision, numbness/weakness, difficulty speaking, or sudden shortness of breath.",
        "recovery": "Requires long-term management. Regular monitoring and medication adherence keep blood pressure in check."
    },
    "Food Poisoning": {
        "description": "Acute gastrointestinal irritation resulting from consumption of contaminated or toxic food.",
        "symptoms": ["vomiting", "nausea", "diarrhea", "stomach pain", "fever", "fatigue", "chills"],
        "causes": "Ingestion of bacteria (Salmonella, E. coli), viruses (Norovirus), or toxins.",
        "risk_level": "Low",
        "specialist": "Gastroenterologist",
        "home_care": "Rest, allow the stomach to settle, sip Oral Rehydration Solutions (ORS) or coconut water, avoid solid food for several hours.",
        "foods_to_eat": "The BRAT diet (bananas, rice, applesauce, toast), crackers, broths, plain boiled potatoes.",
        "foods_to_avoid": "Dairy, caffeine, alcohol, spicy foods, greasy/fried foods, high-fiber raw vegetables.",
        "lifestyle": "Stay close to bathroom facilities, disinfect shared household items, wash hands thoroughly.",
        "prevention": "Cook meats to safe temperatures, wash hands and surfaces, refrigerate food within 2 hours of cooking.",
        "doctor_visit": "Diarrhea or vomiting lasting more than 3 days, high fever, or inability to keep any fluids down for 24 hours.",
        "emergency": "Signs of severe dehydration (extreme thirst, dry mouth, little/no urination, dizziness), or blood in stool/vomit.",
        "recovery": "Typically 1 to 5 days. Reintroduce regular foods slowly as tolerated."
    },
    "Migraine": {
        "description": "A neurological syndrome characterized by intense, throbbing, one-sided headaches.",
        "symptoms": ["headache", "nausea", "vomiting", "fatigue"],
        "causes": "Abnormal brain activity affecting nerve signals and blood vessels, triggered by stress, hormones, or foods.",
        "risk_level": "Low",
        "specialist": "Neurologist",
        "home_care": "Rest in a dark, quiet, cold room, place a cold compress on forehead/neck, take prescribed medication early.",
        "foods_to_eat": "Ginger tea, magnesium-rich foods (almonds, cashews), spinach, hydration with electrolyte water.",
        "foods_to_avoid": "Aged cheese, chocolate, red wine, cured meats with nitrates, MSG, artificial sweeteners.",
        "lifestyle": "Maintain a regular sleep schedule, eat meals at consistent times, avoid bright lights and loud noises.",
        "prevention": "Identify triggers with a headache diary, manage stress, avoid skipping meals, stay hydrated.",
        "doctor_visit": "Headaches increasing in frequency, or not responding to over-the-counter pain relievers.",
        "emergency": "Sudden 'thunderclap' headache (severe pain in seconds), headache with fever, stiff neck, confusion, or double vision.",
        "recovery": "An episode can last from 4 to 72 hours. Post-headache hangover (fatigue) can last 24 hours."
    },
    "Urinary Tract Infection (UTI)": {
        "description": "An infection of the urinary system, typically the bladder, causing pain and irritation.",
        "symptoms": ["pain during urination", "frequent urination", "stomach pain", "fever", "nausea"],
        "causes": "Bacterial invasion (commonly E. coli) of the urethra and bladder.",
        "risk_level": "Low (Medium if infection spreads upwards to the kidneys)",
        "specialist": "Urologist / Gynecologist",
        "home_care": "Drink plenty of water to flush bacteria, use a warm heating pad for pelvic discomfort, avoid holding urine.",
        "foods_to_eat": "Unsweetened pure cranberry juice, blueberries, probiotic yogurt, vitamin C-rich fruits.",
        "foods_to_avoid": "Caffeine, alcohol, spicy foods, carbonated sodas, artificial sweeteners (which irritate bladder).",
        "lifestyle": "Wipe front to back, empty bladder immediately after sexual intercourse, wear breathable cotton underwear.",
        "prevention": "Stay hydrated, urinate when needed, avoid scented feminine products in the genital area.",
        "doctor_visit": "Symptoms lasting more than 24 hours, cloudy or foul-smelling urine, or pelvic pressure (requires antibiotics).",
        "emergency": "High fever, chills, nausea, vomiting, or pain in the side/lower back (signs of kidney infection).",
        "recovery": "Typically 3 to 7 days with antibiotic treatment. Finish the entire course of medication."
    },
    "Allergies": {
        "description": "Immune system hypersensitivity reaction to environmental substances.",
        "symptoms": ["sneezing", "runny nose", "itching", "skin rash", "cough"],
        "causes": "Pollen, dust mites, pet dander, mold, insect stings, or specific foods.",
        "risk_level": "Low",
        "specialist": "Allergist / Immunologist",
        "home_care": "Avoid allergen exposure, wash nasal passages with saline, take over-the-counter antihistamines.",
        "foods_to_eat": "Foods high in quercetin (onions, apples), local honey, citrus fruits, anti-inflammatory ginger.",
        "foods_to_avoid": "Foods containing known allergens, highly processed foods, dairy (which can worsen congestion).",
        "lifestyle": "Keep windows closed during high pollen counts, wash bedsheets weekly in hot water, vacuum with a HEPA filter.",
        "prevention": "Monitor pollen counts, wash face/hair after spending time outdoors, use allergen-proof mattress covers.",
        "doctor_visit": "Symptoms that do not improve with antihistamines, or development of chronic sinus pressure.",
        "emergency": "Difficulty breathing, throat swelling, difficulty swallowing, dizziness, or hives all over the body (Anaphylaxis).",
        "recovery": "Varies. Seasonal allergies last as long as the pollen season. Relief can be felt within hours of taking antihistamines."
    },
    "Dengue": {
        "description": "A viral infection transmitted to humans through the bite of infected Aedes mosquitoes.",
        "symptoms": ["fever", "joint pain", "headache", "muscle ache", "skin rash", "fatigue", "chills", "vomiting", "nausea"],
        "causes": "Dengue virus (types 1-4) transmission.",
        "risk_level": "High (Requires close monitoring of platelet counts)",
        "specialist": "Infectious Disease Specialist",
        "home_care": "Absolute bed rest, drink fluids containing electrolytes (ORS, coconut water, barley water) to prevent dehydration.",
        "foods_to_eat": "Papaya leaf extract (helps increase platelet count), fresh fruit juices, steamed porridge, broths.",
        "foods_to_avoid": "Aspirin, Ibuprofen, Naproxen (they increase bleeding risk; use only paracetamol), dark-colored foods.",
        "lifestyle": "Sleep under a mosquito net, avoid moving around, use mosquito repellents indoors to prevent further transmission.",
        "prevention": "Eliminate standing water near house, wear long sleeves, use mosquito mesh on windows, apply insect repellent.",
        "doctor_visit": "Any suspected dengue cases require medical consultation and a blood test (CBC) to monitor platelets.",
        "emergency": "Severe abdominal pain, persistent vomiting, bleeding from gums/nose, blood in stool, or difficulty breathing (severe dengue).",
        "recovery": "Acute phase lasts 3-7 days; full recovery takes 2-3 weeks. Weakness and joint pain may linger for weeks."
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs (alveoli) in one or both lungs.",
        "symptoms": ["cough", "fever", "breathlessness", "chest pain", "chills", "fatigue", "muscle ache"],
        "causes": "Bacterial (Streptococcus pneumoniae), viral, or fungal infection.",
        "risk_level": "High",
        "specialist": "Pulmonologist",
        "home_care": "Take full course of prescribed antibiotics or antivirals, get plenty of bed rest, use a cool-mist humidifier.",
        "foods_to_eat": "Protein-rich foods (poultry, beans), warm soups, leafy green vegetables, garlic, warm lemon water.",
        "foods_to_avoid": "Cold drinks, fried foods, dairy (if it triggers phlegm production), refined carbohydrate foods.",
        "lifestyle": "Keep head elevated with pillows while sleeping, perform deep breathing exercises, strictly avoid smoking.",
        "prevention": "Get pneumococcal vaccine, annual flu vaccine, maintain good hand hygiene, avoid smoking.",
        "doctor_visit": "Persistent cough with colored phlegm, fever higher than 102°F (38.9°C), or increasing shortness of breath.",
        "emergency": "Inability to catch breath, sharp chest pain when breathing, confusion in elderly, or blue fingernails/lips.",
        "recovery": "Takes 2 to 6 weeks. A follow-up chest X-ray is often recommended to verify clearing of infection."
    },
    "Rheumatoid Arthritis": {
        "description": "A chronic autoimmune inflammatory disorder affecting the lining of the joints.",
        "symptoms": ["joint pain", "fatigue", "muscle ache"],
        "causes": "Autoimmune response where the body's immune system attacks its own joint tissues.",
        "risk_level": "Medium",
        "specialist": "Rheumatologist",
        "home_care": "Gentle range-of-motion exercises, apply heating pads for joint stiffness, apply ice packs during acute swelling.",
        "foods_to_eat": "Omega-3 rich foods (salmon, walnuts, flaxseeds), olive oil, berries, broccoli, green tea.",
        "foods_to_avoid": "Red meat, refined sugar, processed carbohydrates, saturated fats, gluten (if sensitive).",
        "lifestyle": "Balance activity with rest, maintain a healthy body weight to reduce joint load, utilize joint-protection tools.",
        "prevention": "Not preventable, but early diagnosis and DMARD therapy prevent joint destruction; avoid tobacco.",
        "doctor_visit": "Persistent joint pain, warmth, or morning stiffness lasting more than 30 minutes.",
        "emergency": "Sudden inability to move a joint, severe neck pain with numbness, or signs of joint infection (fever and extreme pain).",
        "recovery": "Chronic lifelong condition. Managed with medication, lifestyle, and physical therapy to preserve joint mobility."
    },
    "GERD (Acid Reflux)": {
        "description": "A chronic digestive disease where stomach acid flows back into the esophagus.",
        "symptoms": ["heartburn", "stomach pain", "nausea", "chest pain", "cough"],
        "causes": "Weakness of the lower esophageal sphincter, obesity, poor diet, smoking, or hiatal hernia.",
        "risk_level": "Low",
        "specialist": "Gastroenterologist",
        "home_care": "Eat smaller meals, remain upright for at least 3 hours after eating, elevate the head of the bed by 6 inches.",
        "foods_to_eat": "Oatmeal, bananas, melons, green vegetables (broccoli, asparagus), lean poultry, egg whites.",
        "foods_to_avoid": "Spicy foods, citrus fruits, tomatoes, chocolate, caffeine, alcohol, peppermint, fried foods.",
        "lifestyle": "Maintain healthy weight, quit smoking, wear loose-fitting clothes, avoid eating close to bedtime.",
        "prevention": "Eat smaller portions, limit triggers, stay upright after eating, avoid late-night snacks.",
        "doctor_visit": "Heartburn symptoms occurring more than twice a week, or symptoms not relieved by antacids.",
        "emergency": "Difficulty swallowing (dysphagia), severe chest pain radiating to arm (ruling out cardiac issues), vomiting blood.",
        "recovery": "Managed through diet and lifestyle adjustments. Symptoms usually resolve within days of starting treatment."
    },
    "Eczema": {
        "description": "An inflammatory skin condition that causes dry, red, and intensely itchy patches.",
        "symptoms": ["itching", "skin rash"],
        "causes": "Genetics, immune dysfunction, and environmental triggers like dry air, soaps, or allergens.",
        "risk_level": "Low",
        "specialist": "Dermatologist",
        "home_care": "Moisturize skin twice daily with thick creams/ointments, take short lukewarm baths, use mild soap-free cleansers.",
        "foods_to_eat": "Anti-inflammatory foods (fatty fish, apples, blueberries), yogurt (probiotics), water.",
        "foods_to_avoid": "Common triggers if sensitive (eggs, soy, wheat, dairy), high-sugar foods.",
        "lifestyle": "Maintain cool temperatures indoors, wear soft cotton clothing, trim fingernails short to prevent scratching.",
        "prevention": "Keep skin hydrated, identify and avoid irritants (harsh detergents, synthetic fabrics), manage stress.",
        "doctor_visit": "Skin shows signs of infection (pus, yellow crusting, red streaks), or rash is painful and prevents sleep.",
        "emergency": "Widespread skin blistering, severe pain, high fever, or spreading red skin rash over the entire body.",
        "recovery": "Chronic flare-up pattern. Acute flares can be controlled within a few days using topical treatments."
    }
}

# ---------------------------------------------------------
# STYLING & CUSTOM CSS (Professional Light Palette)
# ---------------------------------------------------------
st.set_page_config(
    page_title=" Disease Prediction & Health Recommendation System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Palette CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Core Background & Typography - LIGHT MEDICAL WORKSPACE */
    .stApp {
        background-color: #f8fafc; /* slate-50 */
        color: #0f172a; /* slate-900 */
    }
    
    h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    color: #0f172a;
    font-weight: 600;
    }

    p, span, li, label {
    color: #334155;
    }
    
    /* Professional Corporate Header */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0d9488 100%); /* Slate-Blue-Teal premium gradient */
        padding: 2.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.15);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #1e3a8a;
    }
    .header-box h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.02em;
        font-size: 2.3rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
    }
    .header-box p {
        color: #f1f5f9 !important; /* Soft white/light gray slate-100 for superb professional contrast */
        font-size: 1.15rem;
        margin: 0;
        font-weight: 500;
        letter-spacing: 0.01em;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    }
    .header-box *{
    color:#ffffff !important;
    }

.header-box h1,
.header-box h2,
.header-box h3,
.header-box p,
.header-box span,
.header-box div{
    color:#ffffff !important;
}
    
    /* Sidebar Layout (Clean, subtle border) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important; 
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #475569 !important; /* slate-600 */
    }
    
    /* Checkbox list refinement (cards layout with hover interaction) */
    div[data-testid="stCheckbox"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 8px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stCheckbox"]:hover {
        border-color: #0d9488;
        background-color: #f0fdfa; /* soft teal hint */
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.05);
    }
    
    /* Premium Styled Action Buttons */
    div.stButton > button {
        background: #0d9488 !important;
        color: #ffffff !important;
        border: 1px solid #0d9488 !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.1), 0 2px 4px -2px rgba(13, 148, 136, 0.1) !important;
    }
    div.stButton > button:hover {
        background: #0f766e !important;
        border-color: #0f766e !important;
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.2), 0 4px 6px -4px rgba(13, 148, 136, 0.2) !important;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Clinical Metrics Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        color: #64748b; /* slate-500 */
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        color: #0f172a; /* slate-900 */
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        font-family: 'Outfit', sans-serif;
    }
    .metric-desc {
        color: #64748b;
        font-size: 0.75rem;
    }
    
    /* Professional Results Presentation Panel */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488; /* strong teal left indicator */
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    .result-disease {
        color: #0f172a;
        font-size: 2.2rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
    }
    .result-confidence {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 1rem;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .badge-low {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-medium {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    /* Clinical Warnings Disclaimer Banner */
    .disclaimer-banner {
        background-color: #fff7ed; /* orange-50 */
        border-left: 4px solid #ea580c; /* orange-600 */
        color: #7c2d12; /* orange-900 */
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .disclaimer-banner strong {
        color: #431407;
    }
    
    /* Streamlit Expander styling overrides */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        color: #475569;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0d9488;
        background-color: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d9488 !important;
        color: #ffffff !important;
        border-color: #0d9488 !important;
    }
    
    /* Standard Checkbox labeling */
    .stCheckbox label p {
        color: #0f172a !important;
        font-weight: 500;
    }
    
    /* User Name input field styling context */
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 1px #0d9488 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_ml_models():
    """
    Loads best_model, scaler, label_encoder, and metadata from disk.
    Returns (model, scaler, label_encoder, metadata, error_message).
    """
    paths = {
        "model": "best_model.pkl",
        "scaler": "scaler.pkl",
        "encoder": "label_encoder.pkl",
        "metadata": "model_metadata.json"
    }
    
    # Check if files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            return None, None, None, None, f"Missing file: `{path}`. Please run the training script to train and serialize the model first."
            
    try:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)
        with open(paths["encoder"], "rb") as f:
            label_encoder = pickle.load(f)
        with open(paths["metadata"], "r") as f:
            metadata = json.load(f)
            
        return model, scaler, label_encoder, metadata, None
    except Exception as e:
        return None, None, None, None, f"Failed to load serialized pickle files due to corruption or version mismatch. Error: {str(e)}"


def get_local_explainability(model, scaler, label_encoder, input_vector: np.ndarray, predicted_class_idx: int) -> pd.DataFrame:
    """
    Calculates local feature contribution by using a probability-drop approach:
    For each active symptom (symptom value = 1), we flip it to 0 and calculate
    how much the predicted class probability drops.
    This provides model-agnostic, reliable local feature explanations.
    """
    feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else []
    if len(feature_names) == 0:
        return pd.DataFrame()
        
    # Get base probability for target class
    input_scaled = scaler.transform([input_vector])
    base_probs = model.predict_proba(input_scaled)[0]
    base_prob = base_probs[predicted_class_idx]
    
    contributions = []
    
    # Identify indices of active symptoms (value = 1)
    active_indices = np.where(input_vector == 1)[0]
    
    for idx in active_indices:
        # Clone input vector and flip the bit
        modified_vector = input_vector.copy()
        modified_vector[idx] = 0
        
        # Scale and predict
        modified_scaled = scaler.transform([modified_vector])
        new_probs = model.predict_proba(modified_scaled)[0]
        new_prob = new_probs[predicted_class_idx]
        
        # Contribution is the drops in probability (base_prob - new_prob)
        # Higher positive value means the symptom was crucial for the prediction
        drop = base_prob - new_prob
        contributions.append({
            "Symptom": feature_names[idx].replace("_", " ").title(),
            "Contribution": max(0.0, float(drop)) # Only keep positive contributions
        })
        
    df = pd.DataFrame(contributions)
    if not df.empty:
        # Normalize contributions to percentages
        total = df["Contribution"].sum()
        if total > 0:
            df["Percentage"] = (df["Contribution"] / total) * 100
        else:
            df["Percentage"] = 100.0 / len(df)
        df = df.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
    return df


class ClinicalReportPDF(FPDF):
    """
    Custom FPDF structure for generating clinical summaries.
    """
    def header(self):
        # Draw teal top band
        self.set_fill_color(13, 148, 136) # Teal
        self.rect(0, 0, 210, 40, "F")
        
        # Title text
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "CLINICAL SUMMARY & PRELIMINARY SCREENING", ln=True, align="C")
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 5, "AI-Powered Educational Support Tool", ln=True, align="C")
        self.ln(15)
        
    def footer(self):
        # Bottom disclaimer
        self.set_y(-30)
        self.set_fill_color(254, 226, 226) # Light Red
        self.rect(10, 267, 190, 15, "F")
        self.set_text_color(127, 29, 29) # Dark Red
        self.set_font("Helvetica", "B", 7)
        self.set_x(12)
        self.cell(186, 4, "EDUCATIONAL DISCLAIMER: This document is a preliminary screening report generated automatically", ln=True, align="C")
        self.cell(186, 4, "by an AI system. It is NOT a medical diagnosis and does NOT substitute for professional clinical treatment.", ln=True, align="C")
        
        # Page numbers
        self.set_y(-12)
        self.set_text_color(100, 116, 139)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")


def build_clinical_pdf(disease_name: str, confidence: float, details: Dict[str, Any], symptoms_checked: List[str], user_name: str = "Anonymous") -> bytes:
    """
    Generates a professional clinical summary PDF.
    Returns: Bytes object containing PDF data (converts bytearray output of fpdf2).
    """
    pdf = ClinicalReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=35)
    
    # Report Details Panel
    pdf.set_text_color(30, 41, 59) # Slate 800
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Report Details", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(95, 6, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=False)
    pdf.cell(95, 6, f"Patient Name: {user_name if user_name else 'Anonymous'}", ln=True)
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Symptoms Reported Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Symptoms Reported", ln=True)
    pdf.set_font("Helvetica", "", 10)
    symptoms_text = ", ".join([s.replace("_", " ").title() for s in symptoms_checked])
    pdf.multi_cell(0, 6, symptoms_text)
    pdf.ln(5)
    
    # Prediction Results Box
    pdf.set_fill_color(241, 245, 249) # Light Gray
    pdf.rect(10, pdf.get_y(), 190, 42, "F")
    pdf.set_y(pdf.get_y() + 3)
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "SCREENING PREDICTION", ln=True)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(13, 148, 136) # Teal
    pdf.cell(0, 8, f"{disease_name}", ln=True)
    
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, f"Confidence: {confidence:.1f}%", ln=True)
    pdf.set_x(15)
    pdf.cell(0, 5, f"Risk Category: {details['risk_level']}", ln=True)
    pdf.set_x(15)
    pdf.cell(0, 5, f"Suggested Specialist: {details['specialist']}", ln=True)
    
    pdf.set_text_color(30, 41, 59)
    pdf.set_y(pdf.get_y() + 8)
    
    # Description
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Condition Description", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, details["description"])
    pdf.ln(4)
    
    # Causes
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Potential Causes", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, details["causes"])
    pdf.ln(4)
    
    # Home Care Advice
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Educational Home Care & Advice", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, details["home_care"])
    pdf.ln(4)
    
    # Dietary Guidance
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Dietary Recommendations", ln=True)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(16, 101, 52) # Dark Green
    pdf.cell(0, 5, "Foods to Prioritize:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 4, details["foods_to_eat"])
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(153, 27, 27) # Dark Red
    pdf.cell(0, 5, "Foods to Avoid:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 4, details["foods_to_avoid"])
    pdf.ln(4)
    
    # Prevention & Lifestyle
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Prevention & Lifestyle Adjustments", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, f"Lifestyle: {details['lifestyle']}\nPrevention: {details['prevention']}")
    pdf.ln(4)
    
    # Warning signs
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(185, 28, 28) # Red
    pdf.cell(0, 6, "Emergency Warning Signs (Seek Urgent Care)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, details["emergency"])
    
    # Cast bytearray to bytes to prevent StreamlitAPIException
    return bytes(pdf.output())


# ---------------------------------------------------------
# APPLICATION LOGIC & UI BUILD
# ---------------------------------------------------------
# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Load models
model, scaler, label_encoder, metadata, load_error = load_ml_models()

# Banner / Header
st.markdown("""
<div class="header-box">
    <h1>AI Disease Prediction & Health Recommendation System</h1>
    <p>Educational Screening & Interactive Clinical Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Navigation Sidebar
with st.sidebar:
    # Attempt to load logo.png if it exists, otherwise show styled text
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("<h2 style='text-align: center; color: #0d9488; margin-top:0;'>🩺 AI Health</h2>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### Navigation")
    app_mode = st.radio(
        "Select Section:",
        [" Symptom Predictor", " Disease Info Center"]
    )
    
    st.markdown("---")
    st.markdown("### Model Summary")
    if load_error is None:
        st.markdown(f"**Best Model**: `{metadata['selected_model']}`")
        st.markdown(f"**Cross-Validation**: `{metadata['cv_score_mean'] * 100:.2f}%` Accuracy")
        st.markdown(f"**Features (Symptoms)**: `{metadata['number_of_features']}`")
        st.markdown(f"**Date Trained**: `{metadata['training_date'].split()[0]}`")
    else:
        st.warning("No model metadata found.")

# Display Error if models aren't loaded
if load_error:
    st.error("🚨 Configuration Error Detected")
    st.markdown(f"""
    **{load_error}**
    
    To resolve this issue:
    1. Open your terminal in the workspace directory.
    2. Install the dependencies: `python -m pip install -r requirements.txt`
    3. Run the model training pipeline: `python train_model.py`
    
    This will generate the synthetic dataset, train the models, and serialize the files.
    """)
    st.stop()

# Get ordered symptoms from scaler
symptoms_list = list(scaler.feature_names_in_)

# Categorized symptom layout mapping
symptom_categories = {
    "General / Systemic Symptoms": ["fever", "fatigue", "headache", "muscle_ache", "chills", "joint_pain"],
    "Respiratory / ENT Symptoms": ["cough", "breathlessness", "chest_pain", "runny_nose", "sneezing", "sore_throat", "loss_of_taste_smell", "wheezing"],
    "Gastrointestinal Symptoms": ["vomiting", "nausea", "diarrhea", "heartburn", "stomach_pain"],
    "Skin & Urinary / Other Symptoms": ["skin_rash", "itching", "increased_thirst", "frequent_urination", "pain_during_urination"]
}

# ---------------------------------------------------------
# SECTION 1: SYMPTOM PREDICTOR
# ---------------------------------------------------------
if "Symptom Predictor" in app_mode:
    st.subheader("📋 Interactive Preliminary Screening")
    st.write("Check all symptoms you are currently experiencing to run the predictive analysis.")
    
    # User name field
    user_name = st.text_input("User Name", placeholder="Enter your name to print in the report (optional)")
    
    # Organize checkboxes in layout
    checked_symptoms = []
    
    cols = st.columns(2)
    
    # Left Column (General & Respiratory)
    with cols[0]:
        with st.expander("🤒 General & Pain Symptoms", expanded=True):
            for s in symptom_categories["General / Systemic Symptoms"]:
                label = s.replace("_", " ").title()
                if st.checkbox(label, key=f"chk_{s}"):
                    checked_symptoms.append(s)
                    
        with st.expander("🫁 Respiratory & ENT Symptoms", expanded=True):
            for s in symptom_categories["Respiratory / ENT Symptoms"]:
                label = s.replace("_", " ").title()
                if st.checkbox(label, key=f"chk_{s}"):
                    checked_symptoms.append(s)
                    
    # Right Column (GI & Skin/Urinary)
    with cols[1]:
        with st.expander("🤢 Gastrointestinal Symptoms", expanded=True):
            for s in symptom_categories["Gastrointestinal Symptoms"]:
                label = s.replace("_", " ").title()
                if st.checkbox(label, key=f"chk_{s}"):
                    checked_symptoms.append(s)
                    
        with st.expander("🩹 Skin, Urinary & Endocrine Symptoms", expanded=True):
            for s in symptom_categories["Skin & Urinary / Other Symptoms"]:
                label = s.replace("_", " ").title()
                if st.checkbox(label, key=f"chk_{s}"):
                    checked_symptoms.append(s)

    st.markdown("---")
    
    # Predict Action
    if st.button("🔍 Predict Probable Disease", type="primary", use_container_width=True):
        if not checked_symptoms:
            st.warning("⚠️ Please select at least one symptom before running predictions.")
        else:
            with st.spinner("Processing symptom patterns and running clinical model..."):
                # Create input vector matching the exact features order
                input_vector = np.zeros(len(symptoms_list))
                for s in checked_symptoms:
                    if s in symptoms_list:
                        idx = symptoms_list.index(s)
                        input_vector[idx] = 1
                
                # Scale input
                scaled_input = scaler.transform([input_vector])
                
                # Predict probability
                probs = model.predict_proba(scaled_input)[0]
                classes = label_encoder.classes_
                
                # Sort indices of probabilities in descending order
                sorted_idx = np.argsort(probs)[::-1]
                
                top_class_idx = sorted_idx[0]
                predicted_disease = classes[top_class_idx]
                confidence_score = probs[top_class_idx] * 100
                
                # Save query to history
                query_info = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "user_name": user_name if user_name else "Anonymous",
                    "symptoms": checked_symptoms.copy(),
                    "prediction": predicted_disease,
                    "confidence": confidence_score
                }
                st.session_state.history.append(query_info)
                
                # Display Results
                
                # Core result box
                details = DISEASE_CATALOG.get(predicted_disease, {})
                risk = details.get("risk_level", "Unknown")
                badge_class = "badge-low"
                if "Medium" in risk:
                    badge_class = "badge-medium"
                elif "High" in risk:
                    badge_class = "badge-high"
                    
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-disease">🩺 Probable Cause: {predicted_disease}</div>
                    <div class="result-confidence">
                        Prediction Confidence: <strong>{confidence_score:.2f}%</strong> | 
                        Risk Level: <span class="badge {badge_class}">{risk}</span> | 
                        Suggested Specialist: <strong>{details.get('specialist')}</strong>
                    </div>
                    <p style="font-size: 1.1rem; line-height: 1.5; color: #334155;">{details.get('description')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 predictions chart & Explainability side-by-side
                res_cols = st.columns([1, 1])
                
                with res_cols[0]:
                    st.markdown("### Top 3 Probable Matches")
                    top3_names = [classes[idx] for idx in sorted_idx[:3]]
                    top3_probs = [probs[idx] * 100 for idx in sorted_idx[:3]]
                    
                    chart_df = pd.DataFrame({
                        "Disease": top3_names,
                        "Probability (%)": top3_probs
                    })
                    
                    fig = px.bar(
                        chart_df,
                        x="Probability (%)",
                        y="Disease",
                        orientation="h",
                        color="Probability (%)",
                        color_continuous_scale="teal",
                        text_auto=".1f",
                        range_x=[0, 105]
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#0f172a",
                        coloraxis_showscale=False,
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=10),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})
                    
                with res_cols[1]:
                    st.markdown("### 🔍 Symptom Contribution (Explainability)")
                    exp_df = get_local_explainability(model, scaler, label_encoder, input_vector, top_class_idx)
                    if not exp_df.empty:
                        fig_exp = px.bar(
                            exp_df.head(5),
                            x="Percentage",
                            y="Symptom",
                            orientation="h",
                            color="Percentage",
                            color_continuous_scale="sunset",
                            labels={"Percentage": "Relative Contribution (%)"},
                            text_auto=".1f"
                        )
                        fig_exp.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#0f172a",
                            coloraxis_showscale=False,
                            height=250,
                            margin=dict(l=0, r=0, t=10, b=10),
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig_exp, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})
                    else:
                        st.info("Explainability chart requires at least one active feature.")
                        
                # Recommendations Tabbed Interface
                st.markdown("---")
                st.markdown("### Educational Health Recommendations")
                
                rec_tabs = st.tabs(["🏡 Care & Lifestyle", "🥗 Diet Guide", "💡 Prevention & Emergency", "📄 Clinical Summary"])
                
                with rec_tabs[0]:
                    st.markdown(f"#### 🏡 Home Care Advice")
                    st.write(details.get("home_care"))
                    st.markdown("---")
                    st.markdown(f"#### 🏃 Lifestyle Recommendations")
                    st.write(details.get("lifestyle"))
                    st.markdown(f"**Recommended Specialist**: `{details.get('specialist')}`")
                    
                with rec_tabs[1]:
                    diet_cols = st.columns(2)
                    with diet_cols[0]:
                        st.markdown("#### 🟢 Foods to Eat (Prioritize)")
                        st.success(details.get("foods_to_eat"))
                    with diet_cols[1]:
                        st.markdown("#### 🔴 Foods to Avoid (Restrict)")
                        st.error(details.get("foods_to_avoid"))
                        
                with rec_tabs[2]:
                    st.markdown("#### 🛡️ Preventive Measures")
                    st.write(details.get("prevention"))
                    st.markdown("---")
                    st.markdown("#### ⚠️ Emergency Warning Signs")
                    st.warning(details.get("emergency"))
                    st.markdown(f"**Suggested Specialist Consultation**: `{details.get('specialist')}`")
                    
                with rec_tabs[3]:
                    st.markdown("#### Download Clinical Screening Report")
                    st.write("Generate a printable clinical screening PDF document containing the symptom profile, predictions, and home care recommendations.")
                    
                    pdf_bytes = build_clinical_pdf(predicted_disease, confidence_score, details, checked_symptoms, user_name)
                    
                    st.download_button(
                        label="📥 Download Clinical Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"clinical_summary_{predicted_disease.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    # CSV Export
                    csv_df = pd.DataFrame([{
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "User Name": user_name if user_name else "Anonymous",
                        "Symptom Profile": ", ".join(checked_symptoms),
                        "Primary Prediction": predicted_disease,
                        "Confidence Score": f"{confidence_score:.2f}%",
                        "Risk Level": risk,
                        "Specialist Suggested": details.get("specialist")
                    }])
                    
                    st.download_button(
                        label="📥 Export Prediction as CSV",
                        data=csv_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"prediction_{predicted_disease.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # Prediction History Section
    if st.session_state.history:
        st.markdown("---")
        with st.expander("📜 Session Screening History", expanded=False):
            hist_rows = []
            for query in st.session_state.history:
                hist_rows.append({
                    "Timestamp": query["timestamp"],
                    "User Name": query.get("user_name", "Anonymous"),
                    "Selected Symptoms": ", ".join([s.replace("_", " ").title() for s in query["symptoms"]]),
                    "Predicted Disease": query["prediction"],
                    "Confidence": f"{query['confidence']:.2f}%"
                })
            hist_df = pd.DataFrame(hist_rows)
            st.dataframe(hist_df, use_container_width=True)
            
            # Export all history
            st.download_button(
                label="📥 Download Complete History (CSV)",
                data=hist_df.to_csv(index=False).encode('utf-8'),
                file_name="screening_history.csv",
                mime="text/csv"
            )

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>Medical Disclaimer:</strong> This application is intended only for educational and preliminary screening purposes. 
        It is <strong>NOT</strong> a substitute for professional medical diagnosis, advice, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions you may have 
        regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of 
        something you have read or analyzed in this app.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 2: DISEASE INFO CENTER
# ---------------------------------------------------------
elif "Disease Info Center" in app_mode:
    st.subheader("📚 Disease Information & Encyclopedia")
    st.write("Browse details, home care instructions, dietary recommendations, and precautions for the 15 conditions covered in our screening model.")
    
    selected_disease = st.selectbox(
        "Choose Condition to Read Info:",
        sorted(list(DISEASE_CATALOG.keys()))
    )
    
    info = DISEASE_CATALOG[selected_disease]
    risk_level = info["risk_level"]
    badge_cls = "badge-low"
    if "Medium" in risk_level:
        badge_cls = "badge-medium"
    elif "High" in risk_level:
        badge_cls = "badge-high"
        
    st.markdown(f"""
    <div class="result-card" style="border-color: #cbd5e1; margin-bottom: 1.5rem;">
        <div class="result-disease" style="color: #0f172a;">📚 {selected_disease}</div>
        <div class="result-confidence" style="margin-bottom: 0.5rem; color:#475569;">
            Risk Category: <span class="badge {badge_cls}">{risk_level}</span> | 
            Suggested Specialist: <strong>{info['specialist']}</strong>
        </div>
        <p style="font-size: 1.1rem; line-height: 1.5; color: #334155;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    info_cols = st.columns(2)
    
    with info_cols[0]:
        st.markdown("### 🤒 Common Indicators")
        symptoms_str = ", ".join([s.replace("_", " ").title() for s in info["symptoms"]])
        st.info(symptoms_str)
        
        st.markdown("### 🦠 Possible Causes")
        st.write(info["causes"])
        
        st.markdown("### 🏡 Care & Lifestyle Tips")
        st.write(info["home_care"])
        st.write(f"**Lifestyle Changes**: {info['lifestyle']}")
        
    with info_cols[1]:
        st.markdown("### 🥗 Dietary Guidance")
        st.success(f"**Foods to Prioritize**:\n\n{info['foods_to_eat']}")
        st.error(f"**Foods to Avoid/Restrict**:\n\n{info['foods_to_avoid']}")
        
        st.markdown("### 🛡️ Prevention & Warnings")
        st.write(f"**Prevention**: {info['prevention']}")
        st.warning(f"**Emergency Warning Signs (Seek Urgent Care)**:\n\n{info['emergency']}")
        st.write(f"**When to Visit Doctor**: {info['doctor_visit']}")
