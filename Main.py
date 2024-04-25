import streamlit as st
import pymongo as po
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os
import base64
import time
import keras_ocr
import matplotlib.pyplot as plt
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains import SimpleSequentialChain
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import cv2
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from PIL import Image
import numpy as np
import os
import tensorflow
import pickle
import requests



ori_data = {}
pdf = ''  

def appointment():
    left_app,right_app=st.columns((2,2))
    with left_app:
        app_logo='https://lottie.host/a582d991-f4e1-46e1-9c69-5a494f2cc675/nKxHtllVca.json'
        app_image=load_lottieurl(app_logo)
        st_lottie(app_image,width=500,height=100)
    with right_app:
        st.title('Appointment Booking')
    st.subheader("Patient Details")
    patient_name = st.text_input('Patient Name:')
    patient_age = st.text_input('Patient Age')
    patient_city = st.text_input('Patient City')
    patient_phone = st.text_input('Patient Phone')
    patient_type = st.selectbox(
        'Select a value',
        ('New Patient', 'Old Patient')
    )
    uploaded_image = ''
    save_path = ''
    if(patient_type == 'Old Patient'):
        uploaded_image = st.file_uploader("Upload Your Prescription", type=["jpg", "jpeg", "png", 'heif'])
        if uploaded_image is not None:
            temp_dir = tempfile.mkdtemp()
            save_path = os.path.join(temp_dir, uploaded_image.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_image.read())

    st.subheader("Appointment Information")
    app_date = st.date_input("Select a date")
    app_date = ''.join(str(app_date).split('-'))
    app_time = st.selectbox(
        'Session',
        ('Morning', 'After Noon', 'Evening', 'Night')
    )

    if st.button("Book Appointment"):
        if not (patient_name and patient_age and patient_city and patient_phone and app_date):
            st.warning("Please fill in all required fields.")
        else:
            data = {
                'patient_name': patient_name,
                'patient_age': patient_age,
                'patient_city': patient_city,
                'patient_phone': patient_phone,
                'patient_type': patient_type,
                'image': save_path,
                'date': app_date,
                'time': app_time
            }
            ori_data = data

            database_name = "Patient"
            collection_name = "Patient_Details"
            clint = po.MongoClient("mongodb://localhost:27017")
            db = clint[database_name]
            collection = db[collection_name]
            collection.insert_one(ori_data)

            global pdf
            pdf = generate_pdf(ori_data)  
            st.success("Booked Successfully")
            pdf_file_path='patient_info.pdf'
            with open(pdf_file_path, "rb") as file:
                pdf_bytes = file.read()
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="appointment_details.pdf">Download Appointment Details</a>'
            st.markdown(href, unsafe_allow_html=True)

def generate_pdf(patient_info):
    pdf_file = "patient_info.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)

    c.setStrokeColor(colors.black)
    c.rect(20, 20, 550, 750)

    logo_path = "logoda.jpg"
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 400, 400, width=100, height=100)

    title = "Appointment Details"
    c.setFont("Helvetica-Bold", 25)
    c.drawCentredString(300, 700, title)

    c.setFont("Helvetica", 18)
    y_position = 600
    c.drawString(50, y_position, f"Patient Name: {patient_info['patient_name']}")
    y_position = 580
    c.drawString(50, y_position, f"Patient Age: {patient_info['patient_age']}")
    y_position = 560
    c.drawString(50, y_position, f"Patient City: {patient_info['patient_city']}")
    y_position = 540
    c.drawString(50, y_position, f"Patient Phone: {patient_info['patient_phone']}")
    y_position = 520
    c.drawString(50, y_position, f"Patient Type: {patient_info['patient_type']}")
    y_position = 500
    patient_info['date'] = patient_info['date'][:4]+'/'+patient_info['date'][4:6]+'/'+patient_info['date'][6:]
    c.drawString(50, y_position, f"Appointment Date: {patient_info['date']}")
    y_position = 480
    c.drawString(50, y_position, f"Appointment Session: {patient_info['time']}")
    y_position = 460
    y_position -= 20

    watermark_logo_path = "logoda.png"
    if os.path.exists(watermark_logo_path):
        c.setFillAlpha(0.5)
        c.drawImage(watermark_logo_path, 150, 250, width=300, height=300, mask='auto')
        c.setFillAlpha(1)

    c.save()
    return pdf_file

def prescription():
    st.title("Prescription Analysis")
    uploaded_image = st.file_uploader("Upload prescription image", type=["jpg", "jpeg", "png", 'heif'])
    if uploaded_image is not None:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, uploaded_image.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_image.read())

        img = save_path
        pipeline = keras_ocr.pipeline.Pipeline()
        value = pipeline.recognize([img])


        df = pd.DataFrame(value[0],columns=["Text",'Size'])
        text = ''
        for i in df['Text']:
            text+=i
            text+=' '
        med7 = spacy.load("en_core_med7_lg")
        doc = med7(text)
        medicine_names = [ent.text for ent in doc.ents if ent.label_ == "DRUG"]
        st.write(medicine_names)
def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
def show():
    lottie_url = "https://lottie.host/00b6f013-63d8-4663-8829-edacfc48286d/v0PwepvHxL.json"
    hello = load_lottieurl(lottie_url)
    with st_lottie_spinner(hello,width=500, height=200,key="main"):
        time.sleep(5)
def medical_bot():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = SECRET_API_TOKEN

    #ocr
    left_column, right_column = st.columns((2,5))
    with left_column:
        logo='https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image=load_lottieurl(logo)
        st_lottie(logo_image,width=300,height=100,key='logo')
    with right_column:
        st.title("MediChat")
    
    if prompt :=  st.file_uploader("Upload a Prescription Image"):
        file_bytes = prompt.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pipeline = keras_ocr.pipeline.Pipeline()
        value = pipeline.recognize([img])
        df = pd.DataFrame(value[0],columns=["Text",'Size'])
        text = ''
        for i in df['Text']:
            text+=' '
            text+=i
        med7 = spacy.load("en_core_med7_lg")
        options = {'ents': med7.pipe_labels['ner']}
        doc = med7(text)
        spacy.displacy.render(doc, style='ent', jupyter=True, options=options)
        a = [(ent.text, ent.label_) for ent in doc.ents]
        l=[]

        result = []
        for i in range(len(a)):
            if a[i][1] not in l:
                l.append(a[i][1])
                result.append(a[i][0])
        lis=['Paracetamol','ABCIXIMAB','ZOCLAR','GESTAKIND']
        
        def chat(lis):
            template1 = """What is {query} used for?"""
            template2 = """Who should not take {query}?"""
            template3 = """How should I take this {query}?"""
            template4 = """General answer to a particular medicine {query}?"""

            prompt1 = PromptTemplate(template=template1, input_variables=["query"])
            prompt2 = PromptTemplate(template=template2, input_variables=["query"])
            prompt3 = PromptTemplate(template=template3, input_variables=["query"])
            prompt4 = PromptTemplate(template=template4, input_variables=["query"])

            llm_model = HuggingFaceHub(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.5, "max_length": 2000}
            )

            llm_chain1 = LLMChain(prompt=prompt1, llm=llm_model)
            llm_chain2 = LLMChain(prompt=prompt2, llm=llm_model)
            llm_chain3 = LLMChain(prompt=prompt3, llm=llm_model)
            llm_chain4 = LLMChain(prompt=prompt4, llm=llm_model)


            model = SimpleSequentialChain(chains=[llm_chain1, llm_chain2, llm_chain3, llm_chain4])

            output = ""


            for query in lis:
                result = model.run(query)
                st.header(query)
                st.write(result)

        show()
        with st.chat_message("User"):
            st.markdown(" , ".join(lis))
        with st.chat_message("assistant"):
            chat(lis)
        st.warning('As an AI language model, I must clarify that I am not a healthcare professional, so I cannot provide medical advice. However, I can offer some general information.', icon="⚠️")

def brain():

    st.markdown("""
<style>
.first-title {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
        
    # Load The Model
    with open('tumor.pkl', 'rb') as f:
        model = pickle.load(f)

    # Label values
    dec = {0: 'No Tumor', 1: 'Positive Tumor'}

    # Define a function for tumor detection
    def detect_tumor(image):
        img1 = cv2.resize(image, (200, 200))
        img1 = img1.reshape(1, -1) / 255
        prediction = model.predict(img1)
        return prediction

    left,right=st.columns((2,2))
    with left:
        brain_logo='https://lottie.host/3adec33c-1f35-4bb5-9e3b-66884f6b715f/qZH9qyljFb.json'
        brain_image=load_lottieurl(brain_logo)
        st_lottie(brain_image,width=400,height=200,key='brain_logo')
    with right:

        st.title("Brain Tumor Detection")

    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        with st.spinner("Detecting tumor..."):
        # Call your detection function
            prediction = detect_tumor(image)


        st.header("Prediction:")
        st.markdown(f'<h1 class="first-title"> {dec[prediction[0]]} </h1>', unsafe_allow_html=True)

        st.image(image, caption='Uploaded Image', use_column_width=True)

def main():
    
    st.sidebar.title("MED TALK 🗣️")
    page = st.sidebar.selectbox(
        'Menu',
        ['Brain Tumor Detection', 'Appointment Booking','MediChat']
    )
    if page == 'Appointment Booking':
        appointment()
    if page == 'Prescription Analysis':
        prescription()
    if page == 'Brain Tumor Detection':
        brain()
    if page == 'MediChat':
        medical_bot()

if __name__ == "__main__":
    main()
