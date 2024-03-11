import datasets as ds
import pandas as pd
import transformers 
import shap # Controllare di usare la versione 0.44.1. Versioni aggiornate non garantiscono il funzionamento corretto
import streamlit as st
from streamlit_shap import st_shap 
import torch
import matplotlib.pyplot as plt

dataset = ds.load_dataset("dair-ai/emotion", split = "train")

data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "nateraw/bert-base-uncased-emotion", use_fast=True
)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "nateraw/bert-base-uncased-emotion"
).cuda()

pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_all_scores=True,
)

# Test della pipeline di predizione con un testo di esempio
test_text = "I am feeling happy today!"
test_pred = pred(test_text)
print("Test Prediction:", test_pred)

# Creazione dell'interfaccia utente con Streamlit
st.title('Predizione delle Emozioni tramite testo ')

st.write("Questo è un'applicazione per prevedere le emozioni dal testo inserito.")

user_input = st.text_area("Inserisci il testo qui")

if st.button("Predizione"):
    
    # Controllo inserimento testo nel campo
    if user_input:  

        # Tokenizzazione del testo dell'utente
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        
        # Sposta i tensori su GPU se disponibile
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Predizione dell'emozione
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Estrazione delle probabilità e selezione dell'emozione con la probabilità più alta
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_label = torch.max(probs, dim=-1)
        predicted_emotion = model.config.id2label[top_label.item()]
        
        # Visualizzazione dell'emozione predetta
        st.write(f"L'emozione predetta è: {predicted_emotion} (Probabilità: {top_prob.item():.4f})")
#-------------------------------------------------------------------------
        
#-----------------------------------------------------------------------------
        
        # Calcolo dei valori SHAP per il testo inserito
        explainer = shap.Explainer(pred)
        shap_values = explainer([user_input])
        
        # Visualizzazione del grafico SHAP
        st.subheader("SHAP Text Plot")
        st_shap(shap.plots.text(shap_values[0]), height=300)


        # Visualizzazione del grafico SHAP per ogni classe
        st.subheader("SHAP Bar Plot")

        num_classes = model.config.num_labels  
    
        # Itera attraverso tutte le classi e genera un grafico SHAP per ciascuna
        for i in range(num_classes):
            # Crea una nuova figura per ogni classe
            plt.figure()  
            
            # Ottieni il nome della classe dall'ID
            class_name = model.config.id2label[i]

            # Imposta il titolo del grafico con il nome della classe
            plt.title(f"Classe: {class_name}")

            # Genera il grafico SHAP per la classe i
            st_shap(shap.plots.bar(shap_values[:,:,i][0]))
            
            # Mostra il grafico in Streamlit

            plt.clf()  # Pulizia della figura
        
        #Waterfall Plot 
        st.subheader("SHAP Waterfall Plot")
        # Itera attraverso tutte le classi e genera un grafico SHAP per ciascuna
        for i in range (num_classes):
             # Crea una nuova figura per ogni classe
            plt.figure()

            # Ottieni il nome della classe dall'ID
            class_name = model.config.id2label[i]

            # Imposta il titolo del grafico con il nome della classe
            plt.title(f"Classe: {class_name}")

            # Genera il grafico SHAP per la classe i
            st_shap(shap.plots.waterfall(shap_values[:,:,i][0]))

            # Mostra il grafico in Streamlit
            plt.clf()  # Pulizia della figura

        st.subheader("SHAP Force Plot") 

        tokens = tokenizer.tokenize(user_input)
        # Aggiunta token speciali per problemi di libreria SHAP
        feature_names = ["[CLS]"] + tokens + ["[SEP]"]

        for i in range (num_classes): 

            plt.figure()

            class_name = model.config.id2label[i]

            # Impostare il titolo del grafico con il nome della classe
            st.markdown(f"<h3 style='text-align: center; font-size: 16px;'>Classe: {class_name}</h3>", unsafe_allow_html=True)


            # Selezionare le SHAP values per una classe specifica (es. la prima classe)
            shap_values_for_class = shap_values.values[0][:, i]  

            # Selezionare il valore di base corrispondente alla stessa classe
            base_value_for_class = shap_values.base_values[0][i]  

            # Generare il force plot per la classe specifica
            st_shap(shap.force_plot(base_value_for_class, shap_values_for_class, feature_names))

            
            plt.clf()  # Pulizia della figura dopo la visualizzazione


        st.subheader("SHAP Decision Plot")

        for i in range (num_classes):

            plt.figure()

            class_name = model.config.id2label[i]

            # Impostare il titolo del grafico con il nome della classe
            st.markdown(f"<h3 style='text-align: center; font-size: 16px;'>Classe: {class_name}</h3>", unsafe_allow_html=True)

            st_shap(shap.decision_plot(shap_values.base_values[0][i],shap_values.values[0][:,i] , feature_names=feature_names))

        st.subheader("SHAP Summary Plot")

        st_shap(shap.summary_plot(shap_values.values[0], feature_names=feature_names, plot_type="violin"))

        plt.clf()  # Pulizia della figura
        
        
    else:
        st.write("Per favore, inserisci del testo per la predizione.")