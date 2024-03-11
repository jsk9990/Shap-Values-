# Modello di riconoscimento di emozioni dal testo 

![Copertina del README](/copertina.jpg)


## Descrizione
Questo progetto utilizza diverse librerie e un modello pre-addestrato per l'emotion recognition. Sono state anche le Shap Values per una miglior comprensione di come il modello fornisce la sua predizione. Tramite l'utilizzo di plot caratteristici di Shap, possiamo comprendere come il modello seleziona le varie features per la predizione. Il modello utilizza le varie libreria CUDA per l'addestramento del modello, quindi assicurarsi di avere i requisiti adatti a livello hardware/driver

## Caratteristiche
- Emotion recognition a partire da un testo inserito
- Visualizzazione delle predizioni e interpretazioni dei modelli.
- Creazione di un'applicazione web per interagire con il modello

## Come iniziare
Per utilizzare questo progetto, seguire i seguenti passi:
1. Clonare il repository.
2. Installare le dipendenze utilizzando `pip install -r requirements.txt`.
3. Avviare l'applicazione Streamlit con `streamlit run emotion_recognition.py`.

## Dipendenze
Questo progetto richiede le seguenti librerie, elencate nel file `requirements.txt`:

```
datasets
pandas
transformers
shap==0.44.1
streamlit
streamlit_shap
torch
matplotlib
```

## Autori
- de Stasio Giuseppe 
- Langiotti Andrea 
- Sergiacomi Daniele


## Contribuire
Se si desidera contribuire a questo progetto, si prega di fare un fork del repository e inviare una pull request.


