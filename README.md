# Modello di riconoscimento di emozioni dal testo 

![Copertina del README](/copertina.jpg)


## Descrizione
Questo progetto utilizza diverse librerie e un modello pre-addestrato per l'emotion recognition. Sono state aggiunte le Shap Values per una miglior comprensione di come il modello fornisce la sua predizione. Tramite l'utilizzo di plot caratteristici di Shap, possiamo comprendere come il modello seleziona le varie features per la predizione. Il modello utilizza le varie libreria CUDA per l'addestramento del modello, quindi assicurarsi di avere i requisiti adatti a livello hardware/driver.

## Plot Supplementari 
Nella cartella `script` sono stati inseriti anche degli esempi su altri modelli per comprendere al meglio come le Shap Values vanno a migliorare la nostra comprensione della predizione effettuata dal modello. Di fatto gli script all'interno della cartella non sono inerenti al progetto stesso, ma sono stati inseriti solo a scopo informativo di come su vari tipi di dati, sono stati impostati i vari plot 

## Caratteristiche del progetto
- Emotion recognition a partire da un testo inserito dall'utente
- Visualizzazione della predizione e interpretazioni dei modelli tramite la visualizzazione di plot.
- Creazione di un'applicazione web per interagire con il modello

## Come iniziare
Per utilizzare questo progetto, seguire i seguenti passi:
1. Clonare il repository.
2. Installare le dipendenze utilizzando `pip install -r requirements.txt`.
3. Avviare l'applicazione Streamlit con `streamlit run emotion_recognition.py`, o in alternativa `python -m streamlit run emotion_recognition.py`.
  3_1. In caso di problemi con keras, digitare da terminale: `pip install tf-keras --user`
  3_2. in caso di errori con la libreria transformers, disintallare e reinstallare:`pip uninstall transformers' e 'pip install transformers`

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


