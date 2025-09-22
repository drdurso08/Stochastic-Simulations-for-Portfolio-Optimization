"IT'S IN ITALIAN. SOMEDAY IT WILL BE TRANSLATED BY A LLM, BUT THAT'S NOT THE DAY"   


# 📊 Dashboard di Ottimizzazione del Portafoglio con Streamlit

Questa è un'applicazione web interattiva costruita con Streamlit per l'analisi e l'ottimizzazione di portafogli finanziari. L'applicazione è divisa in due sezioni principali:

1.  **Comparative Backtesting**: Permette di testare l'allocazione con i vari metodi. Obbligatorio runnarla almeno una volta per dare i ticker allafase due.
2.  **Forward-Looking Optimization**: Usando gli ultimi N anni di dati calcola i pesi ottimali. Ha le stesse funzioni della precedente
---

## 🚀 Guida Rapida: Come Avviare l'App in 5 Passaggi

Segui questi semplici passaggi per far funzionare l'applicazione.

### Passaggio 1: Apri il Terminale
Cerca e apri l'applicazione "Terminale" o "Prompt dei comandi" sul tuo computer.

### Passaggio 2: Vai alla Cartella del Progetto
Copia e incolla questo comando nel terminale e premi Invio. Ti porterà nella cartella corretta.

```bash
cd "c:\Users\...\Portfolio_App_Streamlit"          
```
Ovviamente al posto dei puntini metti la directory in cui hai sta roba. Basterebbe anche solo fare tasto destro nella cartella, apri nel terminale e amen.

### Passaggio 3: Crea e Attiva un "Ambiente Virtuale"
Questi comandi creano una specie di "scatola" separata solo per questo progetto, per evitare conflitti con altri programmi.

**Prima copia e incolla questo:**
```bash
python -m venv venv
```
**E subito dopo, questo:**
```bash
.\venv\Scripts\activate
```
Se tutto è andato bene, vedrai la scritta `(venv)` all'inizio della riga nel terminale.

### Passaggio 4: Installa le Dipendenze
Ora, con l'ambiente attivo, copia e incolla questo comando. Installerà automaticamente tutte le librerie di cui l'app ha bisogno per funzionare.

```bash
pip install -r requirements.txt
```
Attendi che finisca l'installazione. Potrebbe volerci qualche minuto.

### Passaggio 5: Avvia l'Applicazione!
Hai quasi finito. Esegui questo ultimo comando per avviare l'app:

```bash
streamlit run app.py
```
L'applicazione si aprirà da sola in una nuova scheda del tuo browser (come Chrome, Edge, o Firefox).

---

## 📝 Guida all'Uso

### Schermata 1: Comparative Backtesting

Questa schermata ti permette di confrontare la performance storica di diverse strategie di portafoglio.

1.  **Configura il Backtest**:
    *   **Optimization Metric**: Scegli tra `Sharpe Ratio` e `Sortino Ratio` come metrica per l'ottimizzazione. Il Sortino è lo sharpe concentrato sulla volatilità negativa. All'atto pratico cambia poco in termini di allocazione dei pesi. 
    *   **Simulation Model**: Seleziona il modello (`Merton Jump-Diffusion` o `ARIMA-GARCH`) da usare per simulare i rendimenti futuri nel periodo di stima. Il primo modello è un modello semplice dove i prezzi oggi sono il prezzo ieri piu uno shock casuale stimato sui dati storici, la procedura di simulazione è estremamente veloce e lo consiglio per analisi preliminari. ARIMA-GARCH è un modello estremamente piu complesso in termini computazionali, dato un periodo di train troverà il modello econometrico migliore. Userà quel modello e un estrattore di errori per simulare i path futuri dei titoli. A breve in una patch arriverà la versione multivariata per simulare shock negativi multi-asset se e quando ho voglia. Questo secondo modello è estremamente più realistico e stabile, ovviamente ci vuole più tempo.
    *   **Select Tickers**: Scegli gli asset da includere nell'analisi. Puoi anche aggiungerne di personalizzati. La nomenclatura da rispettare è quella di TradingView, quindi ticker e Exchange(che trovi a destra quando cerchi su TV)
    *   **Analysis Parameters**: Imposta il numero di anni da usare per la stima del modello e per il backtest. 4 e 1 è abbastanza lungo, cambia se vuoi ma non aspettarti troppe differenze
    *   **Select Benchmark**: Scegli un benchmark di riferimento (es. S&P 500). Nelle prossime release ci sarà l'aggiunta manuale, forse. 
    *   **Black-Litterman**: Se vuoi, puoi includere la strategia Black-Litterman e definire le tue "views" (aspettative) sul mercato. Ulteriori informazioni sotto in sezione 2.

2.  **Esegui l'Analisi**: Clicca su **"Run Backtest Analysis"**. L'app scaricherà i dati, eseguirà le simulazioni, ottimizzerà i portafogli e mostrerà i risultati del backtest.  I portafogli generati sono i seguenti: **Inverse VOlatility** dove il peso è l'inverso della volatilità totale sul portafoglio. Più o meno è sempre stabile e non ha vincoli ai pesi perché non saranno mai 0 o 100 i pesi. **Weight Equilibrium** è costruito con un ottimizzazione inversa partento da **I.V.** e serve come input per BlackLittermann. **Black Littermann** includele aspettative che hai su qualcosa, modificando i pesi di Equilibrium.

3.  **Analizza i Risultati**:
    *   **Weights & Sector Allocation**: Visualizza i pesi di ogni asset e l'allocazione per settore.
    *   **Portfolio Performance**: Grafico della performance cumulata delle strategie rispetto al benchmark.
    *   **Metrics Table**: Tabella riassuntiva con le principali metriche di performance. 0.7 è lo sharpe medio dell'SP500. Il VaR ti dice al 95% quale è la perdita massima che farai nel mese. ENB più è vicino al numero totale di asset e meglio è diversificato il portafoglio. DR>2 vuol dire buona diversificazione del rischio

4.  **Procedi all'Ottimizzazione**: Se sei soddisfatto degli asset selezionati, clicca su **"Go to Forward-Looking Optimization"**.





### Schermata 2: Forward-Looking Optimization

Questa schermata ti permette di calcolare i pesi ottimali del portafoglio per il futuro, basandoti anche sulle tue aspettative.

1.  **Analisi Storica Preliminare**:
    *   Prima di inserire le tue "views", esegui un'analisi storica per avere un'idea dei rendimenti passati degli asset. Questo ti aiuterà a formulare aspettative più realistiche. I rendimenti sono annualizzati e VANNO formulati annualizzati se stai impostando le aspettative. Questa parte serve a caricare i ticker e a dare all'utente una scala di grandezza dei ritorni annui. Consiglio di usare 5 anni come periodo.

2.  **Configura l'Ottimizzazione**:
    *   **Optimizer Settings**: Scegli la metrica di ottimizzazione, il modello di simulazione e gli anni di dati storici da usare.
    *   **Black-Litterman Constraints**: Definisci i vincoli sui pesi del portafoglio. Ci sono tre modalità. Heuristic sarebbe un vincolo hard-coded basato sulla mia esperienza dove per garantire un minimo di diversificazione imposto come minimo (1/num_asset)/2 e come max (1/num_asset)*4. Short enabled usarlo SOLO se vuoi andare short su uno specifico stock o settore e non è detto che il modello gli assegni un peso short. Custom metti i pesi che cazzo di pare e amen, non farlo a meno che non hai un'idea precisa
    *   **Black-Litterman Views**: **Questa è la parte più importante**. Aggiungi le tue "views" (aspettative) sul rendimento futuro di specifici asset o settori. Per ogni view, definisci il valore atteso (Q) e il tuo livello di confidenza. Se non clicchi "add views" il modello è vuoto. Se ci clicchi avrai diverse opzioni.
    **Absolute** significa uno statement tipo "Questa cosa renderà il Q% in un anno". A questa informazione devi dare una confidenza da 0 a 100% che serve a comprimere o espandere la matrice di covarianza. 
    **Relative** ha bisogno di due input che verranno chiamati "Outperforms" e "Underperforms". Il numero che metti in Q% è quanto ti aspetti che il primo performi più del secondo. 
    **Input** ci puoi mettere Ticker vs Ticker o Sector vs Sector. Dipende da che tipo di views hai

3.  **Calcola i Pesi Ottimali**: Clicca su **"Calculate Optimal Weights"**. L'app eseguirà le simulazioni e il modello di Black-Litterman per calcolare i portafogli ottimali che bilanciano le aspettative del mercato con le tue views.

4.  **Analizza i Risultati**:
    *   **Optimal Weights**: Tabella con i pesi ottimali per ogni strategia.
    *   **Expected Performance**: Metriche di performance attese (Rendimento, Volatilità, Sharpe/Sortino Ratio) per ogni portafoglio.
    *   **Sector Allocations**: Grafici a torta che mostrano l'allocazione per settore di ogni strategia.




**EXTRA** Note metodologiche
Markowitz produce risultati instabili, quindi o lo ottimizzi dando dei limiti (quello che fa la modalità Heuristic) o il portafoglio fa schifo. Per evitare di usare dati passati, questi servono come elementi per effettuare delle simulazioni dei path dei titoli. Su quelle simulazioni saranno calcolate media e covarianza futura ed effettuata l'ottimizzazione. I risultati in questo caso sono più stabili e l'ottimizzazione da risultati più sparsi. ARIMA_GARCH è più completo ma lento.

I risultati nella fase 2 in expected performance potrebbero sembrare negativi rispetto al Black Littermann ma è normale. Se tu stai dando delle views ti stai distaccando dai modelli stocastici che hai usato per simulare i prezzi, quindi automaticamente verranno risultati diversi. Non prestarci troppa attenzione. 

