document.addEventListener('DOMContentLoaded', () => {
    const ebookData = [
        {
            part: "Introduzione",
            chapters: [
                {
                    id: "ch0",
                    title: "Introduzione all'Ebook",
                    content: `
                        <p>Questo ebook è una guida chiara e accessibile all'Intelligenza Artificiale, pensata per chiunque voglia comprenderne l'impatto sul lavoro e sulla vita quotidiana, senza necessità di background tecnico. Impareremo a distinguere la realtà dal sensazionalismo, riconoscendo opportunità e limiti dell'AI.</p>
                    `
                }
            ]
        },
        {
            part: "Parte 1: Fondamenti dell'Intelligenza Artificiale",
            chapters: [
                {
                    id: "ch1",
                    title: "Capitolo 1: Benvenuti/e nel Mondo dell'AI",
                    content: `
                        <h3>Cos'è l'Intelligenza Artificiale (AI)? Definizione e Obiettivi.</h3>
                        <p>L'AI è una branca dell'informatica che mira a creare sistemi capaci di svolgere compiti che richiedono intelligenza umana, come ragionamento, apprendimento e problem solving. L'obiettivo non è replicare il cervello umano, ma creare sistemi utili e intelligenti.</p>
                        <h3>Breve Storia e Tappe Fondamentali.</h3>
                        <p>Dalle radici teoriche negli anni '40-'50 (Turing, Dartmouth Workshop) ai primi "inverni", fino al boom attuale guidato da Big Data, potenza computazionale (GPU) e algoritmi avanzati come i Transformers.</p>
                        <h3>Tipi di AI: AI Ristretta (Narrow), Generale (AGI), Superintelligenza (ASI).</h3>
                        <ul>
                            <li><strong>AI Ristretta (ANI):</strong> L'unica esistente oggi, specializzata in compiti specifici (es. filtri anti-spam, riconoscimento vocale).</li>
                            <li><strong>AI Generale (AGI):</strong> Ipotetica AI con intelligenza paragonabile a quella umana.</li>
                            <li><strong>Superintelligenza (ASI):</strong> Ipotetica AI che supera significativamente l'intelligenza umana.</li>
                        </ul>
                        <h3>Perché l'AI è Importante Oggi? Esempi Quotidiani.</h3>
                        <p>L'AI (ANI) è cruciale per automatizzare compiti, analizzare Big Data, personalizzare esperienze, ottimizzare sistemi e creare nuovi contenuti. È presente in smartphone, intrattenimento, shopping, trasporti, marketing (es. "vibe marketing" per contenuti emotivamente risonanti) e sanità.</p>
                    `
                },
                {
                    id: "ch2",
                    title: "Capitolo 2: Introduzione al Machine Learning (ML)",
                    content: `
                        <h3>Cos'è il Machine Learning? Imparare dai Dati.</h3>
                        <p>Il Machine Learning (ML) è un approccio all'AI che permette ai computer di imparare dai dati senza essere esplicitamente programmati. Invece di regole codificate, si forniscono dati e un algoritmo che apprende pattern per fare previsioni o decisioni.</p>
                        <p>Definizione di Tom Mitchell: <em>Un programma apprende dall'esperienza E rispetto a compiti T con misura di performance P, se la sua performance P su T migliora con E.</em></p>
                        <h3>Differenza tra AI, ML e Deep Learning (DL).</h3>
                        <p>Relazione gerarchica: AI (concetto generale) > ML (sottoinsieme dell'AI, apprendimento dai dati) > DL (sottoinsieme del ML, basato su reti neurali profonde). Il DL è dietro molti successi recenti dell'AI.</p>
                        <h3>Panoramica dei Tipi di Apprendimento: Supervisionato, Non Supervisionato, Per Rinforzo.</h3>
                        <ul>
                            <li><strong>Supervisionato:</strong> L'algoritmo impara da dati etichettati (input con risposta corretta). Obiettivo: mappare input a output. Problemi: Classificazione (etichette discrete) e Regressione (valori continui).</li>
                            <li><strong>Non Supervisionato:</strong> L'algoritmo impara da dati non etichettati. Obiettivo: scoprire strutture nascoste. Problemi: Clustering, Riduzione Dimensionalità, Anomaly Detection.</li>
                            <li><strong>Per Rinforzo (RL):</strong> Un agente impara interagendo con un ambiente, ricevendo ricompense o penalità per le sue azioni. Obiettivo: massimizzare la ricompensa cumulativa.</li>
                        </ul>
                        <h3>Il Ruolo dei Dati nell'ML.</h3>
                        <p>I dati sono il carburante del ML. La loro <strong>Quantità</strong> (più dati spesso = migliori prestazioni), <strong>Qualità</strong> (accurati, completi, senza errori) e <strong>Rappresentatività</strong> (devono riflettere la realtà in cui il modello opererà, per evitare bias) sono cruciali.</p>
                    `
                },
                {
                    id: "ch3",
                    title: "Capitolo 3: Concetti Chiave e Terminologia",
                    content: `
                        <h3>I Dati: Il Punto di Partenza</h3>
                        <ul>
                            <li><strong>Dataset:</strong> Raccolta organizzata di dati per addestrare/valutare modelli (es. tabella, collezione immagini).</li>
                            <li><strong>Istanza (Instance) / Esempio (Sample) / Record:</strong> Singolo elemento/punto dati nel dataset (es. una riga in una tabella).</li>
                            <li><strong>Feature / Attributo / Variabile (Indipendente):</strong> Caratteristica misurabile di un'istanza (es. una colonna in una tabella), usata come input dal modello.</li>
                            <li><strong>Etichetta (Label) / Target / Variabile Dipendente / Output:</strong> Valore "corretto" (output desiderato) che il modello deve predire (usato nell'apprendimento supervisionato).</li>
                        </ul>
                        <h3>L'Algoritmo e il Modello: La Ricetta e il Piatto Cotto</h3>
                        <ul>
                            <li><strong>Algoritmo (Algorithm):</strong> Procedura matematica/computazionale generale per imparare dai dati (la "ricetta"). Esempi: Albero Decisionale, K-Means.</li>
                            <li><strong>Modello (Model):</strong> Risultato specifico ottenuto applicando un algoritmo a un dataset (il "piatto cotto"). Contiene i pattern appresi ed è usato per le previsioni.</li>
                        </ul>
                        <h3>Dentro il Modello e l'Algoritmo: Parametri e Iperparametri</h3>
                        <ul>
                            <li><strong>Parametri (Parameters):</strong> Variabili interne del modello che l'algoritmo apprende dai dati durante l'addestramento (es. pesi in una rete neurale).</li>
                            <li><strong>Iperparametri (Hyperparameters):</strong> Impostazioni dell'algoritmo scelte <em>prima</em> dell'addestramento (es. numero di cluster K in K-Means, profondità di un albero). Il loro tuning è cruciale.</li>
                        </ul>
                        <h3>Il Processo di Apprendimento e Utilizzo</h3>
                        <ol>
                            <li><strong>Addestramento (Training) / Fitting:</strong> Il modello impara i parametri dal training set.</li>
                            <li><strong>Inferenza (Inference) / Previsione (Prediction):</strong> Il modello addestrato fa previsioni su nuovi dati.</li>
                            <li><strong>Valutazione (Evaluation):</strong> Si misurano le prestazioni del modello su un test set (dati mai visti) usando metriche specifiche.</li>
                        </ol>
                        <h3>Due Problemi Comuni: Underfitting e Overfitting</h3>
                        <ul>
                            <li><strong>Underfitting (Sottoadattamento):</strong> Il modello è troppo semplice, non impara abbastanza. Scarse prestazioni sia su training che su test set.</li>
                            <li><strong>Overfitting (Sovradattamento):</strong> Il modello impara troppo bene il training set (incluso il rumore), ma non generalizza a nuovi dati. Ottime prestazioni su training, scarse su test set. Trovare il giusto equilibrio è fondamentale.</li>
                        </ul>
                    `
                },
                {
                    id: "ch4",
                    title: "Capitolo 4: AI Responsabile: Etica e Impatto Sociale",
                    content: `
                        <p>L'AI, pur potente, porta con sé importanti questioni etiche e un impatto sociale profondo. Sviluppare e utilizzare l'AI in modo responsabile è fondamentale.</p>
                        <h3>Bias nei Dati e negli Algoritmi (AI Bias)</h3>
                        <p>Gli algoritmi AI non sono intrinsecamente oggettivi. Possono imparare, perpetuare e amplificare i bias presenti nei dati di addestramento o derivanti dalle scelte di progettazione.</p>
                        <ul>
                            <li><strong>Bias nei Dati (Data Bias):</strong> Causa più comune. Se i dati riflettono pregiudizi storici/sociali/culturali, il modello li apprenderà (es. bias di genere nelle assunzioni, bias di rappresentazione in sistemi di riconoscimento facciale).</li>
                            <li><strong>Bias Algoritmico (Algorithmic Bias):</strong> Può derivare da scelte di progettazione dell'algoritmo o dalla selezione delle feature.</li>
                        </ul>
                        <p><strong>Conseguenze del Bias:</strong> Decisioni ingiuste, discriminatorie e dannose in ambiti critici (assunzioni, credito, giustizia penale, sanità).</p>
                        <h3>Equità (Fairness) e Trasparenza</h3>
                        <ul>
                            <li><strong>Equità (Fairness):</strong> Concetto complesso e dipendente dal contesto. Non esiste una definizione unica. Possibili interpretazioni: parità demografica, uguaglianza di opportunità, uguaglianza di accuratezza tra gruppi.</li>
                            <li><strong>Trasparenza e Spiegabilità (Explainability - XAI):</strong> Molti modelli AI avanzati (specialmente Deep Learning) sono "scatole nere". La mancanza di trasparenza è problematica. XAI mira a rendere i modelli più interpretabili per identificare bias, verificare correttezza e costruire fiducia.</li>
                        </ul>
                        <h3>Privacy e Sicurezza dei Dati</h3>
                        <p>I modelli ML necessitano di grandi quantità di dati, spesso sensibili.</p>
                        <ul>
                            <li><strong>Rischi per la Privacy:</strong> Raccolta eccessiva, re-identificazione da dati anonimizzati, inferenza indesiderata di informazioni sensibili, sorveglianza.</li>
                            <li><strong>Rischi per la Sicurezza:</strong> Data breach, attacchi avversari (input manipolati per ingannare il modello).</li>
                        </ul>
                        <p>Leggi come il GDPR e tecniche come Privacy Differenziale sono importanti.</p>
                        <h3>L'Impatto dell'AI sul Lavoro e sulla Società</h3>
                        <ul>
                            <li><strong>Impatto sul Lavoro:</strong> Automazione e job displacement (sostituzione di lavori), creazione di nuovi lavori (Data Scientist, AI Ethicist), trasformazione dei lavori esistenti (AI come strumento di potenziamento), necessità di reskilling e upskilling.</li>
                            <li><strong>Impatto sulla Società:</strong> Influenza su processi decisionali (ammissioni, prestiti), interazioni umane (chatbot), disuguaglianza (digital divide), informazione e democrazia (deepfake, fake news).</li>
                        </ul>
                        <p>È necessario un approccio responsabile per guidare lo sviluppo dell'AI in modo equo, trasparente, sicuro, rispettoso della privacy e allineato ai valori umani.</p>
                    `
                }
            ]
        },
        {
            part: "Parte 2: Gli Strumenti Essenziali del Mestiere",
            chapters: [
                { id: "ch5", title: "Capitolo 5: L'Ambiente di Sviluppo per l'AI", content: `
                    <p>Dopo aver compreso i concetti base, è il momento di preparare gli strumenti per costruire modelli AI. Questo capitolo illustra gli ambienti di sviluppo più comuni.</p>
                    <h3>Perché Python? Il Linguaggio d'Elezione per l'AI</h3>
                    <p>Python è lo standard de facto per AI/ML grazie a:</p>
                    <ul>
                        <li><strong>Semplicità e Leggibilità:</strong> Sintassi intuitiva, facile da imparare.</li>
                        <li><strong>Vasto Ecosistema di Librerie:</strong> NumPy, Pandas, Scikit-learn, TensorFlow, Keras, PyTorch forniscono funzionalità pronte all'uso.</li>
                        <li><strong>Grande Comunità e Supporto:</strong> Abbondanza di tutorial, documentazione e soluzioni online.</li>
                        <li><strong>Flessibilità e Integrazione:</strong> Adatto sia a prototipazione rapida che a sistemi complessi.</li>
                    </ul>
                    <h3>Setup dell'Ambiente: Anaconda e Ambienti Virtuali</h3>
                    <p>Installare Python e librerie manualmente può essere complesso. <strong>Anaconda</strong> è una distribuzione open-source di Python (e R) per calcolo scientifico e data science, che semplifica il processo includendo Python, un gestore di pacchetti/ambienti (<code>conda</code>), e centinaia di librerie preinstallate.</p>
                    <p><strong>Ambienti Virtuali:</strong> Essenziali per isolare le dipendenze di progetti diversi. <code>conda</code> permette di creare ambienti separati, ognuno con la sua versione di Python e librerie, evitando conflitti. Comandi chiave: <code>conda create --name mio_ambiente python=3.x</code>, <code>conda activate mio_ambiente</code>, <code>conda install nome_libreria</code>, <code>conda deactivate</code>.</p>
                    <h3>Jupyter Notebook e Jupyter Lab: Strumenti Interattivi</h3>
                    <p><strong>Jupyter Notebook:</strong> Applicazione web open-source per creare e condividere documenti ("notebook", estensione <code>.ipynb</code>) che contengono codice vivo (es. Python), visualizzazioni, e testo narrativo (Markdown, equazioni).</p>
                    <p><strong>Jupyter Lab:</strong> Evoluzione di Jupyter Notebook, offre un'interfaccia più flessibile e potente, simile a un IDE, per lavorare con notebook, file di testo, terminali, ecc., in un'unica finestra.</p>
                    <p><em>Perché Jupyter è popolare?</em> Interattività (esecuzione cella per cella), documentazione integrata, visualizzazione immediata.</p>
                    <h3>Alternative Cloud: Google Colaboratory (Colab)</h3>
                    <p><strong>Google Colab (colab.research.google.com):</strong> Servizio gratuito di Google che fornisce un ambiente Jupyter Notebook direttamente nel browser, senza necessità di setup locale. </p>
                    <p><em>Vantaggi:</em> Zero setup, accesso gratuito a GPU/TPU (con limitazioni), preinstallazione di librerie comuni, facile condivisione (notebook salvati su Google Drive).</p>
                    <p><em>Svantaggi Potenziali:</em> Limitazioni risorse gratuite, gestione dati più macchinosa per grandi dataset rispetto all'ambiente locale.</p>
                    <p>È un'ottima opzione per iniziare, sperimentare (specialmente Deep Learning) e condividere.</p>` },
                { id: "ch6", title: "Capitolo 6: Manipolazione e Visualizzazione Dati con Python", content: `
                    <p>Con l'ambiente pronto, introduciamo le librerie Python fondamentali per caricare, pulire, trasformare, analizzare e visualizzare i dati.</p>
                    <h3>NumPy: Il Fondamento per il Calcolo Numerico</h3>
                    <p><strong>NumPy (Numerical Python)</strong> è la libreria base per il calcolo scientifico. Fornisce:</p>
                    <ul>
                        <li><strong>ndarray:</strong> Oggetto array N-dimensionale efficiente, molto più veloce delle liste Python per operazioni numeriche. Tutti gli elementi devono essere dello stesso tipo.</li>
                        <li><strong>Operazioni Vettorializzate:</strong> Permette di eseguire operazioni su interi array senza cicli <code>for</code> espliciti, rendendo il codice conciso e veloce.</li>
                        <li><strong>Funzioni Matematiche:</strong> Ampia collezione di funzioni per algebra lineare, trasformate di Fourier, generazione numeri casuali, ecc.</li>
                    </ul>
                    <p>È la base su cui sono costruite molte altre librerie, inclusa Pandas.</p>
                    <h3>Pandas: Analisi Dati Potente e Flessibile</h3>
                    <p><strong>Pandas</strong> è la libreria di riferimento per la manipolazione e l'analisi di dati tabulari. Introduce due strutture dati chiave:</p>
                    <ul>
                        <li><strong>Series:</strong> Array unidimensionale etichettato (simile a una colonna di un foglio di calcolo o a un dizionario). Ha un indice personalizzabile.</li>
                        <li><strong>DataFrame:</strong> Struttura dati bidimensionale etichettata (simile a una tabella SQL o un foglio di calcolo). È la struttura più usata. Può essere vista come una collezione di Series che condividono lo stesso indice. Ha indici di riga e di colonna.</li>
                    </ul>
                    <p><em>Funzionalità Principali:</em> Caricamento/salvataggio dati (CSV, Excel, SQL, JSON), selezione/filtraggio, gestione dati mancanti (NaN), raggruppamento (groupby), unione/join, trasformazione dati.</p>
                    <h3>Matplotlib e Seaborn: Visualizzazione dei Dati Essenziale</h3>
                    <p>Capire i dati solo da tabelle numeriche è difficile. La visualizzazione è cruciale per esplorare dati, identificare pattern, comunicare risultati e diagnosticare modelli.</p>
                    <h4>Matplotlib:</h4>
                    <p>È la libreria "madre" per creare grafici statici, animati e interattivi. Molto flessibile, permette controllo su quasi ogni aspetto del grafico. Il modulo <code>pyplot</code> (importato come <code>plt</code>) è l'interfaccia più comune.</p>
                    <h4>Seaborn:</h4>
                    <p>Costruita sopra Matplotlib, è orientata a creare grafici statistici più attraenti e informativi con meno codice. Semplifica la creazione di grafici comuni (mappe di calore, distribuzioni, regressioni) e si integra bene con i DataFrame Pandas.</p>
                    <p><em>Tipi di Grafici Comuni:</em> Grafici a linee, scatter plot, istogrammi, box plot, heatmap.</p>
                    <p>Padroneggiare queste librerie è essenziale per preparare i dati e comprendere i risultati nel Machine Learning.</p>` },
                { id: "ch7", title: "Capitolo 7: Scikit-learn: La Cassetta degli Attrezzi per il Machine Learning Classico", content: `
                                        <p>Dopo aver preparato l'ambiente e imparato a manipolare i dati, introduciamo la libreria chiave per applicare algoritmi di Machine Learning classici: <strong>Scikit-learn</strong> (spesso chiamata <code>sklearn</code>).</p>
                    <h3>Cos'è Scikit-learn?</h3>
                    <p>È una delle librerie Python open-source più popolari e complete per il ML. Offre implementazioni efficienti di un'ampia gamma di algoritmi di apprendimento supervisionato (classificazione, regressione) e non supervisionato (clustering, riduzione dimensionalità), insieme a strumenti per valutazione modelli, selezione feature, preprocessing dati, e altro.</p>
                    <h3>Perché è importante?</h3>
                    <ul>
                        <li><strong>Completezza:</strong> Vasta scelta di algoritmi consolidati.</li>
                        <li><strong>Efficienza:</strong> Molte implementazioni ottimizzate (spesso usando NumPy).</li>
                        <li><strong>Interfaccia Unificata e Semplice (API Estimator):</strong> Punto di forza principale. Tutti gli algoritmi condividono un'interfaccia coerente, facilitando la sperimentazione.</li>
                        <li><strong>Ottima Documentazione:</strong> Chiara, ricca di esempi e spiegazioni.</li>
                        <li><strong>Integrazione:</strong> Costruita su NumPy, SciPy, Matplotlib, si integra perfettamente nell'ecosistema scientifico Python.</li>
                        <li><strong>Open Source e Attivamente Sviluppata.</strong></li>
                    </ul>
                    <p><em>Cosa non fa (principalmente)?</em> Non è la libreria di riferimento per il Deep Learning avanzato (per quello useremo TensorFlow/Keras) né per analisi statistica inferenziale profonda.</p>
                    <h3>L'Interfaccia Comune di Scikit-learn: L'Oggetto Estimator</h3>
                    <p>Il cuore dell'API è l'<strong>Estimator</strong>. Quasi ogni oggetto che impara dai dati è un Estimator. Condividono metodi chiave:</p>
                    <ul>
                        <li><strong>Istanziazione:</strong> Si crea un'istanza dell'algoritmo specificando gli iperparametri (es. <code>model = RandomForestClassifier(n_estimators=100)</code>).</li>
                        <li><code>.fit(X, y=None)</code><strong>:</strong> Addestra l'Estimator. Input <code>X</code> (feature), e opzionalmente <code>y</code> (etichette per apprendimento supervisionato).</li>
                        <li><code>.predict(X_new)</code><strong>:</strong> (Per modelli supervisionati) Fa previsioni su nuovi dati <code>X_new</code>.</li>
                        <li><code>.predict_proba(X_new)</code><strong>:</strong> (Per modelli di classificazione) Restituisce le probabilità stimate per ogni classe.</li>
                        <li><code>.transform(X)</code><strong>:</strong> (Per preprocessatori e alcuni modelli non supervisionati) Trasforma i dati <code>X</code> in una nuova rappresentazione.</li>
                        <li><code>.fit_transform(X, y=None)</code><strong>:</strong> Esegue <code>fit</code> e poi <code>transform</code> sugli stessi dati (comodo per il training set).</li>
                        <li><code>.score(X, y)</code><strong>:</strong> (Per modelli supervisionati) Valuta le prestazioni del modello usando una metrica di default.</li>
                    </ul>
                    <h3>Funzioni Utili in Scikit-learn (Oltre agli Algoritmi)</h3>
                    <p>Scikit-learn offre molte utilità fondamentali:</p>
                    <ul>
                        <li><strong>Suddivisione Dati:</strong> <code>model_selection.train_test_split</code>, strumenti per Cross-Validation (<code>KFold</code>, <code>cross_val_score</code>).</li>
                        <li><strong>Preprocessing:</strong> Scaling (<code>StandardScaler</code>, <code>MinMaxScaler</code>), Encoding (<code>OneHotEncoder</code>, <code>LabelEncoder</code>), Gestione Valori Mancanti (<code>SimpleImputer</code>).</li>
                        <li><strong>Selezione Feature:</strong> Moduli come <code>feature_selection</code>.</li>
                        <li><strong>Metriche di Valutazione:</strong> Il modulo <code>metrics</code> (<code>accuracy_score</code>, <code>confusion_matrix</code>, <code>mean_squared_error</code>, ecc.).</li>
                        <li><strong>Pipeline:</strong> Strumenti (<code>pipeline.Pipeline</code>) per concatenare passaggi di preprocessing e modellazione in un unico Estimator.</li>
                    </ul>
                    <p>Con Scikit-learn, siamo pronti per esplorare i paradigmi di apprendimento e gli algoritmi specifici.</p>` }
            ]
        },
        {
            part: "Parte 3: I Paradigmi del Machine Learning",
            chapters: [
                { id: "ch8", title: "Capitolo 8: Apprendimento Supervisionato", content: `
                    <p>Iniziamo l'esplorazione dei paradigmi di ML con il più diffuso: l'Apprendimento Supervisionato. Qui, il modello impara da dati che includono sia gli input (feature) sia gli output desiderati (etichette).</p>
                    <h3>Concetto e Applicazioni</h3>
                    <p>Nell'Apprendimento Supervisionato, l'algoritmo è addestrato su un dataset dove ogni istanza è accompagnata da un'etichetta corretta. L'obiettivo è imparare una funzione che mappi gli input agli output, per generalizzare e predire l'output per nuovi input mai visti prima.</p>
                    <p>Si usa quando si ha un obiettivo di previsione chiaro e si dispone di dati storici etichettati.</p>
                    <h4>Due Tipi Principali di Problemi Supervisionati:</h4>
                    <ul>
                        <li><strong>Classificazione:</strong> L'obiettivo è predire un'etichetta che appartiene a un insieme finito di categorie discrete (classi).
                            <em>Esempi:</em> Filtri anti-spam (spam/non spam), riconoscimento immagini (gatto/cane), diagnosi medica (maligno/benigno).</li>
                        <li><strong>Regressione:</strong> L'obiettivo è predire un'etichetta che è un valore numerico continuo.
                            <em>Esempi:</em> Previsione prezzo case, stima vendite future, previsione temperatura.</li>
                    </ul>
                    <h3>Algoritmi Fondamentali (con cenni a Scikit-learn)</h3>
                    <p>Esistono molti algoritmi, vediamone alcuni fondamentali:</p>
                    <ul>
                        <li><strong>Regressione Lineare:</strong> (Regressione) Modella una relazione lineare tra feature e target. Cerca la linea (o iperpiano) che meglio si adatta ai dati minimizzando l'errore quadratico. Semplice e interpretabile. (<code>sklearn.linear_model.LinearRegression</code>)</li>
                        <li><strong>Regressione Logistica:</strong> (Classificazione, nonostante il nome) Usata per classificazione binaria. Modella la probabilità che un'istanza appartenga a una classe usando una funzione sigmoide. Interpretabile. (<code>sklearn.linear_model.LogisticRegression</code>)</li>
                        <li><strong>K-Nearest Neighbors (KNN):</strong> (Classificazione e Regressione) Algoritmo basato sull'istanza. Per classificare/predire una nuova istanza, guarda ai suoi K vicini più prossimi nel training set e usa la loro classe maggioritaria (classificazione) o la media dei loro valori (regressione). Sensibile alla scala delle feature e alla scelta di K. (<code>sklearn.neighbors.KNeighborsClassifier</code>, <code>KNeighborsRegressor</code>)</li>
                        <li><strong>Alberi Decisionali:</strong> (Classificazione e Regressione) Costruisce un modello simile a un diagramma di flusso, con nodi interni che testano le feature e nodi foglia che rappresentano la predizione. Molto interpretabili, possono catturare relazioni non lineari, ma tendono all'overfitting. (<code>sklearn.tree.DecisionTreeClassifier</code>, <code>DecisionTreeRegressor</code>)</li>
                        <li><strong>Support Vector Machine (SVM):</strong> (Principalmente Classificazione, ma esiste SVR per Regressione) Trova l'iperpiano che meglio separa le classi massimizzando il margine (distanza tra l'iperpiano e i punti più vicini delle diverse classi). Efficace in spazi ad alta dimensionalità e con dati non lineari usando il "kernel trick". (<code>sklearn.svm.SVC</code>, <code>SVR</code>)</li>
                    </ul>
                    <p>Questi algoritmi sono i mattoni per molti compiti di ML e base per metodi più complessi come gli ensemble.</p>` },
                { id: "ch9", title: "Capitolo 9: Apprendimento Non Supervisionato", content: `
                    <p>Passiamo ora all'Apprendimento Non Supervisionato, dove l'obiettivo è scoprire pattern e strutture nascoste in dati che <em>non</em> hanno etichette predefinite.</p>
                    <h3>Concetto e Applicazioni</h3>
                    <p>Nell'Apprendimento Non Supervisionato, l'algoritmo riceve solo dati di input (feature X) senza output corrispondenti (etichette y). Il compito è trovare autonomamente relazioni, gruppi o rappresentazioni significative all'interno dei dati.</p>
                    <p>Si usa per esplorazione dati, quando mancano etichette, per scoprire strutture intrinseche o per preprocessing.</p>
                    <h4>Due Tipi Principali di Problemi Non Supervisionati:</h4>
                    <ul>
                        <li><strong>Clustering:</strong> Raggruppare istanze simili in cluster (gruppi) distinti. Le istanze nello stesso cluster dovrebbero essere molto simili tra loro, e dissimili da quelle in altri cluster.
                            <em>Esempi:</em> Segmentazione clientela, organizzazione documenti, raggruppamento immagini simili.</li>
                        <li><strong>Riduzione della Dimensionalità:</strong> Ridurre il numero di feature (dimensioni) del dataset mantenendo quanta più informazione rilevante possibile.
                            <em>Esempi:</em> Visualizzazione dati ad alta dimensionalità, compressione dati, estrazione feature per migliorare prestazioni di altri modelli ML, riduzione rumore.</li>
                    </ul>
                    <p>(L'Anomaly Detection è un'altra area importante, spesso considerata parte dell'apprendimento non supervisionato).</p>
                    <h3>Algoritmi Fondamentali (con cenni a Scikit-learn)</h3>
                    <h4>Clustering:</h4>
                    <ul>
                        <li><strong>K-Means:</strong> Algoritmo popolare che partiziona i dati in K cluster predefiniti. Cerca di minimizzare la varianza intra-cluster (somma delle distanze quadratiche dei campioni dal centroide del loro cluster). Richiede di specificare K e può essere sensibile all'inizializzazione dei centroidi e alla scala delle feature. (<code>sklearn.cluster.KMeans</code>)</li>
                        <!-- Altri algoritmi di clustering come DBSCAN o Clustering Gerarchico potrebbero essere menzionati qui se presenti nell'ebook completo -->
                    </ul>
                    <h4>Riduzione della Dimensionalità:</h4>
                    <ul>
                        <li><strong>Principal Component Analysis (PCA) - Concettuale:</strong> Tecnica lineare che trasforma i dati in un nuovo sistema di coordinate dove le nuove feature (componenti principali) sono ordinate per varianza decrescente e sono ortogonali (non correlate). Le prime componenti catturano la maggior parte della varianza nei dati. Si possono scartare le componenti con meno varianza per ridurre la dimensionalità. Utile per visualizzazione e compressione. (<code>sklearn.decomposition.PCA</code>)</li>
                        <!-- Altre tecniche come t-SNE potrebbero essere menzionate qui se presenti nell'ebook completo -->
                    </ul>
                    <p>L'apprendimento non supervisionato è uno strumento potente per esplorare i dati e prepararli per ulteriori analisi o modelli supervisionati.</p>` },
                { id: "ch10", title: "Capitolo 10: Apprendimento per Rinforzo (Reinforcement Learning)", content: `
                    <p>Concludiamo l'esplorazione dei paradigmi di ML con l'Apprendimento per Rinforzo (RL), ispirato a come gli esseri viventi imparano interagendo con l'ambiente.</p>
                    <h3>Concetto e Applicazioni</h3>
                    <p>Nell'RL, un <strong>agente</strong> impara a prendere decisioni ottimali compiendo <strong>azioni</strong> in un <strong>ambiente</strong> per massimizzare una <strong>ricompensa</strong> cumulativa. A differenza del supervisionato, non ci sono etichette "giuste"; l'agente scopre quali azioni sono buone tramite un processo di tentativi ed errori (esplorazione vs sfruttamento).</p>
                    <p><em>Quando si usa?</em> Per problemi con decisioni sequenziali, ambienti complessi/dinamici e ottimizzazione a lungo termine.</p>
                    <p><em>Esempi di Applicazioni:</em> Giochi (AlphaGo), robotica (camminare, afferrare), sistemi di controllo autonomo (guida, gestione traffico), sistemi di raccomandazione personalizzati, finanza (trading).</p>
                    <h3>Elementi Chiave dell'Apprendimento per Rinforzo</h3>
                    <ul>
                        <li><strong>Agente (Agent):</strong> L'entità che apprende e prende decisioni.</li>
                        <li><strong>Ambiente (Environment):</strong> Il mondo esterno con cui l'agente interagisce.</li>
                        <li><strong>Stato (State - S):</strong> Descrizione della situazione attuale dell'ambiente.</li>
                        <li><strong>Azione (Action - A):</strong> Mossa che l'agente può compiere in un certo stato.</li>
                        <li><strong>Ricompensa (Reward - R):</strong> Segnale numerico che l'ambiente invia all'agente, indicando quanto è stata "buona" l'ultima azione nello stato precedente.</li>
                        <li><strong>Politica (Policy - π):</strong> Strategia che l'agente usa per decidere quale azione compiere in un dato stato. L'obiettivo è trovare la politica ottimale (π*).</li>
                        <li><strong>Valore (Value Function - V(s) o Q(s, a)):</strong> Stima della ricompensa futura attesa partendo da uno stato (V) o da una coppia stato-azione (Q).</li>
                        <li><strong>Episodio (Episode):</strong> Sequenza di interazioni agente-ambiente da uno stato iniziale a uno terminale.</li>
                    </ul>
                    <h3>Algoritmo Q-learning (Cenni)</h3>
                    <p>Il <strong>Q-learning</strong> è un algoritmo RL model-free (non necessita di un modello dell'ambiente) e off-policy (può imparare la politica ottimale anche seguendone una sub-ottimale).</p>
                    <p><em>Obiettivo:</em> Imparare la funzione valore azione-stato ottimale, Q*(s, a), che rappresenta la massima ricompensa cumulativa scontata ottenibile partendo dallo stato <em>s</em>, compiendo l'azione <em>a</em>, e comportandosi ottimamente da lì in poi.</p>
                    <p><em>Tabella Q (Q-Table):</em> Per problemi con stati e azioni discreti e finiti, Q*(s, a) può essere rappresentata come una tabella.</p>
                    <p><em>Processo Iterativo:</em> L'agente interagisce con l'ambiente per episodi. In ogni passo:</p>
                    <ol>
                        <li>Osserva lo stato <em>s</em>.</li>
                        <li>Sceglie un'azione <em>a</em> (spesso usando una politica ε-greedy per bilanciare esplorazione e sfruttamento).</li>
                        <li>Compie l'azione <em>a</em>, osserva la ricompensa <em>r</em> e il nuovo stato <em>s'</em>.</li>
                        <li>Aggiorna il valore Q(s, a) nella tabella usando la regola di Bellman:
                            <code>Q(s,a) ← Q(s,a) + α * [r + γ * max<sub>a'</sub>Q(s',a') - Q(s,a)]</code>
                            (dove α è il tasso di apprendimento e γ è il fattore di sconto).</li>
                    </ol>
                    <p><em>Politica Ottimale:</em> Una volta appresa la Q-table, la politica ottimale è scegliere in ogni stato l'azione che massimizza Q(s,a).</p>
                    <p><em>Sfide:</em> Spazi stato/azione grandi (richiedono approssimazione della funzione Q, es. con reti neurali - Deep Q-Networks), ricompense sparse, sample efficiency.</p>
                    <p>L'RL è un campo affascinante e in rapida evoluzione, che alimenta molte delle applicazioni AI più avanzate.</p>` }
            ]
        },
        {
            part: "Parte 4: La Pipeline di Sviluppo di un Modello ML",
            chapters: [
                { id: "ch11", title: "Capitolo 11: Preparazione dei Dati (Data Preprocessing)", content: `
                    <p>Nei capitoli precedenti abbiamo esplorato i concetti teorici e gli algoritmi. Ora ci concentriamo sul processo pratico per costruire un modello ML. Il primo passo, cruciale e spesso dispendioso, è la preparazione dei dati (Data Preprocessing).</p>
                    <p>Il principio GIGO (Garbage In, Garbage Out) sottolinea che la qualità del modello dipende dalla qualità dei dati. I dati reali sono raramente "puliti" e pronti. Il preprocessing trasforma i dati grezzi in un formato pulito, coerente e adatto all'addestramento.</p>
                    <h3>Fasi Comuni del Data Preprocessing (con Pandas e Scikit-learn):</h3>
                    <ul>
                        <li><strong>Caricamento Dati:</strong> Importare i dati da file (CSV, Excel, database) in un DataFrame Pandas (es. <code>pd.read_csv()</code>). Esplorazione iniziale con <code>.head()</code>, <code>.info()</code>, <code>.describe()</code>.</li>
                        <li><strong>Gestione Valori Mancanti (Missing Values):</strong> Identificare (<code>.isnull().sum()</code>) e gestire i NaN. Strategie:
                            <ul>
                                <li><em>Eliminazione (Dropping):</em> Rimuovere righe (<code>.dropna(axis=0)</code>) o colonne (<code>.dropna(axis=1)</code>). Rischio di perdita dati.</li>
                                <li><em>Imputazione (Imputation):</em> Sostituire NaN con valori stimati (media, mediana, moda, o valori da modelli ML). Scikit-learn offre <code>SimpleImputer</code>.</li>
                            </ul>
                        </li>
                        <li><strong>Scaling e Normalizzazione delle Feature Numeriche:</strong> Portare le feature numeriche su una scala comparabile, importante per algoritmi sensibili alle distanze (KNN, SVM, Reti Neurali). Eseguire <em>dopo</em> la suddivisione train/test, adattando lo scaler solo sul training set.
                            <ul>
                                <li><em>Standardizzazione (<code>StandardScaler</code>):</em> Trasforma i dati a media 0 e deviazione standard 1 (Z-score).</li>
                                <li><em>Normalizzazione (<code>MinMaxScaler</code>):</em> Scala i dati in un range specifico (es. [0, 1]).</li>
                            </ul>
                        </li>
                        <li><strong>Encoding delle Variabili Categoriche:</strong> Convertire feature testuali/categoriche in numeri.
                            <ul>
                                <li><em>Label Encoding (<code>LabelEncoder</code>):</em> Assegna un intero univoco a ogni categoria. Adatto per target o feature ordinali. Può introdurre un ordine fittizio se usato su feature nominali.</li>
                                <li><em>One-Hot Encoding (<code>OneHotEncoder</code>, <code>pd.get_dummies()</code>):</em> Crea una nuova colonna binaria (0/1) per ogni categoria. Standard per feature nominali, evita ordini fittizi, ma può aumentare molto le dimensioni.</li>
                            </ul>
                        </li>
                        <li><strong>Suddivisione del Dataset:</strong> Dividere i dati in training set (per addestrare il modello) e test set (per valutarne le prestazioni finali su dati mai visti). Fondamentale per evitare data leakage e ottenere una stima realistica della generalizzazione. Usare <code>sklearn.model_selection.train_test_split</code>.</li>
                    </ul>
                    <p>Un preprocessing accurato è la base per modelli ML efficaci.</p>` },
                { id: "ch12", title: "Capitolo 12: Costruzione e Addestramento del Modello (con Scikit-learn)", content: `
                    <p>Con i dati preparati, siamo pronti per il cuore del processo: selezionare un algoritmo, istanziarlo come modello e addestrarlo sui dati di training.</p>
                    <h3>1. Scegliere un Modello (Algoritmo)</h3>
                    <p>La scelta dipende da vari fattori:</p>
                    <ul>
                        <li><strong>Tipo di Problema:</strong> Classificazione, Regressione, Clustering, ecc.</li>
                        <li><strong>Dimensione e Caratteristiche del Dataset:</strong> Pochi/molti dati, alta/bassa dimensionalità, linearità.</li>
                        <li><strong>Interpretabilità:</strong> Necessità di capire le decisioni del modello (es. Regressione Lineare/Logistica, Alberi Decisionali sono più interpretabili di SVM con kernel complessi o Reti Neurali profonde).</li>
                        <li><strong>Tempo di Addestramento/Predizione:</strong> Alcuni algoritmi sono più veloci di altri.</li>
                        <li><strong>Prestazioni Desiderate:</strong> Spesso si provano diversi algoritmi e si confrontano.</li>
                    </ul>
                    <p>La <em>Scikit-learn Cheat Sheet</em> può aiutare nella scelta iniziale: <a href="https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html" target="_blank" rel="noopener noreferrer">scikit-learn.org/stable/tutorial/machine_learning_map/index.html</a></p>
                    <h3>2. Istanziare il Modello (Estimator)</h3>
                    <p>Una volta scelto l'algoritmo, si crea un'istanza dell'oggetto Estimator corrispondente in Scikit-learn, specificando gli <strong>iperparametri</strong> desiderati. Scikit-learn fornisce valori di default ragionevoli, ma il tuning (vedi Cap. 14) è spesso necessario per ottimizzare le prestazioni.</p>
                    <p><em>Esempi:</em></p>
                    <pre><code>from sklearn.linear_model import LogisticRegression
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.ensemble import RandomForestClassifier

                    log_reg_model = LogisticRegression(random_state=42)
                    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)</code></pre>
                    <h3>3. Addestrare il Modello (<code>.fit()</code>)</h3>
                    <p>Questa è la fase di apprendimento. Si utilizza il metodo <code>.fit()</code> dell'Estimator, fornendogli i dati di training preprocessati (<code>X_train</code>) e, per l'apprendimento supervisionato, le etichette di training (<code>y_train</code>).</p>
                    <p><code>model.fit(X_train_scaled, y_train)</code></p>
                    <p>Durante <code>.fit()</code>, l'algoritmo analizza i dati e stima i parametri interni del modello (es. coefficienti per Regressione Lineare, struttura dell'albero per Alberi Decisionali, vettori di supporto per SVM).</p>
                    <p><strong>Importante:</strong> Si usa solo il training set per il <code>.fit()</code>. Il test set rimane "invisibile" al modello fino alla fase di valutazione.</p>
                    <h3>4. Fare Previsioni (<code>.predict()</code>, <code>.predict_proba()</code>)</h3>
                    <p>Una volta addestrato (fitted), il modello è pronto per fare previsioni su nuovi dati (tipicamente, il test set <code>X_test_scaled</code>, che deve aver subito le stesse trasformazioni di preprocessing del training set).</p>
                    <ul>
                        <li><code>model.predict(X_test_scaled)</code>: Restituisce l'etichetta predetta (classe o valore di regressione) per ogni istanza in <code>X_test_scaled</code>.</li>
                        <li><code>model.predict_proba(X_test_scaled)</code>: (Solo per modelli di classificazione che lo supportano) Restituisce le probabilità stimate per ciascuna classe. Utile per capire la "fiducia" del modello.</li>
                    </ul>
                    <p>Il passo successivo è valutare quanto sono buone queste previsioni.</p>` },
                { id: "ch13", title: "Capitolo 13: Valutazione delle Prestazioni del Modello", content: `
                    <p>Dopo aver addestrato un modello e fatto previsioni, è cruciale valutarne le prestazioni. Come facciamo a sapere se il modello è "buono" e generalizza bene a dati mai visti?</p>
                    <p>Valutare solo sul training set è ingannevole (rischio di overfitting). La valutazione si fa quasi sempre sul <strong>test set</strong>.</p>
                    <h3>1. Metriche per la Classificazione</h3>
                    <p>Scelte in base al problema e alla distribuzione delle classi.</p>
                    <ul>
                        <li><strong>Accuratezza (Accuracy):</strong> (Predizioni Corrette / Totale Predizioni). Intuitiva, ma fuorviante con dataset sbilanciati. (<code>sklearn.metrics.accuracy_score</code>)</li>
                        <li><strong>Matrice di Confusione:</strong> Tabella che mostra TP, TN, FP, FN, dando una visione dettagliata degli errori. (<code>sklearn.metrics.confusion_matrix</code>)</li>
                        <li><strong>Precision:</strong> TP / (TP + FP). Delle istanze predette positive, quante erano realmente positive? Utile quando il costo dei FP è alto. (<code>sklearn.metrics.precision_score</code>)</li>
                        <li><strong>Recall (Sensitivity, True Positive Rate):</strong> TP / (TP + FN). Delle istanze realmente positive, quante sono state identificate? Utile quando il costo dei FN è alto. (<code>sklearn.metrics.recall_score</code>)</li>
                        <li><strong>F1-Score:</strong> Media armonica di Precision e Recall (2 * (Precision * Recall) / (Precision + Recall)). Utile per bilanciare Precision e Recall, specialmente con dati sbilanciati. (<code>sklearn.metrics.f1_score</code>)</li>
                        <li><strong>Report di Classificazione:</strong> Mostra Precision, Recall, F1-Score per ogni classe. (<code>sklearn.metrics.classification_report</code>)</li>
                        <li><strong>Curva ROC (Receiver Operating Characteristic) e AUC (Area Under the Curve):</strong> La curva ROC plotta TPR vs FPR (False Positive Rate) a varie soglie di classificazione. L'AUC misura l'area sotto la curva ROC (da 0.5 a 1), indicando la capacità del modello di distinguere le classi. Ottima per dati sbilanciati. (<code>sklearn.metrics.roc_curve</code>, <code>roc_auc_score</code>)</li>
                    </ul>
                    <h3>2. Metriche per la Regressione</h3>
                    <p>Misurano quanto le previsioni numeriche si avvicinano ai valori reali.</p>
                    <ul>
                        <li><strong>Mean Absolute Error (MAE):</strong> Media degli errori assoluti. Stessa unità del target, meno sensibile agli outlier. (<code>sklearn.metrics.mean_absolute_error</code>)</li>
                        <li><strong>Mean Squared Error (MSE):</strong> Media degli errori quadratici. Penalizza di più gli errori grandi, unità del target al quadrato. (<code>sklearn.metrics.mean_squared_error</code>)</li>
                        <li><strong>Root Mean Squared Error (RMSE):</strong> Radice quadrata dell'MSE. Stessa unità del target, più interpretabile dell'MSE. (<code>np.sqrt(mean_squared_error(...))</code>)</li>
                        <li><strong>R-squared (R² o Coefficiente di Determinazione):</strong> Proporzione della varianza nel target spiegata dal modello (da ~0 a 1). 1 è perfetto, 0 significa che il modello non spiega nulla. (<code>sklearn.metrics.r2_score</code>)</li>
                    </ul>
                    <h3>3. Convalida Incrociata (Cross-Validation)</h3>
                    <p>Una singola suddivisione train/test può dare stime di performance instabili. La <strong>K-Fold Cross-Validation</strong> è più robusta:</p>
                    <ol>
                        <li>Divide il training set in K fold (sottoinsiemi).</li>
                        <li>Per ogni fold <em>i</em> (da 1 a K):
                            <ul>
                                <li>Usa il fold <em>i</em> come validation set temporaneo.</li>
                                <li>Usa i restanti K-1 fold come training set temporaneo.</li>
                                <li>Addestra il modello e lo valuta sul validation set.</li>
                            </ul>
                        </li>
                        <li>Calcola la media (e dev. std.) delle K metriche ottenute.</li>
                    </ol>
                    <p>Fornisce una stima più affidabile della generalizzazione. Usata anche per il tuning degli iperparametri. (<code>sklearn.model_selection.cross_val_score</code>)</p>
                    <h3>4. Curve di Apprendimento (Learning Curves)</h3>
                    <p>Plottano le prestazioni del modello (es. errore) su training e validation set in funzione della dimensione del training set. Utili per diagnosticare underfitting (errore alto per entrambi) o overfitting (gap grande tra errore training basso e errore validation alto).</p>
                    <p>(<code>sklearn.model_selection.learning_curve</code>)</p>` },
                { id: "ch14", title: "Capitolo 14: Miglioramento del Modello e Feature Engineering", content: `
                    <p>Dopo la prima valutazione, spesso c'è margine per migliorare le prestazioni del modello. Due strategie principali sono l'ottimizzazione degli iperparametri e il feature engineering.</p>
                    <h3>1. Ottimizzazione degli Iperparametri (Hyperparameter Tuning)</h3>
                    <p>Gli iperparametri sono le "manopole" dell'algoritmo (es. K in KNN, <code>max_depth</code> in Alberi Decisionali). Trovare la combinazione ottimale può migliorare significativamente le prestazioni.</p>
                    <h4>Tecniche Comuni in Scikit-learn:</h4>
                    <ul>
                        <li><strong>GridSearchCV:</strong> Prova tutte le combinazioni di iperparametri specificati in una griglia. Esaustivo ma computazionalmente costoso.
                            <p><em>Input:</em> Estimator, griglia di parametri (dizionario), numero di fold CV, metrica di scoring.</p>
                            <p><em>Output:</em> <code>.best_params_</code>, <code>.best_score_</code>, <code>.best_estimator_</code>.</p>
                        </li>
                        <li><strong>RandomizedSearchCV:</strong> Campiona un numero fisso (<code>n_iter</code>) di combinazioni casuali dallo spazio degli iperparametri (che può includere distribuzioni). Più efficiente di GridSearchCV per spazi grandi.
                            <p><em>Input/Output:</em> Simili a GridSearchCV, ma con <code>n_iter</code> e distribuzioni di parametri.</p>
                        </li>
                    </ul>
                    <p><strong>Importante:</strong> Il tuning va fatto usando il training set (idealmente con CV interna). Il test set finale deve rimanere intoccato per la valutazione del modello finale ottimizzato.</p>
                    <h3>2. Feature Engineering</h3>
                    <p>Spesso, il modo più efficace per migliorare un modello non è solo ottimizzare l'algoritmo, ma migliorare la qualità degli input (le feature). È l'arte e la scienza di usare la conoscenza del dominio e tecniche matematiche per trasformare dati grezzi in feature che rappresentino meglio il problema.</p>
                    <h4>Tecniche Comuni:</h4>
                    <ul>
                        <li><strong>Creazione di Nuove Feature (Feature Creation):</strong>
                            <ul>
                                <li><em>Combinare Feature Esistenti:</em> Creare interazioni (es. prodotto di due feature) o rapporti. Esempio: da 'Lunghezza_Giardino' e 'Larghezza_Giardino' -> 'Area_Giardino'.</li>
                                <li><em>Estrarre Informazioni da Dati Complessi:</em> Da date (giorno/mese/ora), testo (lunghezza, sentiment), coordinate (distanze).</li>
                                <li><em>Usare Conoscenza del Dominio:</em> Incorporare regole di business (es. Indice di Massa Corporea da peso/altezza).</li>
                                <li><em>Feature Polinomiali (<code>PolynomialFeatures</code>):</em> Generare potenze e prodotti di feature. Utile per catturare relazioni non lineari con modelli lineari (usare con cautela per rischio overfitting).</li>
                            </ul>
                        </li>
                        <li><strong>Trasformazione di Feature Esistenti:</strong>
                            <ul>
                                <li><em>Trasformazioni Matematiche:</em> Logaritmo, radice quadrata, esponenziale per cambiare distribuzione o stabilizzare varianza.</li>
                                <li><em>Discretizzazione (Binning):</em> Convertire feature numeriche continue in categoriche ordinali (es. Età -> 'Giovane', 'Adulto').</li>
                            </ul>
                        </li>
                        <li><strong>Selezione delle Feature (Feature Selection):</strong> Rimuovere feature irrilevanti o ridondanti.
                            <p><em>Perché?</em> Riduce complessità, overfitting, accelera addestramento/predizione, migliora interpretabilità.</p>
                            <p><em>Metodi:</em></p>
                            <ul>
                                <li><em>Filtro (Filter):</em> Valutano feature indipendentemente dal modello (es. correlazione, test statistici come <code>SelectKBest</code>). Veloci, ma non considerano interazioni.</li>
                                <li><em>Wrapper:</em> Usano un modello specifico per valutare sottoinsiemi di feature (es. Recursive Feature Elimination - <code>RFE</code>). Più accurati, ma più costosi.</li>
                                <li><em>Embedded:</em> La selezione avviene durante l'addestramento del modello (es. Regolarizzazione L1/Lasso, importanza feature da Random Forest).</li>
                            </ul>
                        </li>
                    </ul>
                    <p>Il Feature Engineering è un processo iterativo e spesso il più impattante per le prestazioni del modello.</p>` }
            ]
        },
        {
            part: "Parte 5: Introduzione al Deep Learning",
            chapters: [
                { id: "ch15", title: "Capitolo 15: Dalle Reti Neurali al Deep Learning", content: `
                    <p>Nei capitoli precedenti abbiamo esplorato il Machine Learning "classico". Ora entriamo nel mondo del Deep Learning (DL), un sottoinsieme del ML che ha rivoluzionato l'AI, specialmente con dati non strutturati come immagini, audio e testo.</p>
                    <h3>1. Ispirazione Biologica: Il Neurone</h3>
                    <p>Il DL si ispira (vagamente) alla struttura e al funzionamento del cervello umano, in particolare al <strong>neurone biologico</strong>. Questo riceve segnali attraverso i dendriti, li elabora nel soma, e se la somma dei segnali supera una soglia, "si attiva" inviando un segnale attraverso l'assone ad altri neuroni tramite sinapsi. L'apprendimento è associato alla modifica della forza di queste connessioni.</p>
                    <h3>2. Il Perceptron: Il Mattone Fondamentale</h3>
                    <p>Il <strong>Perceptron</strong> (Frank Rosenblatt, 1957) è il primo modello matematico di un neurone artificiale.</p>
                    <ul>
                        <li><strong>Input:</strong> Riceve valori di input (x1, x2, ..., xn).</li>
                        <li><strong>Pesi (Weights):</strong> A ogni input è associato un peso (w1, w2, ..., wn), che ne rappresenta l'importanza.</li>
                        <li><strong>Somma Pesata:</strong> Calcola <code>z = (x1*w1) + ... + (xn*wn) + b</code> (dove <code>b</code> è il bias, un'intercetta).</li>
                        <li><strong>Funzione di Attivazione:</strong> Il risultato <code>z</code> passa attraverso una funzione di attivazione (nel Perceptron originale, una step function: output 1 se z > soglia, 0 altrimenti).</li>
                    </ul>
                    <p>Un singolo Perceptron può risolvere solo problemi linearmente separabili (es. non il problema XOR).</p>
                    <h3>3. Reti Neurali Artificiali (ANN): Struttura a Strati</h3>
                    <p>Per superare i limiti del singolo neurone, si collegano più neuroni in strati (layers), formando una <strong>Rete Neurale Artificiale (ANN)</strong> o Multi-Layer Perceptron (MLP).</p>
                    <p>Una tipica ANN feedforward (informazione fluisce in una direzione) ha:</p>
                    <ul>
                        <li><strong>Strato di Input (Input Layer):</strong> Rappresenta le feature del dato.</li>
                        <li><strong>Strati Nascosti (Hidden Layers):</strong> Uno o più strati intermedi dove avviene l'elaborazione. Ogni neurone riceve input da tutti i neuroni dello strato precedente e invia output a quelli successivi. Permettono di apprendere rappresentazioni complesse.</li>
                        <li><strong>Strato di Output (Output Layer):</strong> Produce il risultato/predizione finale. Il numero di neuroni e la funzione di attivazione dipendono dal tipo di problema (regressione, classificazione binaria/multi-classe).</li>
                    </ul>
                    <h3>4. Funzioni di Attivazione (Non Lineari!)</h3>
                    <p>Se si usassero solo attivazioni lineari (o nessuna) negli strati nascosti, l'intera rete si comporterebbe come un singolo strato lineare. Le <strong>funzioni di attivazione non lineari</strong> sono cruciali per permettere alle ANN di modellare relazioni complesse.</p>
                    <p>Comuni funzioni di attivazione:</p>
                    <ul>
                        <li><strong>Sigmoide (Sigmoid):</strong> Output tra 0 e 1. Usata storicamente, ora meno negli strati nascosti (problema vanishing gradient), a volte nell'output per classificazione binaria.</li>
                        <li><strong>Tangente Iperbolica (Tanh):</strong> Output tra -1 e 1. Simile a sigmoide ma centrata a zero, spesso converge più velocemente. Soffre anch'essa di vanishing gradient.</li>
                        <li><strong>ReLU (Rectified Linear Unit):</strong> Output <code>max(0, z)</code>. Molto popolare per strati nascosti: efficiente, non satura per valori positivi. Può soffrire del "neurone morente". Varianti: Leaky ReLU, PReLU.</li>
                        <li><strong>Softmax:</strong> Usata nello strato di output per classificazione multi-classe. Converte gli output dei neuroni in una distribuzione di probabilità (valori tra 0 e 1 che sommano a 1).</li>
                    </ul>
                    <h3>5. Cos'è il Deep Learning? Reti Profonde</h3>
                    <p>Il <strong>Deep Learning</strong> è un tipo di Machine Learning che utilizza Reti Neurali Artificiali con <strong>molteplici strati nascosti</strong> (reti "profonde").</p>
                    <p>La "profondità" permette di apprendere <strong>gerarchie di feature</strong>: strati iniziali imparano feature semplici (es. bordi in un'immagine), strati successivi combinano queste feature per impararne di più complesse (es. forme, oggetti).</p>
                    <p><em>Perché ora?</em> Big Data (dataset enormi), Hardware Potente (GPU), Miglioramenti Algoritmici (funzioni di attivazione, regolarizzazione, ottimizzatori).</p>` },
                { id: "ch16", title: "Capitolo 16: Framework per il Deep Learning: TensorFlow e Keras", content: `
                    <p>Costruire e addestrare reti neurali profonde da zero sarebbe complesso. Fortunatamente, esistono potenti librerie (framework) che semplificano questo processo.</p>
<h3>1. TensorFlow: La Piattaforma Completa</h3>
<p>Sviluppato da Google (2015), <strong>TensorFlow (tensorflow.org)</strong> è una piattaforma end-to-end open-source per il Machine Learning, con un forte focus sul Deep Learning.</p>
<ul>
    <li><strong>Tensori:</strong> Unità dati fondamentale (array multidimensionali, come NumPy).</li>
    <li><strong>Grafo Computazionale (Storico vs Eager Execution):</strong> TensorFlow 1.x usava grafi statici. TensorFlow 2.x ha adottato l'<em>eager execution</em> di default (operazioni eseguite immediatamente), rendendolo più intuitivo e pythonico. I grafi sono ancora usati "sotto il cofano" per ottimizzazione (<code>tf.function</code>).</li>
    <li><strong>Caratteristiche:</strong> Flessibilità (API basso/alto livello), Scalabilità (CPU, GPU, TPU, cluster), Deployment (TF Serving, TF Lite, TF.js), vasto ecosistema (TensorBoard, TF Hub).</li>
</ul>
<h3>2. Keras: L'API ad Alto Livello per Reti Neurali</h3>
<p>Sviluppata da François Chollet (2015), <strong>Keras (keras.io)</strong> è un'API di altissimo livello per definire e addestrare reti neurali, focalizzata su facilità d'uso, prototipazione rapida e leggibilità.</p>
<p><strong>Relazione con TensorFlow:</strong> Keras è stata adottata come l'API di alto livello ufficiale di TensorFlow (<code>tf.keras</code>). Quando si usa <code>tf.keras</code>, si usa l'interfaccia Keras con la potenza di TensorFlow come backend.</p>
<p><em>Perché usare Keras (tf.keras)?</em> Semplicità estrema, prototipazione rapida, leggibilità, componenti modulari, integrazione totale con TensorFlow.</p>
<h3>Costruire Modelli con Keras (tf.keras)</h3>
<h4>Modi per Definire l'Architettura:</h4>
<ul>
    <li><strong>API Sequenziale (<code>tf.keras.models.Sequential</code>):</strong> Per reti "semplici" con strati impilati linearmente. Si aggiungono strati con <code>.add()</code>.</li>
    <li><strong>API Funzionale (Functional API):</strong> Più flessibile, per architetture arbitrarie (input/output multipli, strati condivisi, grafi). Si definiscono strati come funzioni e li si collega esplicitamente.</li>
</ul>
<h4>Tipi di Strati Comuni (<code>tf.keras.layers</code>):</h4>
<ul>
    <li><code>Dense</code>: Strato "fully connected". Parametri: <code>units</code>, <code>activation</code>.</li>
    <li><code>Activation</code>: Per applicare una funzione di attivazione come strato separato.</li>
    <li><code>Dropout</code>: Tecnica di regolarizzazione (spegne neuroni casualmente durante training). Parametro: <code>rate</code>.</li>
    <li><code>Flatten</code>: Appiattisce input multidimensionali (es. da CNN a Dense).</li>
    <li>Strati Convoluzionali (<code>Conv2D</code>, <code>MaxPooling2D</code>) e Ricorrenti (<code>LSTM</code>, <code>GRU</code>) verranno visti nei prossimi capitoli.</li>
</ul>
<h4>Compilare il Modello (<code>.compile()</code>)</h4>
<p>Configura il processo di apprendimento, specificando:</p>
<ul>
    <li><strong>Ottimizzatore (Optimizer):</strong> Algoritmo per aggiornare i pesi (es. <code>'adam'</code>, <code>'sgd'</code>, <code>'rmsprop'</code>). Learning rate è un iperparametro chiave.</li>
    <li><strong>Funzione di Perdita (Loss Function):</strong> Misura quanto è "sbagliata" la previsione (es. <code>'binary_crossentropy'</code>, <code>'categorical_crossentropy'</code>, <code>'mean_squared_error'</code>).</li>
    <li><strong>Metriche (Metrics):</strong> Da monitorare durante training/valutazione (es. <code>['accuracy']</code>, <code>['mae']</code>).</li>
</ul>
<h4>Addestrare un Modello Keras (<code>.fit()</code>)</h4>
<p>Simile a Scikit-learn, ma con specificità DL:</p>
<ul>
    <li><strong>Input:</strong> <code>X_train</code>, <code>y_train</code>.</li>
    <li><code>epochs</code>: Numero di passaggi completi sul dataset di training.</li>
    <li><code>batch_size</code>: Numero di campioni elaborati prima di aggiornare i pesi.</li>
    <li><code>validation_data</code>: (<code>X_val</code>, <code>y_val</code>) per monitorare overfitting e usare Early Stopping.</li>
</ul>
<p>L'oggetto <code>history</code> restituito da <code>.fit()</code> contiene le metriche di training e validazione per ogni epoca, utili per plottare le curve di apprendimento.</p>` },
                { id: "ch17", title: "Capitolo 17: Deep Learning per le Immagini: Convolutional Neural Networks (CNN)", content: `
                    <p>Le reti neurali dense (fully connected) hanno limiti con dati ad alta dimensionalità come le immagini: numero enorme di parametri, perdita di struttura spaziale, mancanza di invarianza traslazionale. Le <strong>Reti Neurali Convoluzionali (CNN o ConvNet)</strong> sono progettate specificamente per superare questi limiti.</p>
<h3>1. Ispirazione: Il Sistema Visivo Biologico</h3>
<p>Le CNN traggono vaga ispirazione dalla corteccia visiva, dove neuroni specifici rispondono a stimoli in piccole regioni del campo visivo ("campi recettivi") e feature più complesse sono costruite gerarchicamente.</p>
<h3>2. Concetti Chiave delle CNN</h3>
<h4>Strato Convoluzionale (<code>Conv2D</code>):</h4>
<ul>
    <li><strong>Filtri (Kernel):</strong> Piccole matrici di pesi (es. 3x3, 5x5) che "scorrono" sull'immagine di input. Ogni filtro impara a rilevare un pattern specifico (bordi, texture, forme semplici).</li>
    <li><strong>Condivisione dei Pesi (Weight Sharing):</strong> Lo stesso filtro viene applicato su tutta l'immagine, permettendo di rilevare il pattern indipendentemente dalla sua posizione (invarianza traslazionale) e riducendo drasticamente il numero di parametri.</li>
    <li><strong>Mappe di Attivazione (Feature Maps):</strong> L'output di un filtro applicato all'immagine. Uno strato Conv2D tipicamente usa molti filtri, producendo un volume di feature map (una per filtro).</li>
    <li><em>Parametri Keras:</em> <code>filters</code> (numero di filtri), <code>kernel_size</code>, <code>strides</code>, <code>padding</code> ('valid' o 'same'), <code>activation</code> (spesso 'relu').</li>
</ul>
<h4>Strato di Pooling (<code>MaxPooling2D</code>):</h4>
<ul>
    <li><strong>Obiettivo:</strong> Ridurre la dimensione spaziale delle feature map, mantenendo le informazioni più importanti, per ridurre la computazione e controllare l'overfitting.</li>
    <li><strong>Max Pooling:</strong> Divide la feature map in griglie (es. 2x2) e prende il valore massimo da ogni griglia. Rende la rappresentazione più robusta a piccole traslazioni.</li>
    <li><em>Parametri Keras:</em> <code>pool_size</code>, <code>strides</code>, <code>padding</code>.</li>
</ul>
<h3>3. Architettura Tipica di una CNN</h3>
<p>Una CNN per classificazione immagini ha spesso questa struttura:</p>
<ol>
    <li><strong>Input Layer:</strong> Immagine (es. 32x32x3 pixel).</li>
    <li><strong>Blocchi Convoluzionali:</strong> Si ripetono:
        <ul>
            <li>Strato(i) <code>Conv2D</code> (con attivazione ReLU) per estrarre feature.</li>
            <li>Strato <code>MaxPooling2D</code> per ridurre dimensionalità.</li>
        </ul>
        (Man mano che si va in profondità, la dimensione spaziale HxW delle feature map diminuisce, mentre il numero di filtri/canali aumenta).
    </li>
    <li><strong>Strato <code>Flatten</code>:</strong> Appiattisce l'output 3D dell'ultimo blocco convoluzionale/pooling in un vettore 1D.</li>
    <li><strong>Strati Densi (Fully Connected):</strong> Uno o più strati <code>Dense</code> (con ReLU e Dropout) per combinare le feature di alto livello.</li>
    <li><strong>Strato di Output:</strong> Strato <code>Dense</code> finale con attivazione appropriata (es. <code>'sigmoid'</code> per binaria, <code>'softmax'</code> per multi-classe).</li>
</ol>
<h3>4. Data Augmentation per Immagini</h3>
<p>Tecnica per aumentare artificialmente la dimensione del training set applicando trasformazioni casuali (rotazioni, zoom, flip, shift) alle immagini originali. Aiuta a prevenire l'overfitting e a rendere il modello più robusto. Keras offre layer per questo (es. <code>tf.keras.layers.RandomFlip</code>).</p>
<h3>5. Transfer Learning: Usare Modelli Pre-addestrati</h3>
<p>Addestrare CNN profonde da zero richiede molti dati e tempo. Il <strong>Transfer Learning</strong> sfrutta modelli potenti (es. VGG16, ResNet50, MobileNet, EfficientNet), già addestrati su dataset enormi come ImageNet.</p>
<p><em>Come funziona:</em></p>
<ol>
    <li>Si prende un modello base pre-addestrato (es. <code>tf.keras.applications.MobileNetV2</code>).</li>
    <li>Si "congelano" i pesi degli strati convoluzionali iniziali/intermedi (<code>base_model.trainable = False</code>), che hanno imparato feature visive generiche.</li>
    <li>Si rimuove lo strato di classificazione originale.</li>
    <li>Si aggiunge un nuovo classificatore in cima (alcuni strati Densi + strato output) adatto al proprio problema.</li>
    <li>Si addestra solo il nuovo classificatore sul proprio dataset (più piccolo).</li>
    <li>(Opzionale) Fine-tuning: Si "scongelano" alcuni degli ultimi strati del modello base e si riaddestra l'intera rete con un learning rate molto basso.</li>
</ol>
<p>Porta a prestazioni migliori con meno dati/tempo. OpenCV è una libreria utile per il preprocessing base delle immagini.</p>` },
                { id: "ch18", title: "Capitolo 18: Deep Learning per Sequenze: RNN e LSTM", content: `
                    <p>Mentre le CNN eccellono con dati a griglia (immagini), altri dati hanno una natura sequenziale dove l'ordine è cruciale (linguaggio, serie temporali). Le <strong>Reti Neurali Ricorrenti (RNN)</strong> sono progettate per questi dati.</p>
<h3>1. Il Problema dei Dati Sequenziali: Memoria del Passato</h3>
<p>Per predire il prossimo elemento in una sequenza (es. prossima parola), è necessario "ricordare" il contesto precedente. Reti Dense o CNN standard processano ogni input indipendentemente, mancando di questa memoria.</p>
<h3>2. Idea di Base delle Reti Neurali Ricorrenti (RNN Semplici)</h3>
<p>Una RNN elabora una sequenza un elemento (passo temporale) alla volta. La caratteristica chiave è un <strong>ciclo (loop)</strong>: l'output di un passo temporale (stato nascosto) viene reimmesso come input al passo successivo, permettendo all'informazione di persistere.</p>
<ul>
    <li><strong>Input al Passo t (x(t)):</strong> Elemento attuale della sequenza.</li>
    <li><strong>Stato Nascosto h(t):</strong> Calcolato combinando x(t) e lo stato nascosto precedente h(t-1). Funge da "memoria". <code>h(t) = activation(W_xh * x(t) + W_hh * h(t-1) + b_h)</code>. I pesi <code>W_xh</code> e <code>W_hh</code> sono condivisi per tutti i passi.</li>
    <li><strong>Output al Passo t (y(t)):</strong> (Opzionale) Può essere prodotto ad ogni passo, calcolato da h(t).</li>
</ul>
<p><em>Architetture Comuni:</em> Many-to-One (input sequenza, output singolo), One-to-Many, Many-to-Many (sincrono o asincrono - Encoder-Decoder).</p>
<h3>3. Il Problema della Memoria a Lungo Termine: Vanishing/Exploding Gradients</h3>
<p>Le RNN semplici faticano a catturare dipendenze a lungo raggio (elementi molto distanti nella sequenza) a causa del problema dei gradienti che svaniscono (diventano troppo piccoli) o esplodono (diventano troppo grandi) durante la backpropagation through time (BPTT).</p>
<h3>4. LSTM (Long Short-Term Memory): La Soluzione</h3>
<p>Le <strong>LSTM</strong> (Hochreiter & Schmidhuber, 1997) sono un tipo avanzato di RNN progettate per risolvere il problema della memoria a lungo termine. Introducono:</p>
<ul>
    <li><strong>Stato della Cella (Cell State - c(t)):</strong> Un "nastro trasportatore" che permette all'informazione di fluire quasi linearmente, con poche modifiche.</li>
    <li><strong>Gate (Cancelli):</strong> Meccanismi (strati densi con attivazione sigmoide) che regolano il flusso di informazioni dentro e fuori lo stato della cella.
        <ul>
            <li><em>Forget Gate:</em> Decide cosa dimenticare dallo stato della cella precedente c(t-1).</li>
            <li><em>Input Gate:</em> Decide quali nuove informazioni aggiungere allo stato della cella.</li>
            <li><em>Output Gate:</em> Decide quale parte dello stato della cella aggiornato c(t) far uscire come stato nascosto h(t).</li>
        </ul>
    </li>
</ul>
<p>Grazie ai gate, le LSTM possono imparare a mantenere informazioni rilevanti per lunghi periodi.</p>
<h3>5. GRU (Gated Recurrent Unit): Una Variante Semplificata</h3>
<p>La <strong>GRU</strong> (2014) è una variante dell'LSTM con una struttura interna più semplice (combina stato cella e nascosto, usa solo due gate: Update e Reset). Spesso ottiene prestazioni simili all'LSTM ma con meno parametri.</p>
<h3>6. Implementazione con Keras (<code>tf.keras.layers.LSTM</code>, <code>tf.keras.layers.GRU</code>)</h3>
<p>Keras rende facile usare strati LSTM/GRU. Input atteso: 3D (batch_size, timesteps, features).</p>
<p><em>Parametri Chiave:</em></p>
<ul>
    <li><code>units</code>: Dimensionalità dello stato nascosto/cella.</li>
    <li><code>return_sequences</code>: Boolean. <code>True</code> se lo strato successivo è un altro strato ricorrente o se si necessita output ad ogni timestep. <code>False</code> (default) per restituire solo l'ultimo output.</li>
    <li><code>return_state</code>: Boolean. Per restituire anche gli stati finali (utile per Encoder-Decoder).</li>
</ul>
<p>Spesso usate dopo uno strato <code>Embedding</code> per convertire indici di parole in vettori densi per task NLP.</p>
<h3>7. RNN Bidirezionali (<code>tf.keras.layers.Bidirectional</code>)</h3>
<p>Per alcuni task (es. NLP), il contesto successivo è importante quanto quello precedente. Un wrapper <code>Bidirectional</code> processa la sequenza in entrambe le direzioni (forward e backward) e combina gli output, fornendo una comprensione più ricca.</p>
<p>LSTM e GRU (spesso bidirezionali) sono state fondamentali per molti progressi in NLP e analisi di serie temporali, anche se ora i Transformer stanno guadagnando terreno.</p>` }
            ]
        },
        {
            part: "Parte 6: Esplorando l'IA Generativa e le Applicazioni Pratiche",
            chapters: [
                { id: "ch19", title: "Capitolo 19: Il Mondo dell'IA Generativa", content: `
                    <p>Finora ci siamo concentrati su AI predittiva (classificazione, regressione) o per scoprire pattern (clustering). Ora entriamo nel mondo affascinante dell'<strong>Intelligenza Artificiale Generativa (GenAI)</strong>, capace di creare contenuti nuovi e originali.</p>
<h3>1. Cos'è l'AI Generativa (GenAI)? Creare Contenuti Nuovi</h3>
<p>A differenza dei modelli discriminativi (che distinguono o predicono), i modelli generativi imparano le caratteristiche e la struttura dei dati di addestramento per poi generare nuovi dati artificiali che assomiglino a quelli originali. L'obiettivo è <strong>creare</strong>, non solo classificare o predire.</p>
<h3>2. Categorie Principali e Applicazioni Straordinarie</h3>
<p>La GenAI ha dimostrato capacità impressionanti in diverse modalità:</p>
<ul>
    <li><strong>Testo (Text-to-Text):</strong> Modelli: Large Language Models (LLM) come GPT, Gemini, Llama, Claude. Capacità: generare articoli, storie, codice, rispondere a domande, tradurre, riassumere. App: Chatbot, creazione contenuti, coding.</li>
    <li><strong>Immagini (Text-to-Image, Image-to-Image):</strong> Modelli: Diffusion Models (Stable Diffusion, DALL-E, Midjourney), GAN (storicamente). Capacità: creare immagini da testo, modificare immagini, super-resolution. App: Design, arte, marketing.</li>
    <li><strong>Audio e Musica (Text-to-Audio/Music/Speech):</strong> Modelli: Basati su Transformer o Diffusion (Suno, Udio, ElevenLabs). Capacità: generare musica, effetti sonori, voci realistiche (TTS), clonare voci. App: Produzione musicale, doppiaggio, accessibilità.</li>
    <li><strong>Video (Text-to-Video, Image-to-Video):</strong> Modelli: Complessi, basati su Diffusion/Transformer (Sora, Veo, Pika). Capacità: generare brevi clip video da testo/immagini. App: Prototipazione video, effetti speciali. (Campo in rapida evoluzione).</li>
    <li><strong>Codice (Text-to-Code):</strong> Modelli: LLM specializzati (GitHub Copilot, Code Llama). Capacità: generare/completare codice, tradurre tra linguaggi. App: Sviluppo software.</li>
</ul>
<h3>3. Introduzione ai Modelli Linguistici di Grande Scala (LLM)</h3>
<p>Gli <strong>LLM</strong> sono il motore di molta GenAI testuale. Sono reti neurali (spesso Transformer) addestrate su enormi quantità di dati testuali per predire la parola successiva in una sequenza.</p>
<ul>
    <li><strong>Scala Enorme:</strong> "Large" si riferisce a miliardi/trilioni di parametri e terabyte/petabyte di dati di training.</li>
    <li><strong>Capacità Emergenti:</strong> Dalla scala emergono abilità non esplicitamente programmate (ragionamento limitato, comprensione contesto, traduzione).</li>
    <li><strong>Pre-training e Fine-tuning:</strong> Pre-training su dati generici, seguito da fine-tuning su dati specifici per un task o stile. RLHF (Reinforcement Learning from Human Feedback) è usato per allineare il modello.</li>
</ul>
<h3>4. L'Architettura Transformer: Il Motore della GenAI Moderna</h3>
<p>Introdotta nel paper "Attention Is All You Need" (2017), l'architettura <strong>Transformer</strong> ha rivoluzionato l'elaborazione delle sequenze, superando le RNN/LSTM per molti task.</p>
<ul>
    <li><strong>Problema delle RNN:</strong> Difficoltà con dipendenze a lungo raggio e parallelizzazione.</li>
    <li><strong>Idea Chiave: Meccanismo di Attenzione (Self-Attention):</strong> Permette al modello di pesare l'importanza di tutte le altre parole nella sequenza quando elabora una parola specifica, catturando il contesto globale.</li>
    <li><strong>Vantaggi:</strong> Cattura dipendenze a lungo raggio, altamente parallelizzabile (efficiente su GPU/TPU).</li>
    <li><strong>Architettura:</strong> Embedding, Positional Encoding, Stack di blocchi Encoder (Self-Attention, Feed-Forward) e/o Decoder. Molti LLM moderni (come GPT) usano solo l'architettura Decoder ("Decoder-Only").</li>
</ul>
<h3>5. AI Multimodale: Oltre il Singolo Formato</h3>
<p>Tendenza chiave: modelli capaci di comprendere e generare contenuti attraverso <strong>diverse modalità</strong> (testo, immagini, audio, video) simultaneamente.</p>
<p><em>Esempi:</em> DALL-E (testo -> immagine), GPT-4V (testo+immagine -> testo), Gemini.</p>
<p><em>Come funziona (concettualmente):</em> Tecniche per "allineare" rappresentazioni da diverse modalità in uno spazio latente comune, spesso usando meccanismi di cross-attention.</p>
<p><em>Potenzialità:</em> Applicazioni AI più ricche e integrate, simili all'interazione umana col mondo.</p>` },
                { id: "ch20", title: "Capitolo 20: Interfacce Conversazionali e LLM Principali", content: `
                    <p>Dopo aver introdotto gli LLM, vediamo come interagire con i modelli più noti e l'arte di comunicare efficacemente con loro: il Prompt Engineering.</p>
<h3>1. Panoramica dei Principali Modelli e Interfacce (Focus su Maggio 2025)</h3>
<ul>
    <li><strong>ChatGPT (OpenAI):</strong> Basato su GPT (GPT-3.5, GPT-4, GPT-4o). Interfaccia web (chat.openai.com) e API. Versioni gratuite e a pagamento. <em>Punti di Forza:</em> Versatilità, ragionamento (GPT-4/4o), generazione creativa, ecosistema plugin/GPTs, multimodalità (GPT-4o).</li>
    <li><strong>Gemini (Google):</strong> Famiglia di modelli (Pro, Ultra/Advanced, Flash). Interfaccia web (gemini.google.com), integrato in prodotti Google, API (Google AI Studio, Vertex AI). <em>Punti di Forza:</em> Multimodalità nativa, contesti lunghi (Gemini 1.5 Pro), integrazione ecosistema Google.</li>
    <li><strong>Claude (Anthropic):</strong> Famiglia di modelli (Haiku, Sonnet, Opus). Interfaccia web (claude.ai) e API. Focus su sicurezza, etica ("AI Costituzionale"). <em>Punti di Forza:</em> Prestazioni elevate (Claude 3 Opus), scrittura/ragionamento complessi, gestione lunghi contesti, attenzione all'etica.</li>
    <li><strong>Modelli Open Source (es. Llama, Mixtral):</strong> Comunità vivace (Meta Llama 3, Mistral AI). Interfacce varie: esecuzione locale (Ollama, LM Studio), piattaforme cloud. <em>Punti di Forza:</em> Trasparenza, fine-tuning, esecuzione locale (privacy/costi).</li>
    <li><strong>Altri Servizi (spesso usano API dei modelli sopra):</strong>
        <ul>
            <li><em>Perplexity AI:</em> Ricerca conversazionale con citazioni fonti.</li>
            <li><em>Microsoft Copilot:</em> Integrato in Windows/Edge/365, usa modelli OpenAI e proprietari.</li>
            <li><em>NotebookLM (Google):</em> Strumento di ricerca e ragionamento basato sui tuoi documenti.</li>
        </ul>
    </li>
</ul>
<h3>2. Focus sull'Interazione (GenAI Text-to-Text): Prompt Engineering</h3>
<p>La qualità dell'output di un LLM dipende enormemente dalla qualità dell'input (<strong>prompt</strong>). Il <strong>Prompt Engineering</strong> è l'arte e la scienza di progettare prompt efficaci.</p>
<h4>Principi Chiave del Prompt Engineering:</h4>
<ul>
    <li><strong>Chiarezza e Specificità:</strong> Evita ambiguità. Invece di "Parlami di Python", prova "Spiega i 3 principali vantaggi di Python per la data science a uno sviluppatore junior".</li>
    <li><strong>Fornire Contesto:</strong> Dai al modello le informazioni necessarie (testo da riassumere, stile da imitare).</li>
    <li><strong>Definire il Ruolo (Persona):</strong> "Agisci come un critico cinematografico esperto..."</li>
    <li><strong>Specificare il Formato di Output:</strong> "Rispondi con un elenco puntato", "scrivi in JSON", "tono formale".</li>
    <li><strong>Usare Esempi (Few-Shot Prompting):</strong> Fornire 1+ esempi di input/output desiderato (one-shot, few-shot) migliora le prestazioni, specialmente per formati specifici o task complessi. (Zero-shot = nessun esempio).</li>
    <li><strong>Scomposizione del Compito (Chain of Thought):</strong> Per task complessi, scomponili in passaggi. Chiedere al modello di "pensare passo dopo passo" (Chain of Thought - CoT) prima di dare la risposta finale migliora il ragionamento.</li>
    <li><strong>Iterazione e Raffinamento:</strong> Il primo prompt raramente è perfetto. Prova, analizza, modifica.</li>
</ul>
<h3>3. La Finestra di Contesto (Context Window)</h3>
<p>Quantità massima di informazioni (prompt + cronologia + output finora) che il modello può considerare. Misurata in <strong>token</strong> (circa una parola o parte di essa). Superata la finestra, il modello "dimentica" le informazioni più vecchie. I modelli recenti hanno finestre molto più ampie (da pochi K token a 1-2 Milioni di token).</p>
<h3>4. Retrieval-Augmented Generation (RAG)</h3>
<p>Tecnica per migliorare le risposte con dati esterni e ridurre allucinazioni. Invece di interrogare direttamente l'LLM:</p>
<ol>
    <li><strong>Recupero (Retrieval):</strong> Data una domanda, cerca informazioni rilevanti da una base di conoscenza esterna (documenti, database). Spesso usa vector search su embeddings.</li>
    <li><strong>Augmentation:</strong> Le informazioni recuperate vengono aggiunte al prompt originale.</li>
    <li><strong>Generazione (Generation):</strong> Il prompt "aumentato" viene inviato all'LLM, che genera una risposta basata sia sulla sua conoscenza interna sia sulle informazioni specifiche fornite.</li>
</ol>
<p><em>Vantaggi:</em> Risposte basate su fonti specifiche/aggiornate, riduzione allucinazioni, citazione fonti.</p>
<h3>5. Esecuzione Locale di LLM (es. Ollama)</h3>
<p>Possibilità di eseguire LLM open source sul proprio computer per privacy, costi, personalizzazione. Strumenti come <strong>Ollama</strong> e <strong>LM Studio</strong> semplificano download, gestione ed esecuzione.</p>
<p><em>Vantaggi:</em> Privacy, no costi API, offline, fine-tuning. <em>Svantaggi:</em> Requisiti hardware, prestazioni potenzialmente inferiori ai modelli cloud più grandi, setup tecnico.</p>
<h3>6. Valutazione degli LLM: Cenni sui Benchmark</h3>
<p>Misurare la "bravura" di un LLM è complesso. Si usano <strong>benchmark</strong> (set standardizzati di compiti/domande) per valutare abilità diverse.</p>
<p><em>Esempi:</em> MMLU (conoscenza generale), HellaSwag (senso comune), GSM8K (matematica), HumanEval (codice). Esistono leaderboard (Chatbot Arena, Hugging Face) che confrontano i modelli.</p>` },
                { id: "ch21", title: "Capitolo 21: Un Universo di Strumenti Generativi", content: `
                    <p>Oltre agli LLM per testo, l'IA Generativa offre strumenti per creare una vasta gamma di contenuti multimediali. Questo capitolo esplora alcune categorie chiave.</p>
<h3>1. Dalla Parola all'Immagine (GenAI Text-to-Image)</h3>
<p>Applicazione popolare che genera immagini da descrizioni testuali (prompt).</p>
<ul>
    <li><strong>Tecnologia Sottostante:</strong> Principalmente Diffusion Models.</li>
    <li><strong>Strumenti Popolari:</strong>
        <ul>
            <li><em>Midjourney:</em> Noto per immagini artistiche e stilizzate (accesso via Discord, abbonamento).</li>
            <li><em>Stable Diffusion:</em> Potente modello open source, molte interfacce (Automatic1111, ComfyUI per locali; DreamStudio online). Grande flessibilità (fine-tuning, LoRA).</li>
            <li><em>DALL-E 3 (OpenAI):</em> Integrato in ChatGPT Plus/Team/Enterprise, API. Buona comprensione prompt, generazione testo in immagini.</li>
            <li><em>Ideogram AI:</em> Bravo a incorporare testo leggibile nelle immagini.</li>
        </ul>
    </li>
    <li><strong>Prompting per Immagini:</strong> Enfasi su descrizione visiva, stile (Van Gogh, fotorealistico), illuminazione, composizione, "negative prompts" (cosa non includere).</li>
</ul>
<h3>2. Dalla Parola al Video (GenAI Text-to-Video)</h3>
<p>Campo all'avanguardia, in rapida evoluzione, per generare brevi clip video da testo.</p>
<ul>
    <li><strong>Sfide:</strong> Coerenza temporale, comprensione fisica, enormi risorse computazionali.</li>
    <li><strong>Strumenti Emergenti:</strong>
        <ul>
            <li><em>Sora (OpenAI):</em> Annunciato con capacità impressionanti (video fino a 1 min), accesso limitato.</li>
            <li><em>Runway (Gen-1, Gen-2):</em> Piattaforma creativa AI con strumenti text-to-video, image-to-video.</li>
            <li><em>Pika:</em> Altro strumento popolare per generare/modificare video da prompt.</li>
            <li><em>Google Veo:</em> Modello competitivo con Sora, accesso limitato.</li>
        </ul>
    </li>
</ul>
<h3>3. Dalla Parola al Suono (GenAI Text-to-Audio)</h3>
<ul>
    <li><strong>Text-to-Speech (TTS):</strong> Sintesi vocale realistica. <em>Strumenti:</em> ElevenLabs (voci realistiche, clonazione), Google Cloud TTS, Azure TTS.</li>
    <li><strong>Speech-to-Text (STT):</strong> Trascrizione audio in testo. <em>Strumenti:</em> Whisper (OpenAI), Google Speech-to-Text. (Più discriminativo che generativo, ma fondamentale).</li>
    <li><strong>Text-to-Music:</strong> Generare musica da descrizioni (stile, genere, mood). <em>Strumenti:</em> Suno AI, Udio (canzoni complete con voce), Google MusicLM, Meta AudioCraft.</li>
    <li><strong>Text-to-Voice / Voice Cloning:</strong> Generare audio con caratteristiche di una voce specifica. <em>Strumenti:</em> ElevenLabs. <strong>!!! Nota Etica Cruciale:</strong> Rischio deepfake, impersonazione. Usare responsabilmente e con consenso.</li>
</ul>
<h3>4. Dalla Parola alle Presentazioni (GenAI Text-to-Slide)</h3>
<p>Automatizzare la creazione di slide. L'utente fornisce argomento/testo, l'AI genera slide con testo, immagini (spesso AI-generated), layout.</p>
<p><em>Strumenti:</em> Gamma.app, Tome, Microsoft Copilot in PowerPoint, Presentations.ai.</p>
<h3>5. Dalla Parola al Web (GenAI Text-to-Website)</h3>
<p>Automatizzare la creazione di siti web o componenti frontend.</p>
<p><em>Strumenti:</em> Vercel v0.dev (codice React da prompt/immagini), Wix ADI, Jimdo Dolphin (siti iniziali basati su AI).</p>
<h3>6. GenAI Notetaker & Assistenti per la Produttività</h3>
<p>Sfruttare AI per gestire sovraccarico informativo da riunioni/registrazioni.</p>
<p><em>Concetto:</em> Registrano/trascrivono audio, usano LLM per riassumere, identificare punti chiave, estrarre azioni.</p>
<p><em>Strumenti:</em> Otter.ai, Fireflies.ai, Fathom, tl;dv, funzionalità integrate in Zoom/Meet/Teams.</p>
<p>Questi strumenti stanno trasformando processi creativi e di produttività in molti settori.</p>` },
                { id: "ch22", title: "Capitolo 22: Intelligenza Artificiale per le Risorse Umane (HR & AI)", content: `
                    <p>L'AI sta trasformando le Risorse Umane (HR), spostando il focus da compiti amministrativi a un ruolo strategico nella gestione e sviluppo del talento. Questo capitolo illustra un ciclo virtuoso in cui l'AI aiuta a definire, misurare e formare competenze, integrandosi con i Learning Management Systems (LMS).</p>
<h3>1. Introduzione: La Trasformazione AI dell'HR</h3>
<p>Tradizionalmente, HR si occupa di reclutamento, paghe, conformità. L'aspettativa moderna è che HR sia un partner strategico per attrarre, sviluppare e trattenere talenti. L'AI può automatizzare processi, migliorare l'esperienza candidato/dipendente e fornire insight per decisioni strategiche, specialmente nella gestione delle competenze.</p>
<h3>Fase 1: Ridefinire le Competenze per Ruolo con l'AI</h3>
<p>Le job description tradizionali diventano obsolete rapidamente. L'AI (specie NLP) può analizzare grandi quantità di dati testuali per identificare competenze chiave in modo dinamico:</p>
<ul>
    <li><strong>Analisi Mercato Esterno:</strong> Scansionare annunci di lavoro online per ruoli simili per identificare competenze richieste ed emergenti.</li>
    <li><strong>Analisi Interna:</strong> Elaborare job description esistenti, performance review, obiettivi di progetto, feedback per capire competenze correlate al successo interno.</li>
    <li><strong>Tecniche AI:</strong> Named Entity Recognition (NER) per estrarre "competenze", analisi frequenza/co-occorrenza, topic modeling.</li>
</ul>
<p><em>Output:</em> Profili di ruolo dinamici, basati su dati, con competenze chiave e loro rilevanza.</p>
<h3>Fase 2: Misurare e Pesare le Competenze con l'AI</h3>
<p>Una volta definite le competenze, la sfida è valutarne il possesso attuale da parte dei dipendenti.</p>
<ul>
    <li><strong>Analisi Dati Esistenti (NLP):</strong> Estrarre menzioni di competenze da performance review, feedback 360°, descrizioni progetti.</li>
    <li><strong>Inferenza da Attività Formative:</strong> Collegare completamento corsi/certificazioni (tracciati da LMS) a competenze.</li>
    <li><strong>Assessment Basati su AI (Potenziale):</strong> Simulazioni o test adattivi per valutare competenze specifiche oggettivamente.</li>
    <li><strong>Pesatura Competenze:</strong> L'AI può aiutare a definire l'importanza relativa (peso) di ogni competenza per un ruolo, analizzando correlazioni con performance elevate o allineamento con obiettivi strategici.</li>
</ul>
<p><em>Considerazioni Etiche:</em> Bias algoritmico, trasparenza, validazione umana sono cruciali.</p>
<h3>Fase 3: Formare le Competenze con Percorsi Personalizzati Guidati dall'AI (Integrazione LMS)</h3>
<p>Identificati i gap (competenze target vs attuali), l'AI suggerisce come colmarli, integrandosi con l'LMS.</p>
<ul>
    <li><strong>Sistema di Raccomandazione AI:</strong> Basandosi su gap, ruolo, preferenze, storico formativo, l'AI interroga il catalogo LMS (e fonti esterne) per suggerire corsi, moduli, articoli, video, mentorship, progetti.</li>
    <li><strong>Percorsi Dinamici (IDP):</strong> L'AI assembla raccomandazioni in un piano di sviluppo personalizzato e adattivo.</li>
    <li><strong>Integrazione con LMS:</strong> L'AI comunica con l'LMS per assegnare corsi, tracciare progressi. Il feedback dall'LMS (corsi completati) aggiorna il profilo di competenze dell'AI, chiudendo il ciclo.</li>
</ul>
<h3>Il Ciclo Virtuoso HR-AI-LMS</h3>
<p>Definizione (AI) → Misurazione (AI support) → Identificazione Gap → Raccomandazione Formativa (AI) → Assegnazione/Fruizione (LMS) → Sviluppo Competenze → Aggiornamento Dati (Feedback a AI) → (Ritorno a Misurazione).</p>
<p><em>Benefici:</em> Sviluppo personale mirato, engagement, allineamento strategico, mobilità interna, decisioni HR data-driven.</p>
<h3>Considerazioni Implementative e Sfide</h3>
<p>Qualità/Integrazione Dati, Tecnologia AI/LMS, Privacy/GDPR, Etica/Bias, Change Management.</p>
<p>L'AI può rendere l'HR un partner strategico più efficace nella valorizzazione del capitale umano.</p>` },
                { id: "ch23", title: "Capitolo 23: AI Agentiva e Strumenti di Supporto allo Sviluppo", content: `
                    <p>Esploriamo sistemi AI che non solo elaborano o generano, ma pianificano, agiscono e usano strumenti per raggiungere obiettivi complessi: l'AI Agentiva. Vediamo anche strumenti AI specifici per lo sviluppo software e l'automazione workflow.</p>
<h3>1. Concetto di Agenti AI Autonomi (Autonomous AI Agents)</h3>
<p>Un <strong>Agente AI Autonomo</strong> percepisce l'ambiente, ragiona su obiettivi, pianifica azioni, usa strumenti (API, web), e adatta il piano in base ai risultati, con intervento umano minimo/nullo post-setup.</p>
<ul>
    <li><strong>Caratteristiche:</strong> Obiettivo definito, Pianificazione, Uso Strumenti (Tool Use), Memoria (breve/lungo termine), Autonomia (limitata).</li>
    <li><strong>Distinzione da Chatbot:</strong> Agisce per compiti, non solo risponde.</li>
    <li><strong>Stato Attuale:</strong> Campo attivo (Auto-GPT, BabyAGI, LangChain/LlamaIndex agents). Sfide: robustezza, sicurezza, controllo, costo per compiti complessi reali.</li>
</ul>
<h3>2. Coding Assistito dall'IA (AI Coding Tools)</h3>
<p>LLM addestrati su codice che assistono gli sviluppatori.</p>
<ul>
    <li><strong>Funzionalità:</strong> Completamento codice intelligente, Text-to-Code (generazione da prompt), spiegazione codice, refactoring, debugging, scrittura test, traduzione linguaggi.</li>
    <li><strong>Strumenti Principali:</strong>
        <ul>
            <li><em>GitHub Copilot:</em> (OpenAI/GitHub) Integrato in editor, suggerimenti in tempo reale, chat.</li>
            <li><em>Amazon Q Developer (ex CodeWhisperer):</em> Offerta AWS, focus sicurezza/integrazione AWS.</li>
            <li><em>Cursor AI:</em> Editor "AI-first" basato su VS Code, integra LLM (GPT-4, Claude).</li>
            <li><em>Kodium, Replit AI, Tabnine, Codeium.</em></li>
        </ul>
    </li>
    <li><strong>L'Approccio "Vibe Coding":</strong> Sviluppatore esprime intento ("vibe") in linguaggio naturale, AI genera/modifica codice. Sviluppatore agisce come revisore/orchestratore.</li>
    <li><strong>Vantaggi:</strong> Produttività, apprendimento, riduzione errori banali.</li>
    <li><strong>Rischi/Considerazioni:</strong> Qualità/correttezza codice AI (necessaria revisione), sicurezza (vulnerabilità), originalità/licenze, over-reliance (perdita skill).</li>
</ul>
<h3>3. Automazione dei Workflow con Strumenti AI-Powered</h3>
<p>L'AI si integra in piattaforme di automazione workflow per collegare app e servizi.</p>
<ul>
    <li><strong>Piattaforme No-Code/Low-Code:</strong> Zapier, Make, IFTTT iniziano a integrare AI per suggerire/eseguire azioni.</li>
    <li><strong>Strumenti per Sviluppatori/Tecnici:</strong>
        <ul>
            <li><em>n8n (n8n.io):</em> Piattaforma potente (self-hosted open-source o cloud) per automazione workflow con interfaccia visuale a nodi. Facile integrare nodi LLM (OpenAI, Anthropic) per elaborazione linguaggio, classificazione, generazione contenuto all'interno di flussi.</li>
        </ul>
    </li>
    <li><em>Esempi Workflow Intelligenti:</em> Ricevere email cliente → LLM estrae intento/info → Apre ticket supporto → LLM genera bozza risposta.</li>
</ul>
<h3>4. Applicazioni Commerciali Potenziali Agenti AI (Breve Menzione)</h3>
<p>Assistenti personali/esecutivi potenziati, automazione servizio clienti avanzata, ricerca/analisi automatizzata, gestione campagne marketing, nuovi servizi basati su agenti.</p>
<p>Questi sviluppi mostrano l'AI come partner attivo, ma richiedono uso critico e supervisione umana.</p>` },
                { id: "ch24", title: "Capitolo 24: Salvare, Caricare e Distribuire (Deployment) i Modelli", content: `
                    <p>Abbiamo costruito, addestrato, valutato e ottimizzato modelli. Ma a cosa servono se rimangono nel nostro script? L'obiettivo finale è renderli utilizzabili per fare previsioni su nuovi dati nel mondo reale: il <strong>Deployment</strong>.</p>
<h3>1. Salvare e Caricare Modelli Scikit-learn</h3>
<p>Per modelli Scikit-learn, si usa la <strong>serializzazione</strong> (convertire oggetto in flusso di byte).</p>
<ul>
    <li><strong>Metodo Principale: <code>joblib</code> (Preferito per Scikit-learn):</strong> Più efficiente con grandi array NumPy.
        <ul>
            <li><em>Salvataggio:</em> <code>import joblib; joblib.dump(model, 'nome_modello.joblib')</code></li>
            <li><em>Caricamento:</em> <code>loaded_model = joblib.load('nome_modello.joblib')</code></li>
        </ul>
    </li>
    <li><strong>Alternativa: <code>pickle</code>:</strong> Modulo standard Python. Sintassi simile.</li>
    <li><strong>Importante:</strong> Salvare anche oggetti di preprocessing (scaler, encoder) adattati sul training set! Vanno applicati ai nuovi dati prima della predizione.</li>
    <li><em>Sicurezza/Versione:</em> Non caricare file da fonti non fidate. Salvare versioni librerie per riproducibilità.</li>
</ul>
<h3>2. Salvare e Caricare Modelli Keras/TensorFlow</h3>
<p>Keras/TF offrono metodi integrati per salvare/caricare interi modelli (architettura, pesi, stato ottimizzatore) o solo i pesi.</p>
<ul>
    <li><strong>Metodo Principale:</strong> <code>model.save()</code> e <code>tf.keras.models.load_model()</code>.</li>
    <li><strong>Formato Keras Nativo (<code>.keras</code>):</strong> Raccomandato e predefinito (da TF 2.x). Salva tutto in un file zip efficiente.</li>
    <li><strong>Formato TensorFlow SavedModel (Directory):</strong> Standard per deployment con TF Serving e altri strumenti TF. Salva modello come directory.</li>
    <li><strong>Formato HDF5 (<code>.h5</code> - Legacy):</strong> Usato da Keras standalone. Ancora supportato.</li>
    <li><strong>Salvare/Caricare Solo Pesi:</strong> <code>model.save_weights()</code> e <code>model.load_weights()</code>. Richiede di ricreare prima un modello con la stessa architettura.</li>
</ul>
<h3>3. Fare Previsioni con i Modelli Caricati</h3>
<p>Una volta caricato, il modello si usa con <code>.predict()</code> (o <code>.predict_proba()</code>). <strong>Cruciale:</strong> preprocessare i nuovi dati ESATTAMENTE come i dati di training (stesso ordine feature, stesso scaler/encoder adattato sul training).</p>
<h3>4. Strategie di Deployment (Panoramica)</h3>
<p>Rendere il modello accessibile. Argomento vasto (MLOps), alcune strategie:</p>
<ul>
    <li><strong>Integrazione Diretta nell'Applicazione:</strong> Caricare modello/preprocessing in script/app desktop. Semplice per prototipi. Limiti: accoppiamento, scalabilità, aggiornamenti.</li>
    <li><strong>API REST (Web Service):</strong> Approccio comune. Servizio web (Flask, FastAPI) espone endpoint API (es. POST /predict). Riceve dati (JSON), preprocessa, predice, restituisce risultato (JSON). Decoupling, scalabile. Richiede gestione infrastruttura.
        <ul><li><em>Containerizzazione (Docker):</em> Impacchettare API e dipendenze in container Docker per facile deployment.</li></ul>
    </li>
    <li><strong>Funzioni Serverless (FaaS):</strong> Logica di predizione in una funzione (AWS Lambda, Google Cloud Functions). Scalabilità automatica, pay-per-use. Limiti su dimensioni/dipendenze/tempo.</li>
    <li><strong>Piattaforme Cloud ML Gestite:</strong> Google Vertex AI Endpoints, AWS SageMaker Endpoints, Azure ML Endpoints. Gestiscono infrastruttura, versionamento, scaling, monitoring. Vendor lock-in, costo.</li>
    <li><strong>Deployment su Dispositivi Edge (Mobile/IoT):</strong> Per inferenza locale (bassa latenza, offline, privacy). Richiede conversione modello in formato leggero (TF Lite, ONNX).</li>
    <li><strong>Predizioni Batch:</strong> Script periodico che carica modello e predice su grandi batch di dati (da DB/data warehouse). Risultati salvati.</li>
</ul>
<h3>5. Considerazioni Chiave per il Deployment</h3>
<p>Prestazioni (latenza, throughput), Scalabilità, Costo, Monitoraggio (performance modello, drift dati/concetti), Manutenzione (aggiornamenti), Sicurezza.</p>
<p>Mettere un modello in produzione è spesso tanto lavoro quanto addestrarlo.</p>` }
            ]
        },
        {
            part: "Parte 7: Conclusione e Prossimi Passi",
            chapters: [
                { id: "ch25", title: "Capitolo 25: Ricapitolazione e Tendenze Future dell'AI", content: `
                    <p>Siamo giunti alla fine del nostro viaggio introduttivo nell'Intelligenza Artificiale. Abbiamo coperto un vasto panorama, dai concetti fondamentali alle applicazioni più recenti. È il momento di ricapitolare e guardare al futuro.</p>
<h3>1. Ricapitolazione dei Concetti Chiave Appresi</h3>
<p>Abbiamo esplorato:</p>
<ul>
    <li><strong>Fondamenti dell'AI:</strong> Definizioni (AI, ML, DL), storia, tipi di AI (ANI, AGI, ASI), etica e impatto (bias, fairness, XAI, privacy).</li>
    <li><strong>Strumenti Essenziali:</strong> Python, Anaconda, Jupyter, librerie (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).</li>
    <li><strong>Paradigmi del ML:</strong> Apprendimento Supervisionato (Classificazione, Regressione), Non Supervisionato (Clustering, Riduzione Dimensionalità), Per Rinforzo.</li>
    <li><strong>Pipeline di Sviluppo ML:</strong> Preprocessing, costruzione/addestramento modello, valutazione (metriche, CV), miglioramento (tuning iperparametri, feature engineering).</li>
    <li><strong>Introduzione al Deep Learning:</strong> Reti Neurali (Perceptron, ANN, attivazioni), framework (TensorFlow, Keras), CNN (immagini), RNN/LSTM (sequenze).</li>
    <li><strong>IA Generativa e Applicazioni Pratiche:</strong> LLM, Transformer, Prompt Engineering, RAG, Text-to-Image/Video/Audio, AI per HR, Agenti AI, Deployment.</li>
</ul>
<h3>2. Direzioni Future della Ricerca e Sviluppo in AI</h3>
<p>Il campo dell'AI è in continua e rapida evoluzione. Alcune tendenze e direzioni promettenti:</p>
<ul>
    <li><strong>Modelli Sempre Più Grandi e Capaci (Foundation Models):</strong> LLM e modelli multimodali continueranno a crescere in scala e capacità, ma con attenzione a sostenibilità e rischi.</li>
    <li><strong>Miglioramento del Ragionamento e Pianificazione:</strong> Superare i limiti attuali degli LLM nel ragionamento logico e causale. L'AI Agentiva è un passo in questa direzione.</li>
    <li><strong>Efficienza e Ottimizzazione:</strong> Sviluppo di architetture più efficienti (MoE), tecniche di quantizzazione/pruning per rendere i modelli eseguibili su hardware meno potente (edge AI).</li>
    <li><strong>Multimodalità Reale:</strong> Integrazione fluida di diverse modalità (testo, immagine, audio, video) sia in input che in output.</li>
    <li><strong>Apprendimento Continuo e Adattamento:</strong> Modelli capaci di imparare da nuovi dati post-deployment senza riaddestramento completo e senza "dimenticare" catastroficamente.</li>
    <li><strong>AI Spiegabile e Interpretabile (Explainable AI - XAI):</strong> Tecniche robuste per capire <em>perché</em> un modello AI prende una certa decisione, cruciale per sistemi critici e fiducia.</li>
    <li><strong>Robustezza e Sicurezza:</strong> Rendere i modelli più resistenti ad attacchi avversari, dati rumorosi e out-of-distribution generalization.</li>
    <li><strong>Etica, Equità e Allineamento (AI Alignment):</strong> Mitigare bias, garantire equità, e assicurare che sistemi AI (specialmente futuri AGI) agiscano in linea con valori e intenti umani (l'"AI Alignment Problem").</li>
    <li><strong>Interazione Uomo-AI (Human-AI Collaboration):</strong> Progettare sistemi AI che potenzino le capacità umane, non solo che le sostituiscano.</li>
    <li><strong>AI Incarnata (Embodied AI):</strong> Integrare AI in agenti fisici (robot) che imparano interagendo con il mondo reale.</li>
    <li><strong>AI e Scienza:</strong> Uso crescente dell'AI per accelerare scoperte in biologia (AlphaFold), medicina, materiali, climatologia.</li>
    <li><strong>AI Quantistica (Quantum AI):</strong> Esplorazione delle potenzialità dell'informatica quantistica per ML (campo iniziale).</li>
    <li><strong>Intelligenza Artificiale Generale (AGI):</strong> Rimane l'obiettivo a lungo termine (e controverso) di creare AI con intelligenza flessibile e paragonabile a quella umana.</li>
</ul>
<p>Il futuro dell'AI promette di essere tanto entusiasmante quanto ricco di sfide.</p>` },
                { id: "ch26", title: "Capitolo 26: Come Continuare il Tuo Percorso nell'AI", content: `
                    <p>Congratulazioni per aver completato questo ebook! Speriamo ti abbia fornito una solida base. Ma l'AI è un campo vasto e in continua evoluzione. Questo è solo l'inizio del tuo viaggio.</p>
<h3>1. Una Possibile AI Roadmap Personale</h3>
<p>Ogni percorso è personale, ma ecco una possibile sequenza logica per approfondire:</p>
<ol>
    <li><strong>Solidifica le Basi:</strong>
        <ul>
            <li><em>Pratica Python:</em> Se non hai basi solide, esercitati con strutture dati, funzioni, classi.</li>
            <li><em>Padroneggia le Librerie Chiave:</em> Diventa abile con NumPy, Pandas, Matplotlib/Seaborn. La preparazione dati è cruciale!</li>
            <li><em>Approfondisci Scikit-learn:</em> Esplora più a fondo algoritmi, iperparametri, pipeline, preprocessing avanzato.</li>
        </ul>
    </li>
    <li><strong>Approfondisci il Machine Learning:</strong>
        <ul>
            <li><em>Metodi d'Insieme (Ensemble):</em> Studia Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), Stacking.</li>
            <li><em>Valutazione Avanzata:</em> Cross-Validation, metriche per dati sbilanciati (Precision-Recall Curve), confronto statistico modelli.</li>
            <li><em>Algoritmi Meno Comuni:</em> Naive Bayes, DBSCAN, Clustering Gerarchico, GMM.</li>
        </ul>
    </li>
    <li><strong>Immergiti nel Deep Learning:</strong>
        <ul>
            <li><em>Basi Matematiche (Opzionale ma utile):</em> Algebra lineare, calcolo differenziale (backpropagation), probabilità/statistica.</li>
            <li><em>Framework (Keras/TF o PyTorch):</em> Scegline uno e approfondiscilo (API Funzionale, strati custom, ottimizzazione training).</li>
            <li><em>Architetture Avanzate:</em> Studia più in dettaglio CNN (ResNet, Inception), RNN/LSTM/GRU (applicazioni), e soprattutto Transformer e meccanismo di Attenzione.</li>
            <li><em>Modelli Generativi:</em> GAN, VAE, Diffusion Models.</li>
        </ul>
    </li>
    <li><strong>Scegli una Specializzazione (Opzionale ma Consigliato):</strong>
        <ul>
            <li><em>Natural Language Processing (NLP):</em> Embedding, RNN/LSTM, Transformer, LLM. (Librerie: NLTK, spaCy, Hugging Face Transformers).</li>
            <li><em>Computer Vision (CV):</em> CNN avanzate, Vision Transformers. (Librerie: OpenCV, Pillow, torchvision).</li>
            <li><em>Reinforcement Learning (RL):</em> (Librerie: Stable Baselines3, TF-Agents).</li>
            <li><em>Serie Temporali:</em> ARIMA, Prophet, RNN/LSTM.</li>
            <li><em>MLOps:</em> Ciclo di vita completo del modello (deployment, monitoring).</li>
            <li><em>AI Ethics & Responsibility.</em></li>
        </ul>
    </li>
    <li><strong>Rimani Aggiornato:</strong> Segui novità, blog, partecipa a discussioni.</li>
</ol>
<h3>2. Risorse Consigliate per Continuare l'Apprendimento</h3>
<ul>
    <li><strong>Corsi Online:</strong> Coursera/DeepLearning.AI (Andrew Ng), fast.ai, edX (MIT, Stanford), Udacity, Kaggle Learn.</li>
    <li><strong>Libri:</strong> "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" (Géron) - ottimo passo successivo; "Deep Learning" (Goodfellow, Bengio, Courville) - riferimento teorico; "Python for Data Analysis" (McKinney).</li>
    <li><strong>Comunità e Piattaforme:</strong> Kaggle (dataset, competizioni, notebook), Hugging Face Hub (modelli, dataset, librerie), Stack Overflow, Reddit (r/MachineLearning, r/deeplearning, r/LocalLLaMA).</li>
    <li><strong>Blog, Newsletter, Ricercatori:</strong> Blog OpenAI, Google AI, Meta AI; Andrej Karpathy, Colah's Blog, Jay Alammar; Newsletter The Batch, Import AI.</li>
    <li><strong>Paper Accademici:</strong> arXiv (cs.LG, cs.AI, cs.CL, cs.CV), Papers with Code, conferenze (NeurIPS, ICML, ICLR, CVPR, ACL).</li>
</ul>
<h3>3. L'Importanza Cruciale dei Progetti Personali</h3>
<p>Leggere e studiare è fondamentale, ma la pratica hands-on è insostituibile. Lavorare su progetti personali è il modo migliore per consolidare, affrontare problemi reali, costruire un portfolio e imparare il debugging.</p>
<p><em>Come Iniziare:</em> Replica tutorial, partecipa a competizioni Kaggle, scegli un dataset pubblico di tuo interesse e definisci un obiettivo, applica l'AI a un tuo interesse/hobby, contribuisci a open source, costruisci una semplice applicazione web con un tuo modello.</p>
<h3>4. L'Importanza dell'Apprendimento Continuo</h3>
<p>L'AI cambia rapidamente. Nuovi modelli, tecniche, strumenti emergono continuamente. Essere un professionista o un appassionato informato richiede un impegno all'apprendimento continuo.</p>
<ul>
    <li><strong>Sii Curioso/a:</strong> Chiediti "come funziona?", "si può fare meglio?".</li>
    <li><strong>Leggi Regolarmente:</strong> Dedica tempo a leggere articoli, blog, paper.</li>
    <li><strong>Sperimenta:</strong> Prova nuovi strumenti, librerie, tecniche.</li>
    <li><strong>Segui la Comunità:</strong> Partecipa a discussioni, fai domande, condividi.</li>
</ul>
<h3>5. Un Ultimo Messaggio</h3>
<p>Il viaggio nell'AI può sembrare impegnativo, ma è incredibilmente gratificante e pieno di opportunità. Hai fatto il primo passo. Ti incoraggiamo a continuare a esplorare, sperimentare e costruire. Non aver paura di sbagliare: ogni errore è un'opportunità. Trova l'area dell'AI che ti appassiona di più e approfondiscila.</p>
<p><em>L'Intelligenza Artificiale non è qui per sostituirci, ma per potenziarci. Non sarà l’AI a rubarci il lavoro, ma la persona che è in grado di padroneggiarla nel suo quotidiano. La vera magia accade quando la nostra creatività, la nostra intuizione e i nostri valori si uniscono alla sua incredibile capacità di elaborare e apprendere, aprendo porte a un futuro che possiamo solo iniziare a immaginare.</em></p>` }
            ]
        }
    ];
        const glossaryData = {
        "AI (Intelligenza Artificiale)": "Branca dell'informatica che mira a creare sistemi capaci di svolgere compiti che richiedono intelligenza umana (ragionamento, apprendimento, percezione, problem solving).",
        "AI Ristretta (Narrow AI / ANI)": "L'unica forma di AI esistente oggi, specializzata in un compito specifico o in un insieme limitato di compiti (es. riconoscimento vocale, filtri anti-spam).",
        "AI Generale (AGI)": "Intelligenza Artificiale ipotetica con capacità cognitive paragonabili a quelle umane in tutti i domini intellettuali.",
        "Superintelligenza (ASI)": "Intelligenza Artificiale ipotetica che supererebbe significativamente le capacità cognitive umane in quasi tutti i domini.",
        "Machine Learning (ML)": "Sottoinsieme dell'AI in cui i sistemi imparano dai dati e migliorano le loro prestazioni su un compito specifico senza essere esplicitamente programmati per quel compito.",
        "Deep Learning (DL)": "Sottoinsieme del ML basato su Reti Neurali Artificiali con molteplici strati nascosti ('profonde'), particolarmente efficace con dati non strutturati.",
        "Apprendimento Supervisionato": "Paradigma del ML in cui il modello impara da dati etichettati (input associati a output corretti). Obiettivo: predire l'output per nuovi input. Include Classificazione e Regressione.",
        "Apprendimento Non Supervisionato": "Paradigma del ML in cui il modello impara da dati non etichettati. Obiettivo: scoprire strutture, pattern o relazioni nascoste. Include Clustering e Riduzione Dimensionalità.",
        "Apprendimento per Rinforzo (RL)": "Paradigma del ML in cui un agente impara a prendere decisioni interagendo con un ambiente per massimizzare una ricompensa cumulativa.",
        "Dataset": "Una raccolta organizzata di dati utilizzata per addestrare e valutare modelli di ML.",
        "Feature (Attributo)": "Una caratteristica o proprietà misurabile di un'istanza nel dataset, usata come input dal modello.",
        "Etichetta (Label / Target)": "Nell'apprendimento supervisionato, è il valore 'corretto' o l'output desiderato che il modello deve imparare a predire.",
        "Algoritmo (in ML)": "La procedura matematica o computazionale generale che viene utilizzata per imparare dai dati (es. Albero Decisionale, K-Means).",
        "Modello (in ML)": "Il risultato specifico ottenuto dopo aver applicato un algoritmo a un particolare dataset. Contiene i pattern appresi e viene usato per fare previsioni.",
        "Parametri (del Modello)": "Variabili interne del modello che l'algoritmo apprende (stima) direttamente dai dati durante l'addestramento (es. pesi di una rete neurale).",
        "Iperparametri": "Impostazioni di configurazione dell'algoritmo che vengono scelte prima di iniziare il processo di addestramento (es. numero di alberi in un Random Forest, learning rate).",
        "Addestramento (Training / Fitting)": "Il processo di apprendimento vero e proprio, in cui il modello stima i suoi parametri analizzando il training set.",
        "Inferenza (Prediction)": "Il processo di utilizzare un modello già addestrato per fare previsioni su nuovi dati.",
        "Valutazione (Evaluation)": "Il processo per misurare le prestazioni di un modello addestrato, solitamente su un test set, usando metriche specifiche.",
        "Underfitting (Sottoadattamento)": "Problema in cui il modello è troppo semplice per catturare la complessità dei dati, risultando in scarse prestazioni sia su training che su test set.",
        "Overfitting (Sovradattamento)": "Problema in cui il modello impara troppo bene i dati di addestramento (incluso il rumore) e non generalizza bene a nuovi dati. Alte prestazioni su training, scarse su test set.",
        "Bias (in AI)": "Distorsioni o pregiudizi sistematici nei dati o negli algoritmi che possono portare a risultati ingiusti, imprecisi o discriminatori.",
        "Equità (Fairness in AI)": "Concetto che mira a garantire che i sistemi AI non producano risultati ingiustamente discriminatori verso individui o gruppi.",
        "Spiegabilità (Explainability / XAI)": "Capacità di un sistema AI di fornire spiegazioni comprensibili agli umani su come arriva alle sue decisioni o previsioni.",
        "Python": "Linguaggio di programmazione popolare e d'elezione per lo sviluppo AI/ML grazie alla sua semplicità e al vasto ecosistema di librerie.",
        "Anaconda": "Distribuzione open-source di Python (e R) per calcolo scientifico e data science, che semplifica la gestione di pacchetti e ambienti.",
        "Jupyter Notebook/Lab": "Strumenti interattivi basati sul web per creare documenti che combinano codice vivo, visualizzazioni e testo narrativo.",
        "Google Colaboratory (Colab)": "Servizio cloud gratuito di Google che fornisce un ambiente Jupyter Notebook con accesso a GPU/TPU.",
        "NumPy": "Libreria Python fondamentale per il calcolo numerico, introduce l'oggetto array N-dimensionale (ndarray).",
        "Pandas": "Libreria Python per la manipolazione e l'analisi di dati tabulari (strutture Series e DataFrame).",
        "Matplotlib": "Libreria Python base per la creazione di grafici statici, animati e interattivi.",
        "Seaborn": "Libreria di visualizzazione dati Python basata su Matplotlib, per creare grafici statistici più attraenti e informativi.",
        "Scikit-learn (sklearn)": "Libreria Python open-source completa per il Machine Learning classico, con algoritmi, preprocessing, valutazione.",
        "Estimator (in Scikit-learn)": "Interfaccia comune in Scikit-learn per oggetti che imparano dai dati (modelli, preprocessatori), con metodi come .fit(), .predict(), .transform().",
        "Data Preprocessing": "Fase cruciale della pipeline ML che trasforma i dati grezzi in un formato pulito, coerente e adatto all'addestramento (include gestione NaN, scaling, encoding).",
        "Convalida Incrociata (Cross-Validation)": "Tecnica per valutare la capacità di generalizzazione di un modello in modo più robusto, dividendo i dati in K 'fold' e addestrando/validando K volte.",
        "Feature Engineering": "L'arte e la scienza di creare nuove feature o selezionare le più informative dai dati originali per migliorare le prestazioni del modello.",
        "Rete Neurale Artificiale (ANN)": "Modello computazionale ispirato alla struttura del cervello biologico, composto da neuroni artificiali interconnessi organizzati in strati (input, hidden, output).",
        "Perceptron": "Il modello matematico più semplice di un neurone artificiale, blocco costruttivo delle ANN.",
        "Funzione di Attivazione": "Funzione non lineare applicata all'output di un neurone in una ANN (es. ReLU, Sigmoid, Tanh, Softmax), cruciale per permettere alla rete di apprendere relazioni complesse.",
        "Reti Profonde (Deep Networks)": "Reti Neurali Artificiali con molteplici strati nascosti, alla base del Deep Learning.",
        "TensorFlow": "Piattaforma end-to-end open-source per il Machine Learning, con forte focus sul Deep Learning, sviluppata da Google.",
        "Keras": "API di alto livello per costruire e addestrare reti neurali, integrata ufficialmente in TensorFlow (tf.keras), focalizzata su facilità d'uso.",
        "CNN (Convolutional Neural Network)": "Tipo di rete neurale profonda particolarmente efficace per l'analisi di dati a griglia come le immagini, utilizza strati convoluzionali e di pooling.",
        "RNN (Recurrent Neural Network)": "Tipo di rete neurale adatta a elaborare dati sequenziali (es. testo, serie temporali), grazie a connessioni ricorrenti che creano una forma di memoria.",
        "LSTM (Long Short-Term Memory)": "Un tipo avanzato di RNN con una struttura a 'gate' (forget, input, output) che le permette di gestire meglio le dipendenze a lungo termine nei dati sequenziali.",
        "GenAI (IA Generativa)": "Branca dell'AI focalizzata sulla creazione di contenuti nuovi e originali (testo, immagini, audio, video, codice) basandosi su pattern appresi da dati esistenti.",
        "LLM (Large Language Model)": "Modelli di linguaggio di grandi dimensioni, tipicamente basati sull'architettura Transformer, addestrati su enormi quantità di testo (es. GPT, Gemini, Llama, Claude).",
        "Architettura Transformer": "Architettura di rete neurale, introdotta nel paper 'Attention Is All You Need', che ha rivoluzionato l'elaborazione delle sequenze grazie al meccanismo di self-attention. Motore degli LLM.",
        "Prompt Engineering": "L'arte e la scienza di progettare input testuali (prompt) efficaci per guidare un LLM a produrre l'output desiderato.",
        "Finestra di Contesto (Context Window)": "La quantità massima di informazioni (misurata in token) che un LLM può considerare in un dato momento per generare la risposta successiva.",
        "RAG (Retrieval Augmented Generation)": "Tecnica che combina LLM con il recupero di informazioni da fonti esterne per migliorare l'accuratezza, la rilevanza e ridurre le 'allucinazioni' nelle risposte.",
        "AI Agentiva (Agentic AI)": "Sistemi AI capaci di percepire l'ambiente, ragionare su obiettivi, pianificare azioni, utilizzare strumenti (API, web) e adattarsi per raggiungere obiettivi complessi con intervento umano minimo.",
        "Deployment (di Modelli)": "Il processo di rendere un modello ML addestrato accessibile e utilizzabile da utenti finali o altri sistemi per fare previsioni su nuovi dati nel mondo reale."
    };

        const allQuizQuestions = [
        // Fondamenti AI & ML
        { question: "Cosa significa l'acronimo AI?", options: ["Artificial Interaction", "Automated Intelligence", "Artificial Intelligence", "Advanced Interface"], answer: "Artificial Intelligence", category: "Parte 1" }, 
        { question: "Qual è l'obiettivo principale dell'AI Ristretta (Narrow AI)?", options: ["Superare l'intelligenza umana", "Svolgere un compito specifico", "Avere coscienza di sé", "Risolvere qualsiasi problema"], answer: "Svolgere un compito specifico", category: "Parte 1" },
        { question: "Il Machine Learning permette ai computer di:", options: ["Essere programmati per ogni singolo caso", "Imparare dai dati senza programmazione esplicita", "Eseguire solo calcoli matematici", "Navigare in internet autonomamente"], answer: "Imparare dai dati senza programmazione esplicita", category: "Parte 1" },
        { question: "Quale di questi NON è un tipo di apprendimento del Machine Learning?", options: ["Supervisionato", "Non Supervisionato", "Per Rinforzo", "Automatico Diretto"], answer: "Automatico Diretto", category: "Parte 1" },
        { question: "Cosa rappresenta un 'Dataset' in ML?", options: ["Un singolo algoritmo", "Una raccolta di dati per addestrare modelli", "Un modello già addestrato", "Un tipo di computer"], answer: "Una raccolta di dati per addestrare modelli", category: "Parte 1" },
        { question: "Cosa si intende per 'Overfitting'?", options: ["Il modello è troppo semplice", "Il modello impara troppo bene il training set e non generalizza", "Il modello non ha abbastanza dati", "Il modello ha troppi pochi iperparametri"], answer: "Il modello impara troppo bene il training set e non generalizza", category: "Parte 1" },
        { question: "Il 'Bias nei Dati' può portare a sistemi AI:", options: ["Sempre più accurati", "Ingiusti o discriminatori", "Più veloci", "Senza errori"], answer: "Ingiusti o discriminatori", category: "Parte 1" },
        { question: "Quale linguaggio di programmazione è considerato lo standard de facto per l'AI/ML?", options: ["Java", "C++", "Python", "JavaScript"], answer: "Python", category: "Parte 1" },
        { question: "Cosa fa la libreria Pandas in Python?", options: ["Calcolo numerico su array", "Creazione di grafici", "Manipolazione e analisi di dati tabulari", "Costruzione di reti neurali"], answer: "Manipolazione e analisi di dati tabulari", category: "Parte 1" },
        { question: "Scikit-learn è una libreria principalmente usata per:", options: ["Deep Learning avanzato", "Machine Learning classico", "Visualizzazione 3D", "Gestione database"], answer: "Machine Learning classico", category: "Parte 1" },
        // Paradigmi ML
        { question: "L'Apprendimento Supervisionato richiede dati che siano:", options: ["Non etichettati", "Etichettati (con risposte corrette)", "Casuali", "Solo numerici"], answer: "Etichettati (con risposte corrette)", category: "Parte 2" },
        { question: "La 'Classificazione' è un tipo di problema di Apprendimento Supervisionato dove l'output è:", options: ["Un valore numerico continuo", "Una categoria discreta", "Una struttura nascosta", "Una sequenza di azioni"], answer: "Una categoria discreta", category: "Parte 2" },
        { question: "Quale algoritmo è un esempio di Apprendimento Non Supervisionato?", options: ["Regressione Lineare", "K-Means (Clustering)", "Albero Decisionale", "Support Vector Machine"], answer: "K-Means (Clustering)", category: "Parte 2" },
        { question: "L'Apprendimento per Rinforzo si basa su un agente che impara tramite:", options: ["Etichette fornite da un supervisore", "Scoperta di pattern in dati non etichettati", "Ricompense e penalità interagendo con un ambiente", "Regole pre-programmate"], answer: "Ricompense e penalità interagendo con un ambiente",  category: "Parte 2" },
        { question: "Cosa significa 'Riduzione della Dimensionalità'?", options: ["Aumentare il numero di feature", "Semplificare i dati riducendo il numero di feature", "Rendere il modello più complesso", "Etichettare i dati"], answer: "Semplificare i dati riducendo il numero di feature", category: "Parte 2" },
        // Pipeline ML
        { question: "Cos'è il 'Data Preprocessing'?", options: ["La fase finale di valutazione del modello", "La trasformazione di dati grezzi in un formato adatto all'addestramento", "La scelta dell'algoritmo", "Il deployment del modello"], answer: "La trasformazione di dati grezzi in un formato adatto all'addestramento",  category: "Parte 3" },
        { question: "Perché si dividono i dati in 'training set' e 'test set'?", options: ["Per rendere l'addestramento più veloce", "Per avere più dati", "Per valutare la generalizzazione del modello su dati mai visti", "Per confondere il modello"], answer: "Per valutare la generalizzazione del modello su dati mai visti", category: "Parte 3" },
        { question: "Una metrica comune per la Classificazione è:", options: ["Mean Squared Error (MSE)", "Accuratezza (Accuracy)", "R-squared", "Root Mean Absolute Error"], answer: "Accuratezza (Accuracy)", category: "Parte 3" },
        { question: "Cos'è la 'Convalida Incrociata' (Cross-Validation)?", options: ["Un tipo di algoritmo ML", "Una tecnica per validare il codice Python", "Una tecnica per ottenere una stima più robusta delle prestazioni del modello", "Un modo per aumentare i dati"], answer: "Una tecnica per ottenere una stima più robusta delle prestazioni del modello", category: "Parte 3" },
        { question: "Il 'Feature Engineering' si riferisce a:", options: ["La scelta dell'hardware per l'AI", "La creazione e selezione di feature migliori per il modello", "L'ottimizzazione degli iperparametri", "La scrittura del codice dell'algoritmo"], answer: "La creazione e selezione di feature migliori per il modello", category: "Parte 3" },
        // Deep Learning
        { question: "Il Deep Learning utilizza Reti Neurali Artificiali (ANN) con:", options: ["Un solo strato di input", "Nessuno strato nascosto", "Molteplici strati nascosti ('profonde')", "Solo strati di output"], answer: "Molteplici strati nascosti ('profonde')", category: "Parte 4" },
        { question: "Quale funzione di attivazione è molto popolare negli strati nascosti delle reti profonde per la sua efficienza?", options: ["Sigmoide", "Tanh", "ReLU", "Lineare"], answer: "ReLU", category: "Parte 4"  },
        { question: "TensorFlow e Keras sono:", options: ["Linguaggi di programmazione AI", "Tipi di dataset", "Framework per il Deep Learning", "Tecniche di visualizzazione"], answer: "Framework per il Deep Learning", category: "Parte 4" },
        { question: "Le CNN (Reti Neurali Convoluzionali) sono particolarmente adatte per:", options: ["Dati tabulari semplici", "Analisi di serie temporali", "Elaborazione di immagini", "Clustering generico"], answer: "Elaborazione di immagini", category: "Parte 4" },
        { question: "Cosa fanno gli strati di 'Pooling' in una CNN?", options: ["Aumentano la dimensione delle feature map", "Riducono la dimensione spaziale e controllano l'overfitting", "Aggiungono rumore ai dati", "Imparano i filtri convoluzionali"], answer: "Riducono la dimensione spaziale e controllano l'overfitting", category: "Parte 4" },
        { question: "Le RNN (Reti Neurali Ricorrenti) sono progettate per elaborare:", options: ["Immagini statiche", "Dati sequenziali (es. testo, serie temporali)", "Dati non etichettati", "Solo numeri"], answer: "Dati sequenziali (es. testo, serie temporali)", category: "Parte 4" },
        { question: "Qual è il problema principale che le LSTM cercano di risolvere rispetto alle RNN semplici?", options: ["L'eccessiva velocità di calcolo", "La difficoltà con poche feature", "Il problema della memoria a lungo termine (vanishing/exploding gradients)", "La mancanza di interpretabilità"], answer: "Il problema della memoria a lungo termine (vanishing/exploding gradients)", category: "Parte 4" },
        { question: "La 'Data Augmentation' per immagini serve a:", options: ["Ridurre la qualità delle immagini", "Aumentare artificialmente la dimensione del training set", "Semplificare l'architettura della CNN", "Velocizzare l'inferenza"], answer: "Aumentare artificialmente la dimensione del training set", category: "Parte 4" },
        { question: "Il 'Transfer Learning' in Deep Learning consiste nel:", options: ["Trasferire dati da un computer all'altro", "Usare un modello pre-addestrato su un grande dataset come base per un nuovo task", "Imparare a trasferire stili tra immagini", "Convertire il modello in un altro linguaggio"], answer: "Usare un modello pre-addestrato su un grande dataset come base per un nuovo task", category: "Parte 4" },
        // IA Generativa e Applicazioni
        { question: "Cosa fa principalmente l'IA Generativa (GenAI)?", options: ["Classifica dati", "Prevede valori numerici", "Crea contenuti nuovi e originali", "Raggruppa dati simili"], answer: "Crea contenuti nuovi e originali", category: "Parte 5" },
        { question: "Gli LLM (Large Language Models) sono tipicamente basati su quale architettura?", options: ["CNN", "RNN Semplice", "Transformer", "K-Means"], answer: "Transformer", category: "Parte 5" },
        { question: "Cos'è il 'Prompt Engineering'?", options: ["La progettazione di hardware per LLM", "L'arte di scrivere input testuali efficaci per guidare un LLM", "L'addestramento di un LLM da zero", "La valutazione delle performance di un LLM"], answer: "L'arte di scrivere input testuali efficaci per guidare un LLM", category: "Parte 5" },
        { question: "La 'Finestra di Contesto' di un LLM si riferisce a:", options: ["Il numero di utenti che possono usarlo contemporaneamente", "La quantità massima di informazioni che può considerare in un dato momento", "La velocità con cui genera risposte", "La dimensione fisica del modello"], answer: "La quantità massima di informazioni che può considerare in un dato momento", category: "Parte 5" },
        { question: "Stable Diffusion e Midjourney sono strumenti per:", options: ["Generare testo", "Generare musica", "Generare immagini da testo", "Analizzare dati finanziari"], answer: "Generare immagini da testo", category: "Parte 5" },
        { question: "Ollama è uno strumento che facilita:", options: ["L'accesso a API cloud di LLM", "La creazione di grafici statistici", "L'esecuzione locale di LLM open source", "La scrittura di codice Python"], answer: "L'esecuzione locale di LLM open source", category: "Parte 5" },
        { question: "L'AI Agentiva si riferisce a sistemi AI che possono:", options: ["Solo rispondere a domande", "Pianificare, agire e usare strumenti per raggiungere obiettivi", "Generare solo immagini", "Funzionare solo offline"], answer: "Pianificare, agire e usare strumenti per raggiungere obiettivi", category: "Parte 5" },
        { question: "GitHub Copilot è un esempio di:", options: ["Un modello di traduzione automatica", "Un AI Coding Tool", "Un sistema di raccomandazione musicale", "Un LLM per la generazione di storie"], answer: "Un AI Coding Tool", category: "Parte 5" },
        { question: "Cos'è il 'Deployment' di un modello ML?", options: ["L'addestramento iniziale del modello", "La scelta degli iperparametri", "Il processo di rendere un modello addestrato accessibile e utilizzabile", "La pulizia dei dati di input"], answer: "Il processo di rendere un modello addestrato accessibile e utilizzabile", category: "Parte 5" },
        { question: "L'obiettivo principale della XAI (Explainable AI) è:", options: ["Rendere i modelli AI più veloci", "Rendere i modelli AI più grandi", "Rendere le decisioni dei modelli AI comprensibili agli umani", "Automatizzare completamente il training"], answer: "Rendere le decisioni dei modelli AI comprensibili agli umani", category: "Parte 5" },
        { question: "Quale di questi è un esempio di 'Apprendimento per Rinforzo'?", options: ["Classificare email come spam o non spam", "Prevedere il prezzo di un'azione", "Un robot che impara a camminare per tentativi ed errori", "Raggruppare clienti con comportamenti simili"], answer: "Un robot che impara a camminare per tentativi ed errori", category: "Parte 5" }
    ];
    
     // --- ELEMENTI DOM ---
    const tocElement = document.getElementById('table-of-contents');
    const contentTitleElement = document.getElementById('content-title');
    const contentBodyElement = document.getElementById('content-body');
    
    const glossarySection = document.getElementById('glossary-section');
    const glossaryContentElement = document.getElementById('glossary-content');
    const showGlossaryButton = document.getElementById('show-glossary');

    const quizSection = document.getElementById('quiz-section');
    const quizContentElement = document.getElementById('quiz-content');
    const showQuizButton = document.getElementById('show-quiz');
    const nextQuestionButton = document.getElementById('next-question');
    const quizFeedbackElement = document.getElementById('quiz-feedback');
    
    const quizCategorySelect = document.getElementById('quiz-category-select'); // Definito
    const restartQuizButton = document.getElementById('restart-quiz-button'); // Definito

    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResultsContainer = document.getElementById('search-results-container');
    const searchResultsList = document.getElementById('search-results-list');

    // --- VARIABILI DI STATO ---
    let currentQuizQuestions = [];
    const numberOfQuestionsForQuiz = 5;
    let currentQuizQuestionIndex = 0;
    let activeChapterItem = null;

    // --- FUNZIONI HELPER ---
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    function setActiveChapter(item) {
        if (activeChapterItem) {
            activeChapterItem.classList.remove('active');
        }
        activeChapterItem = item;
        if (activeChapterItem) {
            activeChapterItem.classList.add('active');
        }
    }

    function hideAllContentSections() {
        contentBodyElement.classList.add('hidden');
        glossarySection.classList.add('hidden');
        quizSection.classList.add('hidden');
        searchResultsContainer.classList.add('hidden');
    }

    // --- LOGICA VISUALIZZAZIONE CONTENUTO E RICERCA ---
    function displayChapterContent(chapter, searchTerm = '') { /* ... come ultima versione ... */
        hideAllContentSections(); 
        contentBodyElement.classList.remove('hidden'); 

        contentTitleElement.textContent = chapter.title;
        
        let htmlContent = chapter.content || "<p>Sintesi non ancora disponibile per questo capitolo.</p>";
        
        if (searchTerm) {
            const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escapedSearchTerm})`, 'gi');
            htmlContent = htmlContent.replace(regex, '<span class="highlight">$1</span>');
        }
        
        contentBodyElement.innerHTML = htmlContent;
        contentBodyElement.scrollTop = 0; 
    }

    function performSearch() { /* ... come ultima versione ... */
        const searchTermOriginal = searchInput.value.trim();
        const searchTerm = searchTermOriginal.toLowerCase();
        searchResultsList.innerHTML = '';

        if (searchTerm.length < 2) {
            searchResultsContainer.classList.add('hidden');
            return;
        }

        const foundChapters = [];
        ebookData.forEach(part => {
            part.chapters.forEach(chapter => {
                if (chapter.content && chapter.content.toLowerCase().includes(searchTerm)) {
                    foundChapters.push(chapter);
                }
            });
        });
        
        hideAllContentSections(); 

        if (foundChapters.length > 0) {
            contentTitleElement.textContent = "Risultati della Ricerca"; 
            foundChapters.forEach(chapter => {
                const li = document.createElement('li');
                const chapterDisplayTitle = chapter.title.includes(':') ? chapter.title.split(':')[1].trim() : chapter.title;
                li.textContent = chapterDisplayTitle;
                li.addEventListener('click', () => {
                    displayChapterContent(chapter, searchTermOriginal); 
                    const tocItem = tocElement.querySelector(`li[data-id="${chapter.id}"]`);
                    setActiveChapter(tocItem); 
                });
                searchResultsList.appendChild(li);
            });
            searchResultsContainer.classList.remove('hidden'); 
        } else {
            contentTitleElement.textContent = "Risultati della Ricerca";
            const li = document.createElement('li');
            li.textContent = "Nessun capitolo trovato.";
            searchResultsList.appendChild(li);
            searchResultsContainer.classList.remove('hidden');
        }
        setActiveChapter(null); 
    }

    // --- GENERAZIONE INDICE ---
    ebookData.forEach(partData => { /* ... come ultima versione ... */
        const partTitleDiv = document.createElement('div');
        partTitleDiv.className = 'part-title';
        partTitleDiv.textContent = partData.part;
        tocElement.appendChild(partTitleDiv);

        const ul = document.createElement('ul');
        partData.chapters.forEach(chapter => {
            const li = document.createElement('li');
            li.className = 'chapter-item';
            const chapterDisplayTitle = chapter.title.includes(':') ? chapter.title.split(':')[1].trim() : chapter.title;
            li.textContent = chapterDisplayTitle;
            li.setAttribute('data-id', chapter.id);
            li.addEventListener('click', () => {
                displayChapterContent(chapter);
                setActiveChapter(li);
            });
            ul.appendChild(li);
        });
        tocElement.appendChild(ul);
    });

    // --- LOGICA GLOSSARIO ---
    showGlossaryButton.addEventListener('click', () => { /* ... come ultima versione ... */
        hideAllContentSections(); 
        glossarySection.classList.remove('hidden'); 

        contentTitleElement.textContent = "Glossario Essenziale AI"; 
        glossaryContentElement.innerHTML = '';
        for (const term in glossaryData) {
            const termDiv = document.createElement('div');
            termDiv.innerHTML = `<strong>${term}:</strong> ${glossaryData[term]}`;
            glossaryContentElement.appendChild(termDiv);
        }
        setActiveChapter(null); 
        nextQuestionButton.classList.add('hidden');
        quizFeedbackElement.textContent = '';
        quizFeedbackElement.className = '';
    });

    // --- LOGICA QUIZ (CON DEBUG E VERIFICHE) ---
    function populateCategorySelector() {
        console.log("Populating categories...");
        const categories = new Set();
        allQuizQuestions.forEach(q => {
            if (q.category && q.category.trim() !== "") {
                categories.add(q.category.trim());
            } else {
                // console.warn("Domanda senza categoria o con categoria vuota:", q.question); // Commentato per meno rumore se le categorie sono OK
            }
        });
        console.log("Categorie uniche trovate:", Array.from(categories)); // Mostra come array per leggibilità

        const existingOptions = quizCategorySelect.options;
        for (let i = existingOptions.length - 1; i > 0; i--) { // Inizia da 1 per non rimuovere "Tutte le Categorie"
            quizCategorySelect.remove(i);
        }
        
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            quizCategorySelect.appendChild(option);
        });
        console.log("Selettore categorie popolato con", quizCategorySelect.options.length, "opzioni totali.");
    }

    function startNewQuiz() {
        const selectedCategoryValue = quizCategorySelect.value;
        console.log("startNewQuiz - Categoria Selezionata (valore):", selectedCategoryValue);
        let questionsToUse = [];

        if (selectedCategoryValue === "all") {
            questionsToUse = [...allQuizQuestions];
        } else {
            questionsToUse = allQuizQuestions.filter(q => q.category && q.category.trim() === selectedCategoryValue);
        }
        
        console.log(`startNewQuiz - Filtrate ${questionsToUse.length} domande per la categoria '${selectedCategoryValue}'.`);

        if (questionsToUse.length === 0) {
            quizContentElement.innerHTML = `<p>Nessuna domanda disponibile per la categoria "${selectedCategoryValue}". Scegline un'altra o seleziona "Tutte le Categorie".</p>`;
            nextQuestionButton.classList.add('hidden');
            restartQuizButton.classList.add('hidden'); // Nascondi se non ci sono domande
            currentQuizQuestions = [];
            console.log("startNewQuiz - Nessuna domanda trovata dopo il filtro, quiz non avviato.");
            return;
        }
        
        shuffleArray(questionsToUse);
        currentQuizQuestions = questionsToUse.slice(0, Math.min(numberOfQuestionsForQuiz, questionsToUse.length));
        
        if (currentQuizQuestions.length < numberOfQuestionsForQuiz && currentQuizQuestions.length > 0) {
            console.warn(`Attenzione: Selezionate ${currentQuizQuestions.length} domande per il quiz, meno delle ${numberOfQuestionsForQuiz} richieste per la categoria "${selectedCategoryValue}".`);
        }

        if (currentQuizQuestions.length === 0) {
             quizContentElement.innerHTML = `<p>Non ci sono abbastanza domande (${numberOfQuestionsForQuiz} richieste) per avviare il quiz per questa categoria.</p>`;
             nextQuestionButton.classList.add('hidden');
             restartQuizButton.classList.add('hidden'); // Nascondi se non ci sono domande effettive per il quiz
             console.log("startNewQuiz - Non abbastanza domande per avviare il quiz.");
             return;
        }

        currentQuizQuestionIndex = 0;
        displayQuizQuestion(currentQuizQuestionIndex);
        restartQuizButton.classList.remove('hidden'); // Ora dovrebbe essere mostrato
        console.log("startNewQuiz - Quiz avviato con", currentQuizQuestions.length, "domande.");
    }

    function displayQuizQuestion(index) { /* ... come ultima versione ... */
        console.log("displayQuizQuestion - index:", index, "currentQuizQuestions length:", currentQuizQuestions.length); // DEBUG
        quizFeedbackElement.textContent = '';
        quizFeedbackElement.className = '';

        if (currentQuizQuestions.length === 0 || index >= currentQuizQuestions.length) {
            let endMessage = "<p>Nessuna domanda per il quiz.</p>"; // Messaggio di default
            if (currentQuizQuestions.length > 0 && index >= currentQuizQuestions.length){ // Se c'erano domande ma sono finite
                endMessage = "<p>Quiz completato! Ottimo lavoro.</p>";
            }
            quizContentElement.innerHTML = endMessage;
            nextQuestionButton.classList.add('hidden');
            // restartQuizButton dovrebbe rimanere visibile per rigiocare
            return;
        }

        const q = currentQuizQuestions[index];
        quizContentElement.innerHTML = `
            <div class="quiz-question">${q.question}</div>
            <div class="quiz-options">
                ${q.options.map((optionText) => `
                    <label>
                        <input type="radio" name="quizOption" value="${optionText}">
                        ${optionText}
                    </label>
                `).join('')}
            </div>
        `;
        nextQuestionButton.classList.remove('hidden');
        nextQuestionButton.textContent = "Verifica Risposta";
        nextQuestionButton.onclick = checkAnswer;
    }

    function checkAnswer() { /* ... come ultima versione ... */
        const selectedOptionInput = quizContentElement.querySelector('input[name="quizOption"]:checked');
        if (!selectedOptionInput) {
            quizFeedbackElement.textContent = "Per favore, seleziona una risposta.";
            quizFeedbackElement.className = 'incorrect';
            return;
        }

        const questionData = currentQuizQuestions[currentQuizQuestionIndex];
        if (selectedOptionInput.value === questionData.answer) {
            quizFeedbackElement.textContent = "Corretto!";
            quizFeedbackElement.className = 'correct';
        } else {
            quizFeedbackElement.textContent = `Sbagliato. La risposta corretta era: ${questionData.answer}`;
            quizFeedbackElement.className = 'incorrect';
        }

        const radioButtons = quizContentElement.querySelectorAll('input[name="quizOption"]');
        radioButtons.forEach(rb => rb.disabled = true);

        if (currentQuizQuestionIndex < currentQuizQuestions.length - 1) {
            nextQuestionButton.textContent = "Prossima Domanda";
            nextQuestionButton.onclick = () => {
                currentQuizQuestionIndex++;
                displayQuizQuestion(currentQuizQuestionIndex);
            };
        } else {
            nextQuestionButton.textContent = "Fine Quiz"; 
            nextQuestionButton.onclick = () => { 
                quizContentElement.innerHTML += "<p><strong>Quiz completato! Clicca 'Nuovo Quiz' per rigiocare.</strong></p>";
                nextQuestionButton.classList.add('hidden'); 
            };
        }
    }

    showQuizButton.addEventListener('click', () => {
        console.log("Bottone Mini-Quiz cliccato");
        hideAllContentSections();
        quizSection.classList.remove('hidden');
        // searchResultsContainer.classList.add('hidden'); // Già in hideAllContentSections

        contentTitleElement.textContent = "Mini-Quiz sull'AI";
        populateCategorySelector(); // Chiamata qui!
        startNewQuiz(); 
        
        setActiveChapter(null);
    });

    quizCategorySelect.addEventListener('change', () => {
        console.log("Categoria cambiata, avvio nuovo quiz.");
        startNewQuiz();
    });

    restartQuizButton.addEventListener('click', () => {
        console.log("Bottone Nuovo Quiz cliccato.");
        startNewQuiz();
    });
    
    // --- EVENT LISTENERS RICERCA ---
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (event) => { /* ... come prima ... */ });
    searchInput.addEventListener('input', () => { /* ... come prima ... */ });

}); // Fine DOMContentLoaded