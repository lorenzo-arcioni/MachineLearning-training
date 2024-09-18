# Lezione 3: Componenti Fondamentali del Machine Learning

Benvenuti alla lezione 3 del nostro corso di Machine Learning.  
Oggi esploreremo i componenti fondamentali del Machine Learning,  
concentrandoci su tre aspetti cruciali: il ruolo dei dati, i modelli  
di Machine Learning e la valutazione dei modelli. Questa panoramica  
sarà teorica e ad alto livello, preparandovi per un approfondimento  
pratico nei corsi successivi.

---

## 3.1 Il Ruolo dei Dati

Iniziamo con il ruolo dei dati. I dati sono il cuore del Machine Learning,  
e la loro qualità e quantità sono determinanti per le prestazioni del  
modello. Senza dati adeguati, un modello di Machine Learning non può esistere.  
L'intero processo di apprendimento si fonda sulla capacità del modello di  
riconoscere pattern, correlazioni e informazioni all'interno dei dati.  
Questo rende i dati uno dei fattori determinanti per il successo  
di un progetto di Machine Learning.

Quando parliamo di qualità dei dati, ci riferiamo a quanto  
bene i dati rappresentano il problema che stiamo cercando di risolvere.  
Dati di alta qualità sono cruciali per ottenere modelli accurati e  
affidabili. Se i dati sono incompleti, rumorosi o inaccurati,  
il modello potrebbe non riuscire a catturare i pattern corretti,  
portando a risultati errati o comunque poco utili.

La quantità dei dati è altrettanto importante. Modelli complessi, come  
le reti neurali profonde, richiedono grandi quantità di dati per  
addestrarsi in modo efficace. Senza un numero sufficiente di esempi,  
il modello può non riuscire a generalizzare bene e potrebbe sovraccaricarsi  
di dettagli specifici del dataset di addestramento, perdendo la  
capacità di funzionare su dati nuovi.

Nel contesto del Machine Learning, i dati vengono generalmente divisi  
in feature e target variables. Le feature, o variabili indipendenti, sono  
gli input del modello, i dati su cui viene addestrato. Al contrario, le  
target variables, o variabili dipendenti, rappresentano l'output che il  
modello cerca di prevedere.

La tabella sopra fornisce un esempio pratico di come le feature e la variabile target  
vengono utilizzate nella previsione dei prezzi delle case. 
 
Le variabili indipendenti, o feature, includono la superficie della casa,  
il numero di stanze, l'anno di costruzione, la posizione (quartiere) e la  
presenza di un giardino. Questi elementi sono utilizzati come input per il  
modello di Machine Learning. Ad esempio, la superficie della casa in metri  
quadrati, il numero di stanze e la posizione nel quartiere sono tutti fattori  
che possono influenzare il valore finale della casa. 

D'altra parte, la variabile dipendente, o target, è il prezzo della casa.  
Questo è l'output che il modello cerca di prevedere basandosi sulle feature.  
In altre parole, il modello utilizza le informazioni sulle caratteristiche  
della casa per stimare il suo prezzo. La previsione del prezzo dipende quindi  
dall'analisi e dalla combinazione delle feature variabili, dimostrando così  
il ruolo cruciale che queste informazioni hanno nel determinare l'output  
finale del modello.

Un aspetto fondamentale della preparazione dei dati è il Feature Engineering.  
Questo processo consiste nella selezione e trasformazione delle  
caratteristiche nei dati per migliorarne l’utilità e le prestazioni del  
modello. Il Feature Engineering può includere la creazione di nuove  
feature che possono rivelare informazioni nascoste, la normalizzazione  
dei dati per garantire che tutte le feature abbiano lo stesso peso, e  
la gestione delle caratteristiche mancanti per evitare che i dati  
incompleti influenzino negativamente il modello.

Tuttavia, durante il Feature Engineering possiamo incontrare problemi  
comuni. Dati non rappresentativi possono introdurre bias nel modello,  
portandolo a fare previsioni errate per determinate categorie di dati.  
Ad esempio, se un modello di riconoscimento facciale è addestrato su  
immagini di persone di una sola etnia, potrebbe non performare bene su  
persone di altre etnie. Dati rumorosi, invece, possono causare  
overfitting, dove il modello impara a riconoscere dettagli specifici e  
distorsioni nel dataset di addestramento invece di pattern generali.

---

## 3.2 Modelli di Machine Learning

Passiamo ora ai modelli di Machine Learning. Un modello è essenzialmente  
una rappresentazione matematica che il sistema apprende dai dati  
di addestramento. Il processo di addestramento comporta l'ottimizzazione  
dei parametri del modello per ridurre l'errore nelle predizioni rispetto  
ai dati etichettati. Questo processo richiede un equilibrio tra  
compessità del modello e la capacità di generalizzare.

I modelli possono variare notevolmente in complessità. Modelli semplici,  
come la regressione lineare, sono relativamente facili da comprendere e  
interpretare. La regressione lineare assume una relazione lineare tra le  
feature e il target, ed è spesso utilizzata per problemi di previsione  
dove si cerca di determinare un valore continuo. Tuttavia, modelli  
semplici possono avere difficoltà a catturare pattern complessi e  
non lineari nei dati.

D'altra parte, modelli complessi come le reti neurali profonde possono  
gestire dati altamente non lineari e complessi. Le reti neurali sono  
composte da strati di nodi interconnessi che elaborano le informazioni,  
e possono identificare pattern complessi in grandi dataset. Tuttavia,  
questa complessità comporta anche rischi. I modelli complessi possono  
essere meno interpretabili e sono più soggetti a overfitting,  
specialmente se non hanno abbastanza dati di addestramento.

---

### Slide 11: La Funzione Reale $f$

La **funzione reale $f$** descrive la relazione tra le variabili in ingresso, o feature, $x$ e i risultati, o target, $y$.  
In altre parole, rappresenta il processo o fenomeno reale che genera i dati che osserviamo.  
Questa funzione esiste, ma nel contesto del Machine Learning non la conosciamo mai con certezza a priori.  
Il nostro obiettivo è quindi quello di avvicinarci il più possibile a questa funzione reale, attraverso l'uso di modelli di apprendimento.

Ad esempio, se consideriamo il prezzo delle case, la funzione reale rappresenterebbe come le caratteristiche della casa, come la dimensione o la posizione, determinano il prezzo finale.

---

### Slide 12: La Funzione Stimata $\hat{f}$

La **funzione stimata $\hat{f}$** è ciò che il nostro modello apprende dai dati disponibili.  
Essa cerca di approssimare la funzione reale $f$, che non è nota.  
L'obiettivo è che la funzione stimata sia il più vicina possibile alla funzione reale, in modo che le previsioni fatte dal modello siano accurate e riflettano il più possibile la realtà.

Nel contesto dell'esempio sul prezzo delle case, la funzione stimata prende le caratteristiche della casa e cerca di predire il prezzo. Più $\hat{f}$ è vicina a $f$, migliori saranno le previsioni del modello.

---

### Slide 14: Misurare la Qualità della Funzione Stimata

Per valutare la **qualità della funzione stimata**, dobbiamo minimizzare l'errore tra $\hat{f}$ e $f$.  
Questo errore può essere misurato in vari modi, a seconda del tipo di problema.

Ad esempio, in un problema di regressione potremmo utilizzare l'Errore Quadratico Medio (MSE), mentre per problemi di classificazione potremmo usare metriche come precisione, recall o F1-score. Vedremo queste metriche in dettaglio in seguito.

In ogni caso, il nostro obiettivo è sempre lo stesso: trovare un modello che minimizzi l'errore e che sia capace di fare previsioni accurate su nuovi dati, non visti in fase di addestramento.

---

In questa lezione abbiamo trattato i componenti fondamentali del  
Machine Learning: il ruolo dei dati, i modelli e le loro valutazioni.  
Abbiamo esplorato i concetti di **funzione reale** e **funzione stimata**, fondamentali per comprendere il funzionamento dei modelli di Machine Learning.  
Abbiamo visto che la funzione reale $f$ rappresenta la vera relazione tra input e output, mentre la funzione stimata $\hat{f}$ è ciò che apprendiamo dai dati per fare previsioni.
Detto questo vi ringrazio per l'attenzione e ci vediamo nella prossima  
lezione, dove approfondiremo ulteriormente il processo di valutazione  
di un modello di Machine Learning.