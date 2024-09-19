## Slide 1: Funzione Reale e Funzione Stimata

In machine learning, ci concentriamo sull'approssimazione 
di una funzione reale che descrive la relazione tra input e output. 
Questa relazione può essere espressa come:

$$
y = f(x)
$$

Dove:
- $x$ è l'input (le feature),
- $f(x)$ è la funzione reale che produce l'output $y$ (variabile target).

Tuttavia, non conosciamo $f(x)$ esattamente, quindi costruiamo 
un modello per stimare una funzione approssimativa $\hat{f}(x)$. 
La funzione stimata è:

$$
\hat{y} = \hat{f}(x)
$$

Dove:
- $\hat{f}(x)$ è la funzione stimata dal modello,
- $\hat{y}$ è il valore predetto dal modello per un dato $x$.

Le **metriche di errore** ci aiutano a misurare la differenza tra 
l'output reale $y$ e l'output predetto $\hat{y}$, 
per capire quanto il modello si avvicini alla funzione reale.

---

## Slide 1: Introduzione alla Regressione

La **regressione** è uno dei problemi più comuni 
nel machine learning supervisionato, utilizzato 
quando il target che vogliamo predire è un valore 
numerico continuo. In altre parole, il compito di un 
modello di regressione è stimare la relazione 
esistente tra un insieme di **feature** (variabili 
di input) e una **variabile dipendente** continua 
(valore di output). 

Questo tipo di problema si verifica quando 
l'obiettivo è prevedere grandezze misurabili, come 
il prezzo di un bene, la temperatura in una regione, 
il reddito annuale di una persona, o qualsiasi 
altra variabile che può assumere un valore lungo una 
scala numerica. 

---

## Slide 2: Esempio di Regressione

Supponiamo di voler prevedere il **prezzo di una 
casa** in base a diverse caratteristiche, come la 
metratura, il numero di stanze, la posizione e 
altri fattori rilevanti. Le **feature** del nostro 
dataset includono questi attributi, mentre il target 
che vogliamo predire è il prezzo della casa.

Il modello di regressione sarà addestrato utilizzando 
un dataset storico di case vendute, con tutte le 
caratteristiche delle case e i rispettivi prezzi. 
Il modello imparerà a costruire una funzione che 
collega le feature al prezzo e potrà così prevedere 
il prezzo di case nuove che non ha mai visto.

---

## Slide 3: Metriche di Valutazione per la Regressione

Valutare un modello di regressione è cruciale per 
determinare quanto bene il modello sia in grado di 
generalizzare su dati nuovi. Le principali metriche 
di errore utilizzate per la regressione includono:

### 1. **Errore Medio Assoluto (MAE)**
Il MAE misura la media degli errori assoluti tra le 
previsioni del modello e i valori reali. Si calcola 
come:
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y_i} |
$$
Dove:
- $y_i$ è il valore reale,
- $\hat{y_i}$ è il valore predetto dal modello,
- $n$ è il numero totale di osservazioni.

Il MAE ci fornisce una misura diretta della 
deviazione media del modello dalle previsioni. Essendo 
basato sugli errori assoluti, è interpretabile e 
facile da comprendere, ma non penalizza 
particolarmente gli errori grandi.

### 2. **Errore Quadratico Medio (MSE)**
Il MSE penalizza maggiormente gli errori grandi 
rispetto al MAE, poiché eleva al quadrato le 
differenze tra le previsioni e i valori reali. La 
formula è:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$
Dove:
- $y_i$ è il valore reale,
- $\hat{y_i}$ è il valore predetto dal modello,
- $n$ è il numero totale di osservazioni.

Il MSE è particolarmente utile quando vogliamo 
dare maggiore importanza agli errori grandi. 
Tuttavia, a causa del quadrato, questa metrica non 
è direttamente interpretabile rispetto alle unità 
originali della variabile target.

### 3. **Radice dell'Errore Quadratico Medio (RMSE)**
La RMSE è semplicemente la radice quadrata dell'MSE 
e restituisce una misura dell'errore che ha le 
stesse unità della variabile target, rendendo più 
facile interpretare la grandezza degli errori. La 
formula è:
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
$$

Questo valore ci indica, in media, di quanto le 
previsioni del modello differiscono dai valori reali, 
tenendo conto delle unità della variabile predetta.

### 4. **Coefficiente di Determinazione (R²)**
L'R², o **coefficiente di determinazione**, misura la 
proporzione della varianza nel target che può essere 
spiegata dalle feature del modello. La formula è:
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
Dove:
- $y_i$ è il valore reale,
- $\hat{y_i}$ è il valore predetto dal modello,
- $\bar{y}$ è la media dei valori reali.

Un R² vicino a 1 indica che il modello spiega quasi 
tutta la varianza presente nei dati, mentre un R² 
vicino a 0 significa che il modello non è migliore di 
una semplice media delle osservazioni.

---

## Slide 4: Introduzione alla Classificazione

La **classificazione** è un altro dei principali 
problemi di machine learning supervisionato. 
L'obiettivo è assegnare una **classe** o una 
**etichetta** discreta a ciascuna osservazione, 
utilizzando un insieme di feature. A differenza 
della regressione, qui il target è una variabile 
categorica, ossia assume solo un insieme limitato di 
valori (o classi). 

Le applicazioni della classificazione sono numerose: 
possiamo utilizzarla per classificare email in 
**spam** o **non spam**, identificare immagini di 
oggetti, o classificare malattie mediche in base ai 
sintomi dei pazienti.

---

## Slide 5: Esempio di Classificazione

Un esempio pratico di classificazione è il 
**rilevamento di email spam**. Immaginiamo di avere 
un insieme di email, e vogliamo addestrare un modello 
per distinguere le email di **spam** da quelle di 
**non spam**. Le **feature** potrebbero includere il 
numero di parole chiave sospette, la lunghezza del 
messaggio, o la presenza di allegati. 

Il **target** sarà binario: ogni email sarà etichettata 
come **spam** o **non spam**. Il modello imparerà a 
riconoscere queste etichette dai dati di training e, 
una volta addestrato, potrà classificare email nuove 
in una delle due categorie.

---

## Slide 6: Metriche di Valutazione per la Classificazione

Le metriche di valutazione per i modelli di 
classificazione sono fondamentali per determinare 
quanto bene il modello riesce a distinguere tra le 
varie classi. Le principali metriche includono:

### 1. **Accuratezza**
L'accuratezza è la proporzione di previsioni corrette 
rispetto al totale delle previsioni effettuate. Si 
calcola come:
$$
Accuratezza = \frac{TP + TN}{TP + TN + FP + FN}
$$
Dove:
- **TP** sono i veri positivi (previsioni corrette di 
  classe positiva),
- **TN** sono i veri negativi (previsioni corrette di 
  classe negativa),
- **FP** sono i falsi positivi (previsioni errate di 
  classe positiva),
- **FN** sono i falsi negativi (previsioni errate di 
  classe negativa).

L'accuratezza è una metrica generale, ma può essere 
ingannevole in presenza di classi sbilanciate.

### 2. **Precisione**
La precisione misura la proporzione di previsioni 
positive corrette rispetto al totale delle previsioni 
positive effettuate. La formula è:
$$
Precisione = \frac{TP}{TP + FP}
$$

È particolarmente utile quando il costo di un falso 
positivo è elevato, come nella diagnosi medica.

### 3. **Recall (Richiamo)**
Il recall misura la proporzione di veri positivi 
identificati rispetto a tutti i veri positivi 
presenti nel dataset. La formula è:
$$
Recall = \frac{TP}{TP + FN}
$$

Il recall è fondamentale quando il costo di un falso 
negativo è elevato, ad esempio, nel rilevamento di 
malattie gravi.

### 4. **F1-Score**
L'F1-score è la media armonica tra precisione e recall, 
e rappresenta un buon compromesso tra le due metriche. 
La formula è:
$$
F1 = 2 \times \frac{Precisione \times Recall}{Precisione + Recall}
$$

È particolarmente utile quando le classi sono 
sbilanciate e non vogliamo favorire né la precisione 
né il recall.

---

## Slide 7: Introduzione al Clustering

Il **clustering** è una tecnica di machine learning 
non supervisionato, utilizzata per raggruppare i dati 
in **cluster** (gruppi) omogenei. A differenza dei 
problemi di classificazione o regressione, nel 
clustering non abbiamo etichette predefinite o valori 
di output da predire. L'obiettivo principale è quello 
di scoprire automaticamente pattern o strutture 
nascoste nei dati, raggruppando le osservazioni in 
base alla loro somiglianza.

I **cluster** rappresentano sottoinsiemi di dati 
dove le osservazioni all'interno di ogni gruppo sono 
più simili tra loro rispetto ai dati che appartengono 
ad altri gruppi. Il concetto di "somiglianza" può 
essere definito in base alla distanza tra punti in 
uno spazio multidimensionale, come la **distanza 
euclidea** o altre metriche di distanza.

Il clustering è ampiamente utilizzato in vari campi: 
ad esempio, in marketing per identificare gruppi di 
clienti con comportamenti simili, in biologia per 
scoprire nuove specie o categorie di organismi, o 
nell'analisi dei social network per identificare 
comunità o influenze all'interno di reti sociali.

---

## Slide 8: Esempio di Clustering

Un esempio concreto di clustering si può trovare 
nell'**analisi di mercato**. Immaginiamo di gestire 
un'azienda e di avere dati su migliaia di clienti. 
Questi dati includono informazioni come l'età, il 
reddito, la frequenza di acquisto e le categorie di 
prodotti preferite. 

Non conosciamo a priori quali gruppi esistono 
all'interno della nostra base clienti, ma possiamo 
utilizzare un algoritmo di clustering per 
raggruppare i clienti in **segmenti di mercato** 
basati su caratteristiche comuni. 

Ad esempio, possiamo identificare un gruppo di 
clienti giovani con reddito medio-alto che 
preferiscono acquistare tecnologia, o un altro 
gruppo di clienti più anziani che acquistano 
principalmente articoli per la casa. 

Questi segmenti possono poi essere utilizzati per 
strategie di marketing mirate, personalizzando 
offerte e pubblicità per i diversi gruppi, massimizzando 
il tasso di conversione.

---

## Slide 9: Metriche di Valutazione per il Clustering

Valutare un modello di clustering è più complesso 
rispetto ai problemi supervisionati come la 
classificazione o la regressione, perché non 
abbiamo etichette reali con cui confrontare i 
risultati. Tuttavia, esistono metriche che ci 
permettono di valutare la qualità dei cluster 
creati. Tra queste, le più importanti sono:

### 1. **Silhouette Score**
Il **Silhouette Score** misura la coesione e la 
separazione dei cluster. In pratica, indica quanto 
bene un'osservazione è assegnata al proprio cluster 
rispetto agli altri cluster. Il punteggio varia tra 
-1 e 1:
- Un valore vicino a 1 indica che il punto è 
  ben raggruppato nel suo cluster e ben separato 
  dagli altri.
- Un valore vicino a 0 indica che il punto si trova 
  al confine tra due cluster.
- Un valore negativo indica che il punto è stato 
  assegnato a un cluster errato.

La formula del silhouette score è:
$$
S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
Dove:
- $a(i)$ è la distanza media tra il punto $i$ 
  e tutti gli altri punti dello stesso cluster.
- $b(i)$ è la distanza media tra il punto $i$ 
  e tutti i punti del cluster più vicino.

### 2. **Indice di Davies-Bouldin**
L'**Indice di Davies-Bouldin** valuta la qualità dei 
cluster considerando sia la **distanza intra-cluster** 
(quanto sono compatti i cluster) che la **distanza 
inter-cluster** (quanto sono separati i cluster tra 
loro). Un valore più basso indica cluster più ben 
separati e definiti. 

La formula è:
$$
DB = \frac{1}{N} \sum_{i=1}^{N} \max_{i \neq j} 
\left( \frac{s_i + s_j}{d(c_i, c_j)} \right)
$$
Dove:
- $s_i$ è la media delle distanze tra i punti 
  all'interno del cluster $i$,
- $d(c_i, c_j)$ è la distanza tra i centri dei 
  cluster $i$ e $j$,
- $N$ è il numero totale di cluster.

Questo indice combina la compattezza interna dei 
cluster con la loro separazione reciproca.

### 3. **Cohesion e Separation**
Un'altra metrica comune valuta la **coesione** 
(intra-cluster) e la **separazione** (inter-cluster) 
dei cluster. La **coesione** misura quanto i dati 
all'interno di ogni cluster sono vicini tra loro, 
mentre la **separazione** misura quanto i cluster 
sono distanti tra loro. Buoni cluster hanno alta 
coesione e alta separazione.

---

## Slide 10: Train-Test Split e Cross-Validazione

### Train-Test Split
Il **Train-Test Split** è un approccio classico e 
fondamentale per la valutazione dei modelli di 
machine learning. Il dataset viene suddiviso in due 
insiemi principali:
- Il **set di addestramento**, utilizzato per 
  addestrare il modello,
- Il **set di test**, utilizzato per valutare le 
  prestazioni del modello su dati mai visti prima.

Un'implementazione tipica prevede di dividere i dati 
con una proporzione di 80/20 o 70/30. In questo modo, 
possiamo ottenere una stima di quanto bene il modello 
è in grado di **generalizzare** su nuovi dati. Tuttavia, 
questo metodo può essere influenzato dalla casualità 
della divisione, e i risultati possono variare a seconda 
di come sono stati separati i dati.

### Cross-Validazione
La **Cross-Validazione** (CV) è una tecnica avanzata 
che offre una valutazione più robusta e affidabile 
delle prestazioni del modello. Nella **K-Fold 
Cross-Validation**, il dataset viene diviso in **K 
sottoinsiemi** o **folds**. Il modello viene addestrato 
su **K-1 folds** e testato sull'ultimo fold, ripetendo 
questo processo K volte. Ogni fold viene usato una volta 
come set di test.

Un esempio comune è la **5-Fold Cross-Validation**, 
dove i dati vengono suddivisi in 5 parti. Questo 
approccio riduce il rischio di **overfitting**, 
poiché garantisce che il modello venga testato su 
diversi sottoinsiemi del dataset e che i risultati 
siano mediati su più valutazioni.

Inoltre, esistono varianti come la **Leave-One-Out 
Cross-Validation (LOOCV)**, in cui si lascia fuori una 
sola osservazione come set di test ad ogni iterazione. 
Questa tecnica è utile per dataset di piccole dimensioni, 
ma può essere computazionalmente costosa su dataset più 
grandi.

Con la **cross-validazione** otteniamo una stima più 
affidabile delle prestazioni del modello, riducendo il 
rischio che i risultati siano influenzati dalla divisione 
dei dati in training e test.

---

## Conclusione

In questa lezione, abbiamo esplorato i tre principali 
problemi di machine learning: regressione, classificazione e clustering, 
analizzando le rispettive metriche di valutazione. 
Abbiamo poi discusso l'importanza della suddivisione del dataset 
e dell'uso di tecniche come il train-test split e la cross-validazione 
per ottenere modelli più robusti e affidabili. 
Un'adeguata valutazione è fondamentale per garantire 
che i modelli possano generalizzare bene su nuovi dati, 
evitando così i rischi di overfitting o underfitting. Detto questo vi ringrazio 
per l'attenzione e ci vediamo nel prossimo video.
