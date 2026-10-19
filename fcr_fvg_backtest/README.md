# First Candle Rule + FVG — banc de test quantitatif

Objectif : déterminer si la stratégie *First Candle Rule + Fair Value Gap* (SPY, QQQ, ES, NQ, séance US) possède
un avantage statistique après coûts — pas démontrer qu'elle marche.

## 0. Statut honnête (à lire d'abord)

| Élément | Statut |
|---|---|
| Moteur Python (règles, exécution, coûts, sizing, stats, bootstrap, graphiques) | **Écrit et testé** : 64 tests pytest passent |
| Backtest sur **données de marché réelles** | **NON EXÉCUTÉ** — voir ci-dessous |
| Résultats de performance (win rate, PF, drawdown…) | **Aucun.** Rien n'a été calculé sur de vraies données ; aucun chiffre n'est donné |
| Pine Script v6 | **Écrit, jamais compilé** (pas d'accès à TradingView) |
| Scripts de téléchargement (`fetch/`) | **Écrits, jamais exécutés** (réseau bloqué) |

**Pourquoi aucune donnée réelle :** l'environnement d'exécution de cette session refuse toute connexion sortante
vers les fournisseurs de données (réponse HTTP 403 du proxy pour Yahoo Finance, Stooq, Alpaca, Polygon, Databento,
Alpha Vantage, Tiingo, Dukascopy, Kaggle, Hugging Face). Le dépôt ne contient pas non plus de barres 1 minute
(seulement des barres 15 minutes, inutilisables : les reconstituer ou les « descendre » en 1 minute est interdit
par le cahier des charges). Aucune donnée n'a été simulée pour la remplacer.

**Ce qui a été exécuté avec des données synthétiques :** uniquement des tests techniques (marches aléatoires générées
dans `tests/synth.py`) pour vérifier que le code tourne et respecte ses invariants. Ces sorties ne sont **pas** des
résultats de marché. À titre de *contrôle négatif* du code : sur ces marches aléatoires avec coûts, le moteur donne
une espérance négative (≈ −0,17 R SPY, −0,26 R ES sur 250 séances simulées), c.-à-d. il ne fabrique pas d'avantage ;
cela ne dit rien sur la stratégie réelle.

### Pour obtenir les vrais résultats (≈ 30 min)

1. Récupérer des barres 1 minute **brutes (non ajustées)** :
   * SPY/QQQ : `fetch/alpaca_fetch.py` (flux SIP ; ce dépôt a déjà documenté un accès SIP depuis 2016 sur le compte
     Alpaca de l'utilisateur — non revérifié ici), ou n'importe quel CSV/Parquet.
   * ES/NQ : `fetch/databento_fetch.py` (CME GLBX.MDP3, payant — lancer `--cost-only` d'abord), ou CSV par contrat.
   * Yahoo (`fetch/yfinance_fetch.py`) : ~30 derniers jours seulement → exploratoire, jamais concluant.
2. `pip install -r requirements.txt && python -m pytest tests -q`
3. ```
   python run_backtest.py --data SPY=data/SPY_1m.parquet QQQ=data/QQQ_1m.parquet \
                          ES=data/ES_1m.parquet NQ=data/NQ_1m.parquet --out results/run1
   ```
4. Lire `results/run1/qc/*.json` **avant** les performances, puis `report_auto.md`, `summary.csv`, `trades/`, `charts/`.
   `audit/` contient les barres brutes des 10 premiers trades de chaque série pour vérification manuelle.

Format des fichiers : colonne temps (`timestamp`/`datetime`/`ts_event`… ; ISO avec offset, epoch s/ms/ns, ou naïf avec
`--input-tz`), `open high low close` (+ `volume`). Futures : colonne `contract` (symbole brut ESH5…), barres par contrat.
`--ts-convention end` si le fournisseur date les barres à leur clôture.

---

## 1. Règles exactes et interprétations

Index : la barre *i* ouvre à 9 h 30 + *i* minutes, heure de New York (EST/EDT via la base tz). `t` = barre qui vient de clôturer ;
rien d'indice > `t` n'est lu pendant son traitement (test de troncature dans `tests/`).

| Règle (cahier des charges) | Implémentation |
|---|---|
| Opening range | H/L = max(high)/min(low) des **cinq** barres 1 min 9 h 30–9 h 34. Séance exclue si une des cinq manque. Utilisable à partir de 9 h 35 |
| FVG | 3 barres 1 min consécutives (c1,c2,c3), **toutes ouvertes à ≥ 9 h 35**. LONG : `open(c2) ≤ H`, `close(c2) > H`, `low(c3) > high(c1)` (inégalité stricte). SHORT : symétrique. Zone LONG = [high(c1), low(c3)] ; SHORT = [high(c3), low(c1)] |
| Existence du FVG | À la clôture de c3 ; testable dès la barre suivante |
| Lecture stricte | « La 2ᵉ bougie traverse le niveau avec son corps » = ouvre ≤ H et clôture > H. **Interprétation stricte**, identifiée comme telle |
| Durée de vie du FVG | **Hypothèse** : 15 barres après c3 (barres t+1…t+15). Sensibilité 5/10/15/20/30 |
| Retest | Barre ultérieure dont [low, high] intersecte la zone fermée |
| Confirmation | **Hypothèse** : bougie englobante sur la barre de retest ou les 2 suivantes. Corps de la barre courante englobe celui de la barre immédiatement précédente (de couleur opposée, présente). Égalités acceptées (`<=`, `>=`). Sensibilité 0/1/2/3/5 |
| Entrée | Ordre marché à la clôture de la barre englobante, exécuté à l'**ouverture suivante** ± slippage |
| 1 trade/jour/instrument, limite 11 h 00 | La barre d'exécution doit ouvrir **strictement avant 11 h 00** (donc ≤ 10 h 59). Un setup rejeté à l'exécution (stop invalide, taille nulle, barre manquante) ne consomme pas la journée : on passe au setup suivant. Rejets journalisés |
| Plusieurs FVG | Plusieurs candidats vivent en parallèle (même en sens opposés) ; le premier à être confirmé et valide est pris |
| Stop A | LONG : plus bas des barres [retest … confirmation] − 1 tick ; SHORT : plus haut + 1 tick |
| Stop B | LONG : bas de zone (high(c1)) − 1 tick ; SHORT : haut de zone (low(c1)) + 1 tick |
| Rejets | Stop ≥ entrée (LONG) / ≤ entrée (SHORT), risque ≤ 0, taille < 1 unité, barre d'exécution absente |
| Objectif | entrée réelle (slippage inclus) ± 3 × risque initial, arrondi au tick vers l'extérieur. Aucun trailing / BE / partiel |
| Fin de séance | Sortie marché à la clôture de la dernière barre de la séance (15 h 59, ou 12 h 59 les demi-séances NYSE), ± slippage |

Les variantes A et B sont toujours rapportées **séparément**, aucune n'est « choisie ».

## 2. Hypothèses d'exécution, de coûts et de sizing (toutes paramétrables, aucune optimisée)

* **Slippage** : défavorable, **à chaque exécution** (entrée, stop, objectif, sortie de fin de séance) : 1 tick (principal), sensibilité 1/2/3/5. Le spread est supposé inclus dans ce slippage (les barres sont des prix de transactions, pas bid/ask).
* **Stop** : stop-marché. Touché si low ≤ stop (LONG). Exécution au niveau du stop, ou à l'ouverture si la barre ouvre au-delà (gap), puis − slippage.
* **Objectif** : traité comme market-if-touched : touché si high ≥ objectif, exécuté à l'objectif − slippage. Un gap au-delà de l'objectif n'est **pas** crédité d'un meilleur prix.
* **Barre ambiguë** (stop et objectif dans la même barre 1 min) : **le stop est pris en premier.**
* La barre d'entrée est elle-même testée pour stop/objectif (option `check_exit_on_entry_bar`, sensibilité fournie).
* **Commissions** (hypothèses documentées, non mesurées) : ETF = 0,005 $/action/ordre, min 1 $, max 1 % de la valeur ; frais réglementaires SEC/FINRA non modélisés (fractions de cent par action). Futures = 2,25 $/contrat/côté tout compris.
* **Capital** 100 000 $ composé trade après trade ; risque cible 0,5 % du capital courant.
* **Taille** = plus grand entier tel que la perte stop-out *tout compris* (distance + slippage de sortie + commissions A/R) ≤ 0,5 % ; **jamais arrondie à 1 unité vers le haut** : si 1 contrat dépasse le budget (ex. NQ : un stop de 25 points = 500 $/contrat avant coûts), le signal est rejeté et compté.
* **Plafonds d'exposition** : ETF notionnel ≤ 4 × capital (marge intraday Reg-T, courtier-dépendant ; sensibilité à 1×). Futures : marge initiale ≤ 50 % du capital (ES 15 000 $, NQ 25 000 $ par contrat — valeurs prudentes **supposées**, elles ont varié dans le temps).
* ETF : actions entières, vente à découvert supposée possible sans coût d'emprunt (SPY/QQQ).
* **R** = P&L net / risque nominal (unités × |entrée − stop| × valeur du point), donc une perte de stop vaut un peu moins de −1 R une fois les coûts inclus.

## 3. Données et contrôle qualité (`fcr_fvg/data.py`)

Jamais d'interpolation ni de remplissage : une minute manquante reste manquante (NaN) et casse la consécutivité.
Pipeline : NaN → alignement à la minute → **filtre RTH avant tout** (pré/post-marché et overnight jamais vus) → cohérence OHLC
(barre invalide supprimée, pas réparée) → futures : **un seul contrat par séance** (le plus échangé ce jour-là, jamais de
raccord intra-séance) → doublons → calendrier NYSE (`pandas_market_calendars` : fériés, demi-séances à 13 h, DST).
Exclusions de séance, fixées *avant* de lancer : 5 premières minutes incomplètes ; toute minute manquante dans [9 h 30, 11 h 00) ;
dernière barre > 5 min avant la clôture officielle ; jour hors calendrier. Minutes manquantes après 11 h : tolérées (un
minute sans transaction ne peut toucher aucun niveau) et jamais remplies. Tout est compté dans `qc/*.json`.

Limites connues : sans colonne `contract`, un futures « continu » ne permet pas de vérifier les rolls intra-séance (signalé dans le QC) ;
les flux IEX (Alpaca gratuit) omettent des minutes et ont des OHLC différents du consolidé → à éviter.

## 4. Validation hors échantillon

Chronologique : 70 % des séances (dates triées, coupure commune à tous les instruments) = développement, 30 % = validation.
Aucun paramètre n'est ajusté ni sur l'une ni sur l'autre (les règles sont fixées a priori) ; les deux segments sont donc
des mesures de **stabilité**. Un walk-forward avec ré-optimisation n'a pas de sens ici (rien à optimiser) ; il est remplacé
par les tables par année / semestre et la grille de sensibilité (un paramètre à la fois, jamais utilisée pour choisir).
Incertitude : bootstrap **par séance** (les jours sont rééchantillonnés, y compris sans trade), IC 95 % de l'espérance en R et du PF ;
concentration sur les meilleurs jours (part des gains des 5 meilleurs trades, P&L sans eux).
Si l'historique est < ~1 an, `report_auto.md` le dit via les effectifs ; aucun verdict ne doit être tiré d'un petit échantillon.

## 5. Ce qui a été testé (64 tests, `pytest tests -q`)

* **Signaux** (`test_signals.py`) : OR = 5 premières minutes seulement ; FVG exigeant des barres ≥ 9 h 35 ; inégalités strictes ; c2 ouvre ≤ H / clôture > H ; pas de retest sur c3 ; durée de vie 15 barres (frontière 27/28) ; fenêtre de confirmation (k, k+1, k+2 acceptés, k+3 refusé) ; conditions d'englobement et égalités ; limite 11 h 00 (frontière 10 h 59/11 h 00) ; barres manquantes ; miroir LONG/SHORT exact ; **absence de lookahead** (détruire toutes les barres après le signal ne change pas le signal) ; aucun signal avant 9 h 38.
* **Exécution/P&L** (`test_execution.py`) : trade LONG et ES calculés **à la main** (prix d'entrée, stop A/B, objectif 3R, taille, commissions, P&L net, R) ; stop avant objectif dans une barre ambiguë ; gap au-delà du stop (exécuté à l'ouverture) ; gap au-delà de l'objectif non crédité ; sortie fin de séance et demi-séance ; stop invalide et barre d'entrée manquante rejetés ; slippage 1/2/3/5 ; valeurs ES 12,50 $/tick (50 $/pt) et NQ 5 $/tick (20 $/pt) ; règles de taille et plafonds.
* **Données** (`test_data.py`) : UTC → 9 h 30 New York de part et d'autre du passage à l'heure d'été ; timestamps naïfs refusés sans fuseau ; epoch s/ms/ns ; extended hours, week-end et Vendredi saint exclus ; demi-séance 210 min ; OR incomplète, minutes manquantes jamais remplies, séance tronquée ; doublons et OHLC invalides ; futures = un contrat/séance.
* **Invariants sur marches aléatoires** (`test_engine_properties.py`) : ≤ 1 trade/jour ; entrée entre 9 h 39 et 10 h 59 ; sortie ≥ entrée et ≤ 16 h ; stop du bon côté ; objectif = entrée ± 3 × risque ; risque nominal ≤ 0,5 % ; P&L = brut − slippage − commissions ; capital composé correctement ; causalité du scan ; slippage croissant ⇒ espérance non croissante.
* Pipeline complet (chargement → rapport → graphiques → bootstrap → sensibilité) exécuté sur données synthétiques : s'exécute sans erreur.

**Non testé** : tout sur données réelles (aucune vérification manuelle de trades historiques réels n'a pu être faite — `audit/` est fait pour ça) ; les scripts `fetch/` ; la compilation et le comportement du Pine Script ; la fidélité des hypothèses de commissions/marge à un courtier réel ; le comportement des futures réels autour des rolls.

## 6. Pine Script vs Python (`pine/first_candle_fvg_strategy.pine`)

Même machine à états. Différences **attendues** (non mesurées) :

1. **Barre d'entrée** : Python teste stop/objectif dès la barre d'entrée ; en Pine le bracket n'est posé qu'après le fill (pour mesurer le risque sur le vrai prix), donc actif à partir de la barre suivante.
2. **Taille** : Pine estime l'entrée par la clôture de la barre signal (+ slippage) ; Python utilise la vraie ouverture suivante.
3. **Slippage** : Python l'applique aussi à l'objectif ; TradingView ne l'applique en principe pas aux ordres limite.
4. **Commissions** : minimum de 1 $/ordre ETF non reproductible dans le simulateur TradingView.
5. **Fin de séance** : `strategy.close_all(immediately=true)` (nom du paramètre à vérifier) à la dernière barre régulière ; si indisponible, sortie à l'ouverture suivante.
6. **Qualité des données** : Python exclut les séances incomplètes ; TradingView utilise les données du fournisseur du graphique (consolidé ou non, historique 1 min limité selon l'abonnement).
7. Calendrier/contrats : TradingView utilise le contrat continu choisi sur le graphique.

Si les résultats divergent, chercher la cause dans cette liste (et via `audit/`) plutôt que retenir le meilleur chiffre.

## 7. Structure

```
fcr_fvg/            instruments.py params.py data.py signals.py execution.py engine.py stats.py plots.py pipeline.py
run_backtest.py     CLI
fetch/              alpaca_fetch.py databento_fetch.py yfinance_fetch.py   (jamais exécutés)
pine/               first_candle_fvg_strategy.pine                          (jamais compilé)
tests/              64 tests (fixtures construites à la main + marches aléatoires pour les invariants)
REPORT.md           rapport final (état actuel : évaluation impossible, voir §0)
```
