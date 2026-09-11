# Value/quality factor exploration log

Univers cible : S&P 500 / sous-ensemble liquide 100-150 titres. Signal :
score combiné value (P/B, P/E) + qualité (marge, dette/equity, stabilité
des bénéfices), long sur le meilleur quintile/décile, rebalancement
mensuel. Règles de validation et d'arrêt : voir la consigne utilisateur du
2026-07-19 (mode autonome pour la journée) et `lessons_learned.md`.

## État au démarrage de la session autonome (2026-07-19, ~03:49 UTC)

- Univers de base : les 97 tickers déjà collectés pour PEAD (`earnings_cache/`).
- Fondamentaux (bilan + résultat) : 11/97 tickers en cache (JPM, ABBV, AME,
  AMGN, AON, AXON, AXP, BAC, BAX, BBWI, CAT).
- Secteur (OVERVIEW) : 0/97 -- jamais collecté, bloqué par le quota avant
  même de commencer.
- **Quota Alpha Vantage confirmé toujours épuisé** (testé en direct à
  nouveau au démarrage de cette session : réponse de rate-limit réelle de
  l'API, pas un artefact de session). Aucune nouvelle donnée ne peut être
  collectée tant que le quota (calendaire, côté AV, indépendant de cette
  conversation) n'est pas réinitialisé.
- Décision : construire tout le code (normalisation, score composite,
  moteur de backtest) et le valider mécaniquement sur données synthétiques
  pendant que le quota se recharge, puis relancer le drip-feed
  périodiquement. Ne PAS lancer le "vrai" backtest sur les 11 tickers
  actuellement en cache -- un quintile/décile sur 11 titres n'aurait aucun
  sens statistique (2 titres par groupe), et rapporter un chiffre sur un
  échantillon aussi mince serait exactement le genre d'erreur que ce projet
  a appris à éviter (voir lessons_learned.md, piège #4 sur la sélection
  bruitée par petit échantillon).

## Viability check attempt -- 2026-07-19T03:54:09.427529+00:00

- Usable tickers (earnings + balance_sheet + income_statement + overview all cached): 0 of 97
- **Below MIN_UNIVERSE_SIZE (40) -- refusing to run a backtest.** A quintile/decile on 0 tickers is not statistically meaningful (see lessons_learned.md). Waiting for the drip-feed to progress further before attempting a real run.

---

## Bilan de la session autonome du 2026-07-19 (~00h00-04h00, heure locale)

**Code construit et vérifié mécaniquement (données synthétiques) :**
- `bot/factor_normalization.py` -- rang/percentile sectoriel, repli
  Industry->Sector si groupe < 8, `group_size_report()` pour documenter
  le taux de repli. Auto-test synthétique : mécanique confirmée correcte.
- `bot/value_quality_signal.py` -- métriques point-in-time par ticker
  (P/B, P/E trailing-4Q, marge nette TTM, dette/equity, stabilité des
  bénéfices = -CV des 8 derniers trimestres d'EPS), cross-section
  point-in-time par date de rebalancement, score composite (poids égal
  value/qualité, poids égal intra-pilier -- aucune raison documentée de
  dévier, donc poids égaux comme convenu). Auto-test synthétique (20
  titres, 2 secteurs, 4 industries) : pipeline correct de bout en bout.
- `bot/value_quality_backtest.py` -- rebalancement mensuel, panier =
  meilleur quintile (`select_basket`, fraction configurable pour
  décile), équipondéré, AUCUN levier (le plafond d'exposition agrégée de
  lessons_learned.md est respecté PAR CONSTRUCTION ici, pas besoin d'un
  mécanisme de cap séparé comme pour PEAD -- documenté explicitement dans
  le code). Réutilise `SimTrade`/`compute_metrics`/`monte_carlo_bootstrap`
  de `backtest_engine.py` et `equity_curve_from_trades` de `pead_backtest.py`
  pour cohérence avec le reste du projet. `cost_sensitivity()` et
  `regime_slice_metrics()` (découpage par sous-période, même patron que
  le test PEAD 2016-2020) ajoutés. Auto-test synthétique (20 titres,
  marche aléatoire, 2019-2020) : 19 rebalancements, 77 trades, pipeline
  bout-en-bout fonctionnel (résultat négatif attendu -- c'est un bruit
  aléatoire, pas un signal réel, le test ne prouve que la mécanique).
- `run_value_quality_viability.py` -- orchestrateur final : découvre
  automatiquement les tickers avec earnings+bilan+résultat+secteur TOUS
  en cache, refuse de lancer un backtest sous `MIN_UNIVERSE_SIZE=40`
  (pas de résultat sur un échantillon trop mince -- leçon du projet),
  sinon construit tout, lance le backtest, la sensibilité aux coûts, et
  le découpage par régime (T4-2018, COVID-2020, 18 derniers mois). Prêt
  à tourner tel quel dès que la collecte de données le permet -- aucune
  modification nécessaire, juste le relancer.

**Bug trouvé et documenté** (déjà noté plus haut dans ce fichier) :
`fiscal_date_ending` en texte brut dans `earnings_data.py` cassait
silencieusement la jointure -- corrigé dans `fundamentals_data.py`.

**Collecte de données -- bloquée par une contrainte externe réelle, pas
par manque d'effort :**
- Fondamentaux (bilan + résultat) : 11/97 tickers.
- Secteur (OVERVIEW) : 0/97 -- jamais pu démarrer, quota épuisé avant.
- Quota Alpha Vantage : confirmé épuisé à 3 reprises distinctes pendant
  cette session (réponses de rate-limit réelles de l'API, pas un
  artefact) -- c'est un quota calendaire côté Alpha Vantage, indépendant
  de cette conversation, qui ne se réinitialisera pas plus vite parce
  qu'on continue d'essayer.
- **`run_value_quality_viability.py` a été exécuté et refuse correctement
  de tourner (0 tickers utilisables actuellement < seuil de 40)** --
  comportement voulu, pas un échec.

**Pourquoi aucun backtest "réel" n'a été rapporté aujourd'hui** : avec
0/97 secteurs et 11/97 fondamentaux, il n'existe tout simplement pas
encore assez de données pour un résultat statistiquement défendable. Un
backtest lancé quand même produirait un chiffre qui RESSEMBLE à un
résultat mais n'en serait pas un -- exactement le piège que ce projet a
appris à éviter (voir lessons_learned.md). Ce n'est pas une règle d'arrêt
au sens du blocage "ressource externe" (le code, lui, avance) mais la
collecte de données réelle ne peut pas être accélérée au-delà du quota
journalier de l'API, quel que soit le nombre de tentatives.

**Prochaine étape, dès que possible (aucune modification de code
nécessaire)** : relancer `drip_feed_fundamentals.py` (reprend
automatiquement où il s'est arrêté) jusqu'à couvrir un nombre suffisant
de tickers (idéalement l'univers complet), puis relancer
`run_value_quality_viability.py` -- il détectera automatiquement combien
de tickers sont utilisables et lancera le vrai backtest multi-régimes dès
que le seuil de 40 est atteint. Au rythme actuel (~8-12 tickers
complets/jour selon combien ont déjà 1-2 des 3 pièces), plusieurs jours
supplémentaires de drip-feed restent nécessaires avant d'atteindre ce
seuil.

**Aucune règle d'arrêt n'a été déclenchée** (pas de résultat mitigé à
documenter, pas de câblage dans `bot/main.py`, pas de paper trading) --
la session s'arrête simplement parce que l'étape suivante (plus de
données) ne peut pas être produite plus vite que le quota externe ne le
permet, pas parce qu'un critère de la consigne a été atteint.

## Run automatique (tâche planifiée) -- 2026-07-19T22:54-23:45 (heure locale)

- `drip_feed_fundamentals.py` relancé. Deux passes dans cette session :
  1. Première invocation a progressé de JPM à CLX (14 tickers, secteur
     OVERVIEW récupéré pour chacun) avant d'être interrompue par un
     timeout d'outil (SIGTERM externe, pas un arrêt du script
     lui-même) -- les 14 tickers étaient déjà correctement persistés en
     cache au moment de l'interruption.
  2. Deuxième invocation a repris automatiquement où la première s'était
     arrêtée (comportement attendu, pas de redémarrage à zéro), a
     récupéré CMI, puis le quota Alpha Vantage (25 req/jour) a été
     atteint sur COST : `[ERROR] Alpha Vantage rate limit hit for
     COST/overview ... standard API rate limit is 25 requests per day`.
     Le script s'est arrêté proprement de lui-même
     (`Daily quota exhausted, stopping for today (reached COST)`) --
     comportement voulu, pas un bug.
- **Bilan tickers complets (earnings + balance_sheet + income_statement +
  overview tous en cache) : 15/97** (ABBV, AME, AMGN, AON, AXON, AXP,
  BAC, BAX, BBWI, CAT, CCL, CDW, CLX, CMI, JPM) -- en hausse par rapport
  aux 11/97 (fondamentaux seuls, 0 secteur) du dernier état connu.
  Progression nette de cette session : +4 tickers fondamentaux, +15
  secteurs (0 -> 15).
- `run_value_quality_viability.py` relancé par prudence (toujours sûr
  même sous le seuil, cf. consigne) : refuse correctement de lancer un
  backtest (15 < `MIN_UNIVERSE_SIZE`=40) -- comportement voulu, pas un
  échec.
- Aucune règle d'arrêt de la consigne déclenchée (pas de résultat à
  documenter comme candidat/rejeté, pas de blocage hors quota). Rythme
  observé aujourd'hui (~15 tickers complets cumulés, avec seulement 1
  nouveau ticker OVERVIEW obtenu après reprise avant d'épuiser le quota)
  confirme l'estimation précédente : plusieurs dizaines de jours de
  drip-feed restent nécessaires à ce rythme pour atteindre 40 tickers
  utilisables, puis l'univers complet de 97. Prochaine exécution
  planifiée : relancer `drip_feed_fundamentals.py` puis
  `run_value_quality_viability.py` dans le même ordre, aucune
  modification de code nécessaire.

## Viability check attempt -- 2026-07-20T03:45:51.503300+00:00

- Usable tickers (earnings + balance_sheet + income_statement + overview all cached): 15 of 97
- **Below MIN_UNIVERSE_SIZE (40) -- refusing to run a backtest.** A quintile/decile on 15 tickers is not statistically meaningful (see lessons_learned.md). Waiting for the drip-feed to progress further before attempting a real run.

---

## Évaluation des options pour accélérer la collecte de fondamentaux (2026-07-21)

**Contexte** : 15/97 tickers complets (earnings+bilan+résultat+secteur),
limité par le quota gratuit Alpha Vantage (25 req/jour). Recherche
demandée : un plan payant AV le justifierait-il, ou existe-t-il une
alternative gratuite/moins chère ? Documentation seulement -- rien
implémenté, décision laissée à l'utilisateur (coût réel engagé).

### Option 1 : Alpha Vantage, plan payant

| Palier | Prix/mois | Débit | Plafond journalier |
|---|---|---|---|
| Gratuit (actuel) | $0 | 5/min | **25/jour** |
| Premium 75rpm | $49.99 | 75/min | Aucun |
| Premium 150rpm | $99.99 | 150/min | Aucun |
| Premium 300rpm+ | $149.99-$249.99 | 300-1200/min | Aucun |

Le palier le moins cher ($49.99/mois) supprime totalement le plafond de
25/jour -- à 75/min, les ~200 appels restants (97 tickers x ~2 pièces
manquantes en moyenne) se feraient en quelques minutes au lieu de
semaines. **Mais** : la page de tarification ne précise pas explicitement
si BALANCE_SHEET/INCOME_STATEMENT/OVERVIEW sont inclus sans surcoût --
à confirmer auprès du support avant de payer si cette option est retenue.
Source : [alphavantage.co/premium](https://www.alphavantage.co/premium/).

### Option 2 : SEC EDGAR (`data.sec.gov`) -- GRATUIT, testé en direct

**Testé en direct maintenant, pas juste lu sur une page marketing** :
`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` fonctionne
sans clé API, retourne pour AAPL 503 concepts XBRL avec tout l'historique
des dépôts (10-K/10-Q depuis des années). Champs présents et vérifiés :
`StockholdersEquity`, `Assets`, `Liabilities`, `Revenues`, `NetIncomeLoss`,
`LongTermDebt`, `GrossProfit`, `CommonStockSharesOutstanding` -- tout ce
dont `bot/fundamentals_data.py` a besoin. `ShortTermBorrowings` absent
pour AAPL (utilise probablement un tag voisin -- inconsistance XBRL
normale entre entreprises, à gérer par du code de repli, pas un blocage).

**Découverte importante, meilleure que notre méthode actuelle** : chaque
entrée XBRL inclut un champ `filed` (date de dépôt RÉELLE, ex.
`"filed":"2026-01-30"` pour le 10-Q du trimestre clos le 2025-12-27) --
c'est exactement la date de disponibilité publique que
`estimate_public_availability_dates()` doit actuellement ESTIMER via
`reportedDate + 5 jours ouvrables` (hypothèse documentée, pas un fait).
SEC EDGAR donne la vraie date, éliminant cette hypothèse entièrement --
plus précis qu'Alpha Vantage, pas juste moins cher.

**Débit** : pas de clé API requise ; usage raisonnable généralement admis
par la SEC à ~10 requêtes/seconde avec un User-Agent identifiant
l'appelant (testé ici avec un User-Agent de test, fonctionne). Une seule
requête par entreprise retourne TOUT l'historique -- 97 tickers
tiendraient en une poignée de secondes/minutes, pas en 80+ jours.

**Coût réel, pas monétaire** : pas de champ secteur/industrie GICS
équivalent à celui d'Alpha Vantage (le point d'entrée `submissions`
donne un code SIC -- `"sic":"3571","sicDescription":"Electronic
Computers"` pour AAPL -- une taxonomie plus ancienne/grossière, utilisable
mais pas identique à `bot/factor_normalization.py`'s hypothèse actuelle).
Et les tags XBRL ne sont PAS uniformes entre entreprises (certaines
utilisent `Revenues`, d'autres
`RevenueFromContractWithCustomerExcludingAssessedTax`, etc.) -- il faudrait
écrire une couche de repli/mapping de tags, un vrai travail d'ingénierie
ponctuel (quelques heures à un jour ou deux), pas un abonnement récurrent.
Pas de données de surprise de résultats (EARNINGS) -- mais ce n'est plus
nécessaire pour value/qualité (contrairement à PEAD, déjà retiré) puisque
`filed` remplace l'usage qu'on faisait de `reportedDate`.
Source : [sec.gov/search-filings/edgar-application-programming-interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

### Option 3 : Financial Modeling Prep (FMP)

Plans payants de $99/mois à $2500/an, limites basées sur la bande
passante mensuelle (500Mo gratuit -> 20-150Go payant), pas clairement
un débit de requêtes. La page consultée ne précise pas explicitement à
partir de quel palier BALANCE_SHEET/INCOME_STATEMENT avec profondeur
historique sont inclus -- moins bien vérifié que les autres options ici,
à re-creuser si retenu. Plus cher que les alternatives pour un service
qui, in fine, retraite probablement les mêmes dépôts SEC.
Source : [site.financialmodelingprep.com/pricing-plans](https://site.financialmodelingprep.com/pricing-plans).

### Option 4 : Polygon.io (rebaptisé "Massive")

| Palier | Prix/mois | Débit | Financials inclus |
|---|---|---|---|
| Stocks Basic (gratuit) | $0 | 5/min | Non |
| Stocks Starter | $29 | illimité | Non |
| Stocks Developer | $79 | illimité | Non |
| Stocks Advanced | $199 | illimité | Oui |
| **Financials & Ratios (autonome)** | **$29** | -- | Oui (produit dédié) |

Le produit autonome "Financials & Ratios" à $29/mois est l'option payante
la MOINS chère parmi toutes les alternatives évaluées -- moins chère que
le palier minimal d'Alpha Vantage ($49.99). Données déjà nettoyées/
normalisées (pas de travail de mapping de tags XBRL à faire soi-même).
Source : [massive.com/pricing](https://massive.com/pricing) (redirection
confirmée depuis polygon.io/pricing -- même entreprise, nouveau nom).

### Tableau récapitulatif

| Option | Coût | Débit réel | Fiabilité/complétude | Travail d'intégration |
|---|---|---|---|---|
| AV payant (75rpm) | $49.99/mois | Illimité (75/min) | Identique à l'actuel (déjà intégré) | Aucun -- juste retirer le plafond |
| **SEC EDGAR** | **Gratuit** | **~10/s, 1 req = tout l'historique** | **Meilleure (date de dépôt réelle)**, secteur via SIC (plus grossier) | Modéré -- mapping de tags XBRL |
| FMP | $99+/mois | Bande passante, pas clair | Non vérifié en détail | Faible (API déjà structurée) |
| Massive Financials add-on | $29/mois | Non précisé | Bonne (déjà nettoyée) | Faible |

### Recommandation

**SEC EDGAR d'abord** -- gratuit, vérifié en direct maintenant (pas
supposé), et objectivement MEILLEUR que ce qu'Alpha Vantage donne
actuellement sur le point le plus fragile de toute cette piste (la date
de disponibilité publique, jusqu'ici une hypothèse documentée mais non
vérifiée). Le vrai coût n'est pas monétaire mais un peu de temps
d'ingénierie pour gérer les variantes de tags XBRL et le mapping SIC ->
secteur -- raisonnable vu le temps déjà investi dans le drip-feed actuel,
et ce travail ne se répète pas (contrairement à un abonnement).

**Si l'ingénierie XBRL est jugée trop coûteuse en temps** : le produit
autonome Massive "Financials & Ratios" à $29/mois est la meilleure
alternative payante (moins cher qu'Alpha Vantage, données déjà propres).

**Ne PAS recommander** : payer Alpha Vantage juste pour lever le plafond
de débit -- ça ne résout aucun problème de fond (précision des dates,
complétude) que SEC EDGAR ne résout pas déjà gratuitement, et ça engage
un coût récurrent pour un problème qui n'a besoin que d'un investissement
ponctuel.

**Rien implémenté** -- décision laissée à l'utilisateur. Le drip-feed
`drip_feed_fundamentals.py` continue de tourner normalement en tâche de
fond, inchangé.

## Viability check attempt -- 2026-07-23T15:10:13.985972+00:00

- Usable tickers (SEC EDGAR fundamentals + Alpha Vantage sector both cached): 15 of 97
- **Below MIN_UNIVERSE_SIZE (40) -- refusing to run a backtest.** A quintile/decile on 15 tickers is not statistically meaningful (see lessons_learned.md). Waiting for the drip-feed to progress further before attempting a real run.

---

## Migration Alpha Vantage -> SEC EDGAR pour les fondamentaux (2026-07-23)

Suite à la décision de l'utilisateur (voir section précédente) de partir
sur SEC EDGAR. Plan exécuté en 6 points, documenté ci-dessous dans l'ordre.

### 1. `bot/fundamentals_data.py` -- nouvelle source

`SECEdgarFundamentalsClient` remplace `AlphaVantageFundamentalsClient` pour
BALANCE_SHEET/INCOME_STATEMENT. Utilise `data.sec.gov/api/xbrl/companyfacts/`
(gratuit, sans clé). `get_quarterly_fundamentals()` garde EXACTEMENT le
même format de sortie (colonnes identiques) donc `value_quality_signal.py`
n'a nécessité aucune modification. `get_sector_industry` (Alpha Vantage
OVERVIEW) reste inchangé.

### 2. Couche de mapping XBRL -- ordre de priorité documenté

`BALANCE_SHEET_TAGS`/`LONG_TERM_DEBT_TAGS`/`SHORT_TERM_DEBT_TAGS`/
`INCOME_STATEMENT_TAGS` dans `bot/fundamentals_data.py` : listes ordonnées
de tags candidats, testées PAR TRIMESTRE (pas une fois par entreprise) --
gère correctement les entreprises qui ont changé de tag en cours d'historique
(ex. beaucoup sont passées de `Revenues` à
`RevenueFromContractWithCustomerExcludingAssessedTax` vers 2018, ASC 606).
Si AUCUN tag candidat n'a de donnée pour un concept, il reste explicitement
`NaN` et le nom du concept est ajouté à `missing_concepts` (jamais deviné
silencieusement, jamais substitué par un autre concept). Un repli
supplémentaire "stale" (`RECENT_DATA_CUTOFF_YEARS=3`) détecte les concepts
qui ont EU des données historiquement mais plus aucune récemment (voir
le cas JPM/`total_revenue` ci-dessous).

**3 bugs réels trouvés et corrigés pendant la construction** (pas des
suppositions -- chacun vérifié sur les données brutes avant et après
correction, voir `lessons_learned.md` pour le détail complet) :
1. `CommonStockSharesIssued` n'est pas interchangeable avec
   `CommonStockSharesOutstanding` -- créait un faux motif en dents de scie
   dans le book value par action. Corrigé en utilisant
   `dei:EntityCommonStockSharesOutstanding` comme repli au lieu.
2. Ce même tag `dei` utilise la date de PAGE DE COUVERTURE comme `end`,
   pas la vraie fin de trimestre -- créait des lignes fantômes. Corrigé
   par un ré-alignement au `total_shareholder_equity` le plus proche
   (tolérance 45 jours).
3. Un concept de résultat (revenu T4 discret de CAT) disponible SEULEMENT
   via un 8-K très tardif (déposé plus d'un an après le trimestre) polluait
   la date de disponibilité de TOUTE la ligne, y compris des concepts de
   bilan disponibles normalement 6 semaines après. Corrigé : la date de
   disponibilité n'est désormais déterminée que par les concepts de
   BILAN (equity/actifs/passifs/dette), jamais par ceux du compte de
   résultat.

### 3. Champ `filed` comme date de disponibilité réelle

Fait -- voir `lessons_learned.md` pour la documentation complète de ce
changement en tant qu'amélioration de PRÉCISION (pas juste de coût).
L'ancienne fonction `estimate_public_availability_dates`
(reportedDate+5j ouvrés) est supprimée -- plus nécessaire, la vraie date
est directement disponible.

### 4. Secteur -- Alpha Vantage OVERVIEW conservé

Inchangé, comme décidé. `drip_feed_fundamentals.py` réécrit pour ne plus
faire QUE ça (secteur seul) -- voir point 6 pour l'estimation du temps
restant.

### 5. Impact du code SIC sur le seuil de repli -- résultat net, pas ajusté

Testé sur les 97 tickers réels (codes SIC récupérés gratuitement via
`data.sec.gov/submissions/`, aucun coût additionnel) : **si SIC était
utilisé à la place d'Alpha Vantage OVERVIEW, le repli Industry->Sector se
déclencherait pour 100% des tickers (67/67 groupes SIC à 4 chiffres sont
sous le seuil de 8 -- la plupart des entreprises ont un code SIC unique
dans notre univers de 97 titres).** Pire encore : même au niveau
"sector" plus large (2 premiers chiffres du SIC, 36 groupes), la plupart
restent SOUS le seuil de 8 aussi (seuls 2 des 36 groupes -- "28"
Chimie/Pharma et "38" Instruments -- atteignent 10 membres ; le reste va
de 1 à 7). **Conclusion : SIC ne serait PAS une alternative viable à AV
OVERVIEW pour cet univers et ce seuil, à AUCUN niveau de granularité.**
Ça confirme et renforce la décision du point 4 (garder AV OVERVIEW) --
ce n'est pas juste "plus grossier", c'est structurellement inadapté à un
univers de cette taille. **Seuil non modifié**, comme demandé --
signalé ici pour décision.

### 6. Collecte complète relancée sur les 97 tickers

**106.9 secondes pour 96/97 tickers réussis** (`collect_sec_fundamentals.py`)
-- contre plusieurs semaines avec le drip-feed Alpha Vantage. Le 97e
(XOM) a échoué à la première tentative pour une raison inattendue et
vérifiée, pas un bug de ce projet : **le fichier officiel
`company_tickers.json` de la SEC associe actuellement le ticker "XOM" à
une filiale interne ("ExxonMobil Holdings Corp", CIK 2115436,
`tickers:[]`, `exchanges:[]` sur son propre endpoint /submissions/) et
PAS à la vraie ExxonMobil cotée (CIK 34088, confirmé
`tickers:["XOM"]`, `exchanges:["NYSE"]`)**. Corrigé via un mapping de
repli explicite et documenté (`CIK_OVERRIDES`), pas une supposition --
vérifié directement contre les deux endpoints avant de trancher. **97/97
tickers ont maintenant leurs fondamentaux SEC EDGAR en cache.**

**Limites de données réelles trouvées, documentées, pas contournées** :
- `gross_profit` absent pour ~70/97 tickers (concept XBRL non
  obligatoire, beaucoup d'entreprises -- notamment financières et
  services -- ne le taguent jamais). **Sans impact sur le signal actuel**
  -- ce concept n'est utilisé par aucune des 5 métriques de
  `bot/value_quality_signal.py`.
- `total_liabilities` absent pour ~25/97 tickers. **Également sans
  impact actuel** -- pas utilisé directement par le signal (récupéré
  pour un usage futur éventuel).
- `total_revenue` manquant ou obsolète (plus de 3 ans sans donnée) pour
  plusieurs financières (JPM, GS, RF, TFC confirmés) -- les banques
  cessent souvent de taguer un revenu trimestriel agrégé. **Impact réel** :
  `net_margin_ttm` (une des 3 métriques qualité) sera `NaN` pour ces
  titres.
- `total_debt` manquant ou obsolète pour ~8 tickers (GRMN, MPWR, PCAR,
  DHI, INCY, L, NVR, TROW). **Impact réel** : `debt_to_equity` sera `NaN`
  pour ces titres.
- `shares_outstanding` manquant (MKC, STZ) ou obsolète (REGN, UPS, V).
  **Impact réel, le plus significatif** : ce concept alimente P/B ET P/E
  (les 2 métriques value) -- ces titres auraient un `value_score`
  inutilisable. À surveiller si ça affecte la taille effective de
  l'univers une fois le seuil de 40 atteint.
- Pour les financières spécifiquement (JPM confirmé, probablement
  d'autres) : `debt_to_equity` calculé via les tags standards est ~20x
  plus faible qu'avec l'ancien calcul Alpha Vantage (0.13-0.19 contre
  2.0-3.4) -- les banques ne rentrent pas dans le même moule bilan que
  les entreprises industrielles ; leur endettement réel (dépôts, passifs
  de trading, etc.) n'est pas capturé par les tags dette long/court terme
  standards. **Ce chiffre n'est probablement pas comparable, même
  sectoriellement, pour les financières** -- à garder en tête si le
  score final semble anormalement favorable pour ce secteur.

### État actuel de la collecte

- **Fondamentaux (SEC EDGAR) : 97/97 tickers.**
- **Secteur (Alpha Vantage OVERVIEW) : 15/97 tickers** (inchangé --
  `drip_feed_fundamentals.py` n'a pas encore tourné depuis la migration).
  À ~25/jour dédiés entièrement au secteur maintenant (plus de
  concurrence avec les fondamentaux), ~4 jours restants pour les 82
  manquants, comme anticipé.
- **Tickers avec les 2 complets : 15/97** -- `run_value_quality_viability.py`
  relancé par prudence : refuse correctement (15 < seuil de 40),
  comportement voulu, pas une erreur.

**Prochaine étape, aucune modification de code nécessaire** : laisser
`drip_feed_fundamentals.py` (secteur uniquement désormais) continuer via
la tâche planifiée déjà en place -- elle appelle ce script par son nom de
fichier, qui pointe maintenant vers la version secteur-seul automatiquement.

## Viability check attempt -- 2026-08-04T00:57:14.707412+00:00

- Usable tickers (SEC EDGAR fundamentals + Alpha Vantage sector both cached): 97 of 97
- After building point-in-time metrics: 97 tickers usable (dropped 0 with empty metrics/price)
- Backtest span: 2016-01-04 to 2026-07-31

## Viability check attempt -- 2026-08-04T00:58:05.244014+00:00

- Usable tickers (SEC EDGAR fundamentals + Alpha Vantage sector both cached): 97 of 97
- After building point-in-time metrics: 97 tickers usable (dropped 0 with empty metrics/price)
- Backtest span: 2016-01-04 to 2026-07-31
- Basket size over time: min=19 max=22 (quintile of ~97 tickers)
- Sharpe=0.561  Sortino=0.587  MaxDD=27.0%  MC_DD95=97.2%  PF=1.350  WinRate=54.7%  Trades=2480
- Cost sensitivity: 0.05%->127.4%, 0.10%->114.8%, 0.15%->102.2%
- Viability (full span): {'sharpe_ge_0.5': True, 'mc_dd95_le_20pct': False, 'cost_15bps_positive': True, 'all_pass': False}

### Regime breakdown

- Q4-2018 correction (2018-09-20 to 2018-12-26): {'sharpe': -2.440664662494002, 'sortino': -3.206363800624641, 'max_drawdown': 0.22405855827992288, 'recovery_days': nan, 'win_rate': 0.2112676056338028, 'profit_factor': 0.12378867813326007, 'trade_count': 71, 'total_return': np.float64(-0.16838409748004113), 'final_equity': 83161.59025199589}
- COVID crash (2020-02-19 to 2020-04-07): {'sharpe': -0.35719464425654035, 'sortino': -0.57535250860085, 'max_drawdown': 0.3556433377467273, 'recovery_days': nan, 'win_rate': 0.525, 'profit_factor': 0.5712812360982197, 'trade_count': 40, 'total_return': np.float64(-0.11239240757912516), 'final_equity': 88760.75924208749}
- Recent (last 18mo) (2025-02-02 to 2026-08-04): {'sharpe': 0.23174967486180623, 'sortino': 0.2918936099892127, 'max_drawdown': 0.14874912480042637, 'recovery_days': 105.0, 'win_rate': 0.5124223602484472, 'profit_factor': 1.0908594876397408, 'trade_count': 322, 'total_return': np.float64(0.04589529104357748), 'final_equity': 104589.52910435775}

---

## VERDICT FINAL -- value/qualité NE PASSE PAS (2026-08-03)

Univers complet (97/97 tickers, fondamentaux SEC EDGAR + secteur Wikipédia/AV),
backtest réel lancé sur 2016-01-04 à 2026-07-31.

- **Sharpe global** : 0.561 (passe de justesse le seuil de 0.5).
- **Par régime, décisif et sans ambiguïté** :
  - T4-2018 (correction) : Sharpe = -2.44
  - Krach COVID (2020-02 à 2020-04) : Sharpe = -0.36
  - 18 derniers mois : Sharpe = 0.23
  **Aucun des 3 régimes testés n'atteint 0.5** -- le Sharpe global positif
  vient entièrement de périodes non testées individuellement, exactement
  le piège que ce projet a appris à éviter (ne jamais faire confiance à
  un Sharpe agrégé sans le vérifier par régime -- voir PEAD, ORB).
- **MC_DD95 = 97.2%** -- vérifié que ce n'est pas un bug de données (pires
  trades individuels ~-46%, plausibles, survenus en mars 2020). Plus
  probablement un artefact de méthode : le bootstrap Monte Carlo réutilisé
  de `backtest_engine.py` traite chaque trade comme une position
  SÉQUENTIELLE à capital plein, alors qu'ici ~20 positions tournent
  CONCURREMMENT (panier quintile) -- non corrigé, documenté pour référence
  future, mais **sans incidence sur le verdict** puisque le MaxDD réel
  (27.0%) et l'échec par régime suffisent déjà à trancher.
- **MaxDD réel historique** : 27.0%.

**VERDICT : NE PASSE PAS les critères de viabilité de ce projet.** Aucune
retouche de paramètres tentée. Rien câblé dans `bot/main.py`, aucun paper
trading.

**Cause probable, pour référence** : le signal value/qualité classique
(P/B, P/E, marge, dette/equity) semble avoir un edge réel sur le long
terme (Sharpe brut positif) mais insuffisant et instable par régime --
cohérent avec la littérature académique qui documente que les facteurs
value/qualité simples ont des périodes prolongées de sous-performance
(ex. toute la décennie 2010s pour value pur) plutôt qu'un edge stable
trimestre après trimestre.

## SYNTHÈSE -- 4 pistes explorées, 4 rejets

PEAD (retiré), ORB-fade intraday (rejeté), put-write options (rejeté),
value/qualité (ne passe pas). Aucune piste testée dans ce projet n'a
survécu à une validation multi-régime rigoureuse jusqu'ici. `bot/main.py`
reste désactivé.
