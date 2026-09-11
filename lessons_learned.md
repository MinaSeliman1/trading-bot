# Lessons learned -- méthodologie de validation pour ce projet

Synthèse des deux investigations menées à ce jour (Stratégies A/B --
mean-reversion / momentum / trend-following / regime+covariance -- puis
PEAD). Objectif : ne pas redécouvrir les mêmes pièges sur la prochaine
piste. Détail complet dans `comparison_report.md` et `exploration_log_pead.md`.

## Verdict des pistes explorées jusqu'ici

| Piste | Statut | Raison |
|---|---|---|
| Stratégie A (mean reversion / momentum / trend-following) | **Rejetée** | OOS Sharpe -3.46, MaxDD 97%, aucun scénario de coût ne survit |
| Stratégie B (regime + covariance + vol targeting) | **Rejetée** | OOS Sharpe -2.51, MaxDD 43% -- meilleure que A (p=0.006) mais toujours très négative |
| PEAD (post-earnings drift) | **Rejetée** | Signal de base fragile (paramètres instables) ; le correctif qui le stabilisait ne généralise pas à une deuxième récession de forme différente |

`bot/main.py` reste désactivé (lève une `RuntimeError` explicite) jusqu'à
ce qu'une stratégie validée remplace le câblage actuel.

---

## Pièges structurels identifiés (à vérifier systématiquement sur toute nouvelle piste)

### 1. Sizing sans plafond d'exposition agrégé
Deux fois, indépendamment, le même bug conceptuel : une taille de position
correcte *par trade* (ATR-risk pour A, notional fixe pour PEAD) sans
plafond sur l'exposition **brute simultanée** de tout le portefeuille.
Résultat : le book peut s'empiler bien au-delà du capital pendant les
périodes où les signaux se regroupent dans le temps (semaines de
résultats pour PEAD, clusters de signaux corrélés pour A) -- risque de
queue qui n'apparaît que dans des tirages malchanceux (`Monte Carlo`),
pas nécessairement dans le chemin historique observé.
**Vérification systématique** : dès la construction du moteur, imposer
un cap d'exposition brute (ex. 1.5x le capital) + un plafond de nouvelles
positions par jour, avant même de chercher un signal -- ce n'est pas un
raffinement, c'est un prérequis d'ingénierie.

### 2. Filtres de régime à l'entrée : rarement causaux
Le filtre "bloquer les nouvelles entrées si le marché est en CORRECTION"
a amélioré les métriques agrégées (Sharpe 2.41→4.70) mais l'attribution
trade-par-trade a montré qu'il retirait des gagnants et gardait des
perdants presque au hasard -- l'amélioration venait surtout de la
réduction de la taille d'échantillon, pas d'un mécanisme causal.
**Vérification systématique** : ne jamais accepter une amélioration
agrégée (Sharpe/drawdown) sans reconstruire, trade par trade, CE QUI a
été retiré et POURQUOI -- si le filtre retire autant de gagnants que de
perdants, ce n'est pas un signal, c'est du bruit habillé en insight.

### 3. Filtres à retard (lagging) : ne généralisent pas entre formes de récession
Un filtre de sortie anticipée basé sur SPY vs sa MM200 a réglé le
problème d'attribution ci-dessus (il regarde toute la période de
détention, pas seulement l'entrée) et a passé les 3 critères sur la
récession lente de 2021-2022. Testé sur une deuxième récession
structurellement différente (correction rapide en V, T4 2018 + krach
COVID), il a échoué nettement : un filtre à retard protège les baisses
lentes et prolongées mais reste actif pendant le rebond qui suit une
chute rapide, détruisant la reprise au lieu de la capturer.
**Vérification systématique** : toute construction basée sur une
moyenne mobile / un indicateur de tendance doit être testée sur AU MOINS
une récession lente ET une récession rapide en V avant d'être crue --
un pass sur une seule forme de récession ne prouve rien sur l'autre.

### 4. Sélection de paramètres par argmax in-sample : peut être pire qu'aléatoire
Sur PEAD, le rang OOS du combo choisi par argmax-Sharpe in-sample s'est
révélé PIRE que le hasard (rang moyen 8.6/12 contre ~6.5 attendu) --
pas seulement bruité, mais activement biaisé, parce que les tenues
longues captent la tendance propre à la fenêtre d'entraînement (qui ne
se répète pas) et gagnent l'argmax pour cette raison non généralisable.
**Vérification systématique** : quand une grille de recherche a des
cellules non indépendantes (ex. seuils imbriqués), sortir la grille
complète, pas seulement le gagnant -- vérifier si le rang IS du gagnant
prédit son rang OOS mieux que le hasard avant de faire confiance au
mécanisme de sélection lui-même.

### 5. Bugs de calcul qui inflatent silencieusement les métriques
L'interpolation linéaire entre prix d'entrée et de sortie pour le P&L
non réalisé (`equity_curve_from_trades`) inflate le Sharpe de 10 à 50x et
masque tout drawdown intra-trade -- détecté seulement parce qu'un
Sharpe de 8.1 semblait implausible et a été creusé plutôt qu'accepté.
**Vérification systématique** : tout chiffre spectaculaire (Sharpe > 3-4
sur une stratégie long-only equities) doit déclencher une investigation
du calcul avant d'être reporté, jamais accepté tel quel. Marquer le P&L
non réalisé au prix réel de clôture, jamais par interpolation.

### 6. Le Monte Carlo (resampling i.i.d.) peut sous-estimer le risque réel de queue
Le drawdown historique réel a tourné à ~2x l'estimation du 95e centile
Monte Carlo dans plusieurs runs -- le resampling par trade indépendant
dilue le risque de pertes corrélées/regroupées dans le temps (plusieurs
positions ouvertes simultanément pendant une même période de stress).
**Vérification systématique** : toujours comparer le MaxDD HISTORIQUE
réel au MC_DD95 estimé -- un écart large (le réel bien au-dessus du MC)
signale un risque de clustering temporel que le Monte Carlo ne capture pas.

### 7. Limites de couverture par source de données -- à vérifier AVANT de construire
- **Alpaca (prix)** : le flux IEX par défaut de ce projet n'a aucune
  donnée pour les actions individuelles avant ~mi-2020, même si le
  flux SIP couvre le même symbole depuis le 2016-01-04 (plafond dur du
  compte, vérifié directement, aucune donnée avant sur aucun flux).
- **Alpha Vantage (earnings)** : profond pour la plupart des titres
  (92/97 avant 2010, souvent jusqu'à 1996) mais quota de 25
  requêtes/jour, 5/minute sur le tier gratuit -- un univers de 100+
  titres nécessite un drip-feed sur plusieurs jours, pas une collecte
  en une passe.
- **Financial Modeling Prep / EODHD** : les endpoints fondamentaux
  utiles nécessitent un plan payant sur le tier gratuit (vérifié par
  recherche, pas testé en live) -- à reconfirmer si on les envisage.
- **Alpaca options (put-write)** : `get_option_bars`/`get_option_trades`
  donnent des prix/trades historiques réels (depuis ~fév. 2024) mais
  AUCUN bid/ask -- pas de classe `OptionQuotesRequest` du tout dans
  alpaca-py (contrairement à `StockQuotesRequest` pour les actions).
  `get_option_latest_quote` ne fonctionne que sur les contrats encore
  cotés (vide sur un contrat expiré). Le spread bid-ask historique
  n'existe donc simplement pas dans cette API, à aucune date -- seule une
  mesure EN DIRECT (aujourd'hui) via `get_option_chain` est possible.
- **yfinance** : `auto_adjust=True` par défaut retourne le prix AJUSTÉ
  aux dividendes, pas le prix réel coté -- vérifié ~3% d'écart sur SPY
  pour une date aussi récente que février 2024. Sans grande incidence
  sur un P&L normalisé (ATM, K=S0) mais fausse directement toute
  construction de strike/contrat réel derrière. Toujours `auto_adjust=False`
  dès qu'un prix "réel" (pas juste un rendement) est nécessaire en aval.
**Vérification systématique** : avant d'écrire une ligne de moteur de
backtest, vérifier en LIVE (pas sur la doc marketing) la profondeur
réelle, le flux/feed correct, et le débit de requêtes de CHAQUE source
de données nécessaire à la nouvelle piste.

**Mise à jour (migration value/quality vers SEC EDGAR, 2026-07-23)** :
- **Précision, pas juste coût** : le champ `filed` de SEC EDGAR (date de
  dépôt réelle du 10-Q/10-K) remplace l'estimation
  `reportedDate + 5 jours ouvrables` utilisée jusqu'ici. C'est une
  amélioration de PRÉCISION, pas seulement de coût/vitesse -- l'hypothèse
  documentée (marge de 5 jours) est désormais un fait mesuré.
- **Mais l'API SEC EDGAR brute a ses propres pièges**, plusieurs trouvés en
  vérifiant les valeurs plutôt qu'en les acceptant telles quelles :
  1. `CommonStockSharesIssued` n'est PAS un synonyme de
     `CommonStockSharesOutstanding` -- le premier inclut les actions
     propres (treasury), le second non. Les mélanger (comme repli l'un de
     l'autre) crée un faux "dents de scie" dans le book value par action
     (vérifié sur JPM : Issued=4.1Md constant, Outstanding=~2.7-2.85Md
     décroissant -- deux séries complètement différentes).
  2. `dei:EntityCommonStockSharesOutstanding` (le bon repli) utilise la
     date de PAGE DE COUVERTURE du dépôt comme `end`, pas la date de fin
     de période fiscale réelle -- nécessite un ré-alignement explicite
     (le plus proche `total_shareholder_equity` à ±45 jours), sinon ça
     crée des lignes fantômes.
  3. Certains concepts (revenu trimestriel discret du T4, typiquement)
     n'existent parfois QUE dans un dépôt bien plus tardif (8-K) -- vérifié
     sur CAT : le T4 2024 discret n'apparaît que dans un 8-K déposé en
     mars 2026, plus d'un an après le trimestre. Laisser un concept aussi
     tardif déterminer la date de disponibilité de toute la ligne
     pénaliserait à tort des ratios (P/B, dette/equity) qui n'en ont pas
     besoin -- la date de disponibilité doit être calculée séparément par
     groupe de concepts (bilan vs résultat), pas globalement.
  4. Le fichier officiel `company_tickers.json` de la SEC peut lui-même
     être incorrect -- "XOM" y était mappé vers une filiale interne sans
     rapport avec le titre cotée (vérifié via `/submissions/` : la vraie
     ExxonMobil cotée a un CIK différent). Un ticker -> CIK "officiel" n'est
     pas forcément fiable sans contre-vérification.
  5. **Pour les entreprises financières (banques), les tags standards ne
     suffisent pas** : `Revenues`/`RevenueFromContract...` et
     `LongTermDebt*`/`ShortTermBorrowings` sont souvent absents ou
     obsolètes pour les banques (vérifié sur JPM : plus aucune donnée de
     revenu trimestriel taguée depuis 2014 ; dette calculée ~20x plus
     faible qu'avec Alpha Vantage, qui fait probablement une agrégation
     propriétaire côté banque). Marge nette et dette/equity ne sont
     structurellement pas fiables pour le secteur financier avec cette
     source -- flagué explicitement (`missing_concepts`), pas deviné.

**Mise à jour (ORB intraday)** : le flux SIP d'Alpaca a aussi un embargo
temps réel (~1h, mesuré en direct : `end=now()` exact échoue avec
"subscription does not permit querying recent SIP data", `end=now()-1h`
fonctionne) -- DISTINCT du plancher de profondeur historique (2016-01-04,
déjà établi ci-dessus). Les deux limites (profondeur passée ET fraîcheur
récente) doivent être vérifiées séparément, pas supposées identiques. Par
ailleurs, ce même plancher de janvier 2016 pour SIP s'applique aussi aux
barres MINUTE (pas seulement journalières) -- vérifié en direct sur
SPY/AAPL. Le flux IEX, en comparaison, n'a aucune donnée minute par action
fiable avant ~janvier 2021 (vérifié en direct), plus tardif que le
"mi-2020" qu'on supposait avant vérification.

### 8. Comparer un modèle à la réalité exige de matcher TOUTES les dimensions pertinentes, pas seulement certaines
Sur put-write : comparer la prime Black-Scholes théorique (calculée à
maturité fixe T=30j) contre la prime réellement cotée d'un contrat
appairé a d'abord montré un "biais" de 224% -- alarmant, mais en grande
partie un artefact : l'échéance mensuelle standard la plus proche tombe
souvent à 14-20 jours réels, pas 30. Comparer un prix à 30j théorique
contre un prix à 15j réel n'est pas un test du modèle, c'est comparer
deux produits différents. Une fois la comparaison faite à maturité
RÉELLEMENT identique, le biais est tombé à ~31% -- toujours réel, mais
d'un ordre de grandeur cohérent avec une limite connue de la méthode
(VIX comme proxy d'IV ATM), pas une erreur cachée.
**Vérification systématique** : avant de conclure qu'un modèle est
biaisé (ou qu'il ne l'est pas) face à des données réelles, vérifier que
la comparaison matche bien TOUTES les variables qui affectent le prix/la
métrique (ici : maturité, mais aussi strike, devise, ajustement de
dividende -- voir aussi le bug yfinance ci-dessus) -- pas seulement
celles qu'on a pensé à égaliser au premier essai.

---

## Ce qui a bien fonctionné -- à reproduire systématiquement

1. **Walk-forward avec règle anti-overfitting explicite** (rejet si
   OOS dévie de plus de 30% de l'IS) -- a mis en évidence l'instabilité
   des paramètres PEAD dès le premier run, jamais ignorée.
2. **Monte Carlo sur les trades OOS** pour une estimation de queue --
   utile même si (point 6 ci-dessus) il faut le croiser avec le
   drawdown historique réel, pas le prendre seul.
3. **Validation croisée par univers de tickers disjoints** -- le test
   le moins cher et le plus efficace contre le surajustement à un
   ensemble de titres particulier. Réutilisé trois fois dans ce projet,
   jamais pris en défaut quand appliqué.
4. **Validation par récessions structurellement différentes** (pas
   juste des fenêtres temporelles différentes dans le même régime) --
   LE test le plus décisif de tout le projet. Le walk-forward standard
   (2023-2026) n'a jamais contenu de correction rapide en V ; c'est en
   allant chercher délibérément 2016-2020 que l'échec du filtre à
   retard a été détecté. Un pass qui ne survit qu'à un seul type de
   récession n'est pas un pass.
5. **Critères de viabilité fixés avant de voir les résultats**
   (Sharpe>=0.5, MC_DD95<=20%, coût-15bps positif) -- décidés en amont,
   jamais déplacés après coup pour faire "passer" un résultat limite.
6. **Attribution trade-par-trade avant de croire une amélioration
   agrégée** -- a débusqué le filtre de régime à l'entrée non causal
   (point 2) alors que les métriques seules auraient semblé valider
   le mécanisme.
7. **Tests de régression sur les bugs de calcul découverts** -- une
   fois le bug d'interpolation linéaire trouvé, un test dédié a été
   ajouté pour qu'il ne puisse pas revenir silencieusement.

## Check-list méthodologique pour toute nouvelle piste

- [ ] Vérifier la disponibilité RÉELLE des données (profondeur, flux/feed,
  débit de requêtes) en live, avant toute ligne de moteur.
- [ ] Plafond d'exposition brute agrégée dès la première version du moteur.
- [ ] Walk-forward avec règle anti-overfitting IS/OOS explicite.
- [ ] Critères de viabilité fixés AVANT de voir un seul résultat.
- [ ] Validation croisée par univers de tickers disjoints.
- [ ] Validation par au moins deux régimes/récessions structurellement
  différents, pas juste deux fenêtres temporelles dans le même régime.
- [ ] Monte Carlo ET comparaison au drawdown historique réel.
- [ ] Attribution trade-par-trade avant de croire toute amélioration
  agrégée liée à un filtre/mécanisme de sélection.
- [ ] Tout chiffre "trop beau" (Sharpe > 3-4) déclenche une relecture du
  calcul avant d'être reporté.

---

## Piste en cours : facteur value/qualité -- hypothèses méthodologiques déjà posées

**Hypothèse de date de disponibilité publique des données fondamentales**
(`bot/fundamentals_data.py`, `estimate_public_availability_dates`) :
Alpha Vantage BALANCE_SHEET/INCOME_STATEMENT ne fournit que
`fiscalDateEnding` (fin de période comptable), jamais de date de dépôt/
publication -- contrairement à EARNINGS qui a une vraie `reportedDate`.
**Choix retenu, explicite et modifiable** : `reportedDate` du trimestre
correspondant (jointure sur `fiscal_date_ending`) + **5 jours ouvrables**
de marge de sécurité CONSERVATRICE (pas 1-2), pour rester du bon côté du
biais look-ahead même si le dépôt SEC traîne après l'annonce des résultats.
C'est une **hypothèse méthodologique, pas un fait vérifié** -- si un
résultat futur semble sensible à ce choix, refaire le test avec 10 ou 15
jours ouvrables pour voir si la conclusion change, plutôt que de faire
confiance à 5 jours sans le questionner. Voir `AVAILABILITY_LAG_BDAYS`
dans le code.

**Piège rencontré en construisant cette jointure (mineur, corrigé
immédiatement, mais à surveiller)** : `bot/earnings_data.py` laisse
`fiscal_date_ending` comme chaîne de caractères brute (jamais convertie en
date), alors que le nouveau module la convertit correctement -- une
jointure entre les deux a d'abord silencieusement retourné 0 lignes au
lieu d'une erreur explicite. Détecté seulement parce que le résultat (0
lignes) était visiblement absurde. **Leçon générale** : après toute
jointure inter-module, vérifier explicitement le compte de lignes obtenu
ET les dtypes des deux côtés -- un mismatch de type peut échouer
silencieusement (0 lignes) plutôt que lever une erreur.

**Autre point à garder en tête pour la construction du signal** : le
debt/equity et le P/B de JPM (secteur financier, échantillon vérifié)
n'ont pas la même échelle qu'une entreprise industrielle -- les banques
sont structurellement très leviérisées. Un classement value/qualité
brut, non normalisé par secteur, biaiserait mécaniquement contre/pour
certains secteurs entiers plutôt que de comparer des entreprises
comparables. À traiter explicitement lors de la construction du score
(normalisation sectorielle), pas juste au moment du reporting.
