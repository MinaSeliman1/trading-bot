# ORB-fade 15-minute intraday strategy exploration log

Stratégie : "Opening Range Fade" -- pour chaque titre, chaque jour, la
première bougie de 15 minutes après l'ouverture définit un range
[orb_low, orb_high]. Cassure au-dessus -> short (fade) ; cassure
en-dessous -> long (fade). Stop = 1 largeur de range au-delà de l'entrée,
target = TP_MULT largeurs de range. Une position max par titre par jour,
clôture en fin de journée si ni stop ni target touché.

Univers : SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, CAT, XOM, JNJ.
Split walk-forward 60/40 chronologique (un seul cutoff global, pas par
titre). Monte Carlo par blocs (bloc=5 jours) sur les rendements
journaliers agrégés, pour respecter l'autocorrélation (voir
lessons_learned.md, piège #6). Règles d'arrêt strictes définies dans
`bot_intraday_15min/orb_fade_backtest.py` (docstring) : pas de câblage
dans `bot/main.py` ni paper trading quel que soit le résultat ; REJET si
Sharpe OOS < 0.5, MC_DD95 > 20%, ou retournement de signe IS/OOS ; pas de
retouche des paramètres pour forcer un résultat positif.

## Run -- 2026-07-21T00:37:03.274579+00:00

- Feed: sip (see module docstring: IEX floor for minute bars verified ~Jan 2021 live, SIP verified back to Jan 2016 -- same account-level floor as the daily-bar SIP depth already established for PEAD/value-quality)
- Primary window: 2024-07-21 to 2026-07-21 (730 days), 60/40 chronological split, universe: ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'CAT', 'XOM', 'JNJ']

**Bug trouvé et corrigé pendant ce run** : le premier essai (horodatage
ci-dessus) a crashé sur `end=now()` exact avec le flux SIP -- erreur API
réelle : `"subscription does not permit querying recent SIP data"`.
Bisection en direct : 0h en arrière -> erreur, 1h en arrière -> OK. C'est
un simple embargo temps-réel (même nature que `DATA_EMBARGO_MINUTES=16`
déjà utilisé ailleurs dans le projet pour IEX), PAS une restriction de
profondeur historique -- ne remet pas en cause le plancher SIP de janvier
2016 déjà établi pour les barres journalières. Corrigé en utilisant `now -
24h` au lieu de `now()` exact (marge large, sans incidence sur un backtest
de 730 jours). Relancé ci-dessous.

## Run -- 2026-07-20T00:38:17.975549+00:00

- Feed: sip (see module docstring: IEX floor for minute bars verified ~Jan 2021 live, SIP verified back to Jan 2016 -- same account-level floor as the daily-bar SIP depth already established for PEAD/value-quality)
- Primary window: 2024-07-20 to 2026-07-20 (730 days), 60/40 chronological split, universe: ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'CAT', 'XOM', 'JNJ']
- Split cutoff: 2025-09-30 (train: 2024-07-22 to 2025-09-30, test: 2025-09-30 to 2026-07-17)

### Per-TP-multiple results (all 3 reported independently -- no in-sample argmax selection, per lessons_learned.md piège #4)

- **TP=1.0x**: IS n=3555 sharpe=-0.844 | OOS n=2351 sharpe=0.422 MC_DD95=25.9% | **REJECTED** (OOS Sharpe 0.422 < 0.5; MC_DD95 25.9% > 20%; sign flip: IS avg_ret=-0.00018, OOS avg_ret=0.00009)
- **TP=1.5x**: IS n=3555 sharpe=-0.816 | OOS n=2351 sharpe=0.550 MC_DD95=27.4% | **REJECTED** (MC_DD95 27.4% > 20%; sign flip: IS avg_ret=-0.00020, OOS avg_ret=0.00012)
- **TP=2.0x**: IS n=3555 sharpe=-1.037 | OOS n=2351 sharpe=0.589 MC_DD95=28.9% | **REJECTED** (MC_DD95 28.9% > 20%; sign flip: IS avg_ret=-0.00027, OOS avg_ret=0.00014)

### Cost sensitivity (OOS only, best-behaved TP multiple by OOS Sharpe among non-rejected, or TP=1.0x if all rejected -- reported for transparency, NOT used to pick a winner)

- 5.0bps: OOS sharpe=-1.764 avg_ret/trade=-0.00036
- 10.0bps: OOS sharpe=-4.193 avg_ret/trade=-0.00086
- 15.0bps: OOS sharpe=-6.622 avg_ret/trade=-0.00136

### PRIMARY VERDICT

**REJECTED.** Every TP multiple tested ([1.0, 1.5, 2.0]) triggers at least one stop-rule criterion on the primary 60/40 walk-forward split. Per the stop rules given: not retuning parameters to force a pass. This strategy, as specified, does not clear this project's bar.

### Data coverage limitation (as specified)

Alpaca's IEX feed (this project's default) has no reliable individual-stock minute-bar data before approximately January 2021 (verified live: zero rows for SPY/AAPL on sampled weekdays in 2018-2020, real data from Jan 2021 onward) -- so on IEX alone, the 2018 correction is NOT reachable for intraday validation, and the primary window above (last 730 days) cannot by itself constitute multi-regime validation.

## Bonus regime check: Q4 2018 correction (2018-08-01 to 2019-02-01) -- SIP-verified reachable

- Not part of the primary 60/40 walk-forward or its stop-rule verdict above. Included because live verification (see module docstring) showed SIP reaches back to Jan 2016 for minute bars, same as this project's established daily-bar SIP floor -- so unlike the IEX-only assumption in the original file, this specific regime IS testable, and testing it costs nothing extra once SIP is already being used for the primary run.
- TP=1.0x: n=1504 sharpe=-2.901 MC_DD95=49.0%
- TP=1.5x: n=1504 sharpe=-3.726 MC_DD95=60.0%
- TP=2.0x: n=1504 sharpe=-3.587 MC_DD95=60.7%

---

## SYNTHÈSE FINALE -- ORB-fade REJETÉ

**Verdict : REJETÉ, sans ambiguïté, sur les deux fenêtres testées.**

- **Fenêtre primaire (60/40, 2024-07 à 2026-07)** : les 3 multiples de TP
  échouent chacun au moins un critère (Sharpe OOS<0.5 pour TP=1.0x ;
  MC_DD95>20% pour les 3 ; retournement de signe IS/OOS pour les 3 -- IS
  fortement négatif partout, OOS faiblement positif, signature typique de
  bruit plutôt que d'un edge réel qui se degraderait dans le sens inverse).
- **Sensibilité aux coûts** : à 5/10/15bps (convention standard du projet,
  contre 0.5bps codé à l'origine dans le fichier), le Sharpe OOS s'effondre
  à -1.76 / -4.19 / -6.62 -- la stratégie n'a aucune marge face à des coûts
  de transaction réalistes, même modestes.
- **Vérification bonus (T4 2018, via SIP)** : encore plus négatif que la
  fenêtre primaire (Sharpe -2.9 à -3.7, MC_DD95 49-61%) -- renforce le rejet
  plutôt que de le contredire, donc pas de cas "passe sur un régime mais pas
  l'autre" nécessitant un arrêt pour validation utilisateur : c'est un rejet
  net sur les deux régimes testés.

**Aucune retouche de paramètres n'a été tentée pour forcer un résultat
positif**, conformément à la règle d'arrêt et à la leçon #4 de
`lessons_learned.md`.

**Bug trouvé et corrigé en cours de route** : le flux SIP a un embargo
temps réel (~1h, testé en direct) sur ce compte -- distinct du plancher de
profondeur historique (janvier 2016, déjà établi). Corrigé en utilisant
`now - 24h` plutôt que `now()` exact. Documenté ici pour éviter de
retomber sur la même surprise dans un futur script utilisant SIP proche du
temps présent (ex. `run_value_quality_viability.py`, qui utilise déjà une
marge de 20 minutes -- suffisante si l'embargo réel est bien <1h comme
mesuré, mais à garder en tête).

**Limite de couverture de données (précisée comme demandé)** : le flux
IEX (celui que le bot live utiliserait par défaut) n'a aucune donnée
minute fiable par action avant environ janvier 2021 (vérifié en direct :
zéro ligne pour SPY/AAPL sur des jours ouvrés échantillonnés 2018-2020,
données réelles à partir de janvier 2021) -- donc SEUL sur IEX, la
correction de 2018 n'est pas accessible pour une validation intraday, et
la fenêtre primaire (derniers 730 jours) ne peut pas à elle seule
constituer une validation multi-régime.

**Écart avec l'hypothèse initiale, signalé explicitement** : le fichier
d'origine supposait qu'aucune validation sur 2018 n'était possible avec
Alpaca. C'est vrai pour IEX (le flux par défaut de ce projet, vérifié à
~janvier 2021) mais PAS pour SIP, qui couvre janvier 2016 -- déjà établi
pour les barres journalières dans PEAD/value-quality, et confirmé ici
valable aussi pour les barres minute. La vérification bonus T4-2018
ci-dessus a donc pu être faite, en plus de ce qui était demandé, sans coût
significatif puisque SIP était déjà utilisé pour la fenêtre primaire.
Résultat : ça ne change rien à la conclusion (rejet net dans les deux
cas) mais corrige la limite de couverture annoncée -- IEX seul est
effectivement limité à ~2021+, mais SIP ne l'est pas, contrairement à ce
qui avait été supposé au départ.

**Aucune règle d'arrêt supplémentaire déclenchée** : rien câblé dans
`bot/main.py`, aucun paper trading. Cette piste s'arrête ici, comme prévu
par la consigne en cas de rejet.
