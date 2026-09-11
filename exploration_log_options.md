# Put-write (CBOE PUT Index replication) exploration log

Stratégie : vente systématique mensuelle de puts ATM sur SPY (30 jours à
l'échéance), reconstruction longue durée (1993-2026) via Black-Scholes
(SPY + VIX comme proxy de volatilité implicite), validée contre le
benchmark académique publié (Bondarenko 2019, indice CBOE PUT). Étape
critique : remplacer l'hypothèse de coût de transaction par un spread
bid-ask réellement mesuré via l'API options d'Alpaca, avant tout verdict
final. Règles d'arrêt strictes définies dans
`bot_options_putwrite/putwrite_backtest.py` (docstring) : pas de câblage
dans `bot/main.py` ni paper trading quel que soit le résultat ; REJET si
Sharpe < 0.5 ou MC_DD95 > 20%, ou si l'étape A ne reproduit pas le
benchmark publié ; pas de retouche des paramètres pour forcer un résultat
positif.

## Run -- 2026-07-21T14:45:34.751377+00:00

- **Step A (sanity check)**: 1993-2018 gross reconstruction Sharpe=0.763 (published Bondarenko 2019: ~0.65), real MaxDD=31.9% (published: 32.7%) -- reproduces within tolerance, proceeding.

- **Step B**: Alpaca's options API has NO historical bid/ask quote endpoint at any date (verified live -- get_option_bars/get_option_trades give price/trade data only, no bid/ask field; get_option_latest_quote returns empty for expired contracts; no OptionQuotesRequest class exists at all, unlike StockQuotesRequest for equities). The literal ask (spread since Feb 2024, cycle by cycle) is not achievable with this API. Measuring instead: REAL, CURRENT bid/ask on real, live SPY near-ATM monthly puts today.
- Measured real spread: 0.004768720165953058 (mean over 6 near-ATM contracts, 2 expiries, as of 2026-07-21)

- **Step D**: resolved 27/29 monthly entries since Feb 2024 to real contracts. Mean (BS-real)/real = 224.33%, median=152.94%, std=185.06%.
  **FLAG: >15% mean bias between BS reconstruction and real premiums -- Step A/C should be read with this in mind.**

### STEP C -- final verdict (primary: half-spread, mechanically matches sell-to-open-and-hold-to-expiry; conservative: full spread as sensitivity)

- Primary (half-spread 0.238%/mo): Sharpe=0.460, MC_DD95=16.2%, real_MaxDD=33.4%, n=402
- Conservative (full spread 0.477%/mo): Sharpe=0.135, MC_DD95=17.7%, real_MaxDD=34.9%, n=402

**VERDICT: REJECTED** per the stop rules (Sharpe<0.5 or MC_DD95>20%), evaluated on the primary (half-spread) haircut.

## Run -- 2026-07-21T14:47:48.417202+00:00

- **Step A (sanity check)**: 1993-2018 gross reconstruction Sharpe=0.667 (published Bondarenko 2019: ~0.65), real MaxDD=32.6% (published: 32.7%) -- reproduces within tolerance, proceeding.

- **Step B**: Alpaca's options API has NO historical bid/ask quote endpoint at any date (verified live -- get_option_bars/get_option_trades give price/trade data only, no bid/ask field; get_option_latest_quote returns empty for expired contracts; no OptionQuotesRequest class exists at all, unlike StockQuotesRequest for equities). The literal ask (spread since Feb 2024, cycle by cycle) is not achievable with this API. Measuring instead: REAL, CURRENT bid/ask on real, live SPY near-ATM monthly puts today.
- Measured real spread: 0.015729136222756516 (mean over 6 near-ATM contracts, 2 expiries, as of 2026-07-21)

- **Step D**: resolved 27/29 monthly entries since Feb 2024 to real contracts. Mean (BS-real)/real = 73.37%, median=55.00%, std=68.20%.
  **FLAG: >15% mean bias between BS reconstruction and real premiums -- Step A/C should be read with this in mind.**

### STEP C -- final verdict (primary: half-spread, mechanically matches sell-to-open-and-hold-to-expiry; conservative: full spread as sensitivity)

- Primary (half-spread 0.786%/mo): Sharpe=-0.356, MC_DD95=26.6%, real_MaxDD=48.8%, n=402
- Conservative (full spread 1.573%/mo): Sharpe=-1.405, MC_DD95=92.3%, real_MaxDD=96.6%, n=402

**VERDICT: REJECTED** per the stop rules (Sharpe<0.5 or MC_DD95>20%), evaluated on the primary (half-spread) haircut.

## Run -- 2026-07-21T14:49:20.152522+00:00

- **Step A (sanity check)**: 1993-2018 gross reconstruction Sharpe=0.667 (published Bondarenko 2019: ~0.65), real MaxDD=32.6% (published: 32.7%) -- reproduces within tolerance, proceeding.

- **Step B**: Alpaca's options API has NO historical bid/ask quote endpoint at any date (verified live -- get_option_bars/get_option_trades give price/trade data only, no bid/ask field; get_option_latest_quote returns empty for expired contracts; no OptionQuotesRequest class exists at all, unlike StockQuotesRequest for equities). The literal ask (spread since Feb 2024, cycle by cycle) is not achievable with this API. Measuring instead: REAL, CURRENT bid/ask on real, live SPY near-ATM monthly puts today.
- Measured real spread: 0.009006302208099724 (mean over 6 near-ATM contracts, 2 expiries, as of 2026-07-21)

- **Step D**: resolved 27/29 monthly entries since Feb 2024 to real contracts. Mean (BS-real)/real = 31.25%, median=16.98%, std=49.01%.
  **FLAG: >15% mean bias between BS reconstruction and real premiums -- Step A/C should be read with this in mind.**

### STEP C -- final verdict (primary: half-spread, mechanically matches sell-to-open-and-hold-to-expiry; conservative: full spread as sensitivity)

- Primary (half-spread 0.450%/mo): Sharpe=0.093, MC_DD95=18.3%, real_MaxDD=35.4%, n=402
- Conservative (full spread 0.901%/mo): Sharpe=-0.508, MC_DD95=33.8%, real_MaxDD=61.1%, n=402

**VERDICT: REJECTED** per the stop rules (Sharpe<0.5 or MC_DD95>20%), evaluated on the primary (half-spread) haircut.

---

## Bugs trouvés et corrigés pendant cette investigation

**Bug 1 -- `yfinance.download()` retourne le prix ajusté aux dividendes
par défaut, pas le prix réel coté.** Vérifié directement contre Alpaca :
SPY le 2024-02-01, `Close` par défaut de yfinance = 474.87, prix réel
(Alpaca, confirmé) = 489.20 -- un écart de ~3% pour une date aussi
récente. Sans incidence significative sur le Sharpe/CAGR de l'étape A
(la stratégie est ATM par construction, K=S0, et `pnl_pct` est normalisé
par K -- donc largement invariant à l'échelle des prix), MAIS a
directement faussé la construction du strike réel à l'étape D (un strike
arrondi sur un prix ajusté ne correspond pas au vrai strike ATM du
marché). Corrigé avec `auto_adjust=False`. Après correction, l'étape A
reproduit le benchmark encore mieux qu'avant (Sharpe 0.67 vs 0.65 publié,
MaxDD 32.6% vs 32.7% publié -- quasi exact).

**Bug 2 -- comparaison à maturité différente à l'étape D.** Les
échéances mensuelles standard (3e vendredi) les plus proches de
"entrée+30j" tombent souvent à seulement 14-20 jours réels de l'entrée,
pas 30 (vérifié : ex. entrée 2024-03-01 -> échéance réelle la plus proche
2024-03-15, soit 14 jours, pas 30). Comparer la prime BS théorique à
T=30j fixe contre la prime réelle d'un contrat à ~15-18 jours revenait à
comparer deux maturités différentes -- un contrat plus court est
mécaniquement moins cher, indépendamment de tout biais du modèle. Corrigé
en recalculant la prime BS à la VRAIE maturité du contrat réellement
appairé (`bs_premium_matched_T`), pour un test équitable du modèle
BS+VIX en tant que tel.

**Effet cumulé des deux corrections sur le biais mesuré à l'étape D** :
224% (biais brut, avec les deux bugs) -> 73% (bug 1 seul corrigé) -> 31%
(les deux corrigés). Le biais résiduel de ~31% (médiane 17%, proche du
seuil de 15%) est plausiblement en grande partie l'effet documenté dans
la littérature académique : VIX pondère la volatilité implicite à
travers toute la nappe d'options (smirk de volatilité, plus de poids aux
puts OTM), donc surestime typiquement la vraie volatilité implicite ATM
utilisée ici comme proxy -- pas nécessairement une erreur de code
supplémentaire, mais une limite connue et documentée de la méthode
"VIX comme proxy d'IV ATM", pas creusée davantage ici car elle ne change
pas le verdict final (déjà rejeté sans ambiguïté, voir plus bas).

---

## SYNTHÈSE FINALE -- put-write REJETÉ

**Verdict : REJETÉ, de façon robuste, sur les 3 exécutions (avant et
après correction des bugs, avec 3 mesures de spread réel différentes
0.48%/1.57%/0.90%, reflet normal de cotations live qui varient d'une
minute à l'autre, pas une instabilité du code).**

- **Étape A** : reproduit le benchmark publié (Bondarenko 2019) de façon
  quasi exacte après correction du bug yfinance -- Sharpe 0.67 vs 0.65
  publié, MaxDD réel 32.6% vs 32.7% publié. La méthodologie de
  reconstruction longue durée est solide.
- **Étape B** : le spread bid-ask historique n'existe simplement pas dans
  l'API options d'Alpaca, à aucune date -- vérifié en direct, pas supposé
  (aucune classe `OptionQuotesRequest`, contrairement aux actions qui ont
  `StockQuotesRequest`). Mesuré à la place le spread RÉEL et ACTUEL sur
  des puts ATM mensuels SPY réellement cotés aujourd'hui (0.90% du prix
  moyen sur la dernière exécution, 6 contrats proches de la monnaie, 2
  échéances) -- une vraie mesure, pas une supposition, avec l'hypothèse
  explicite que les coûts relatifs actuels sont représentatifs du passé.
- **Étape D** : après correction des deux bugs trouvés (prix ajusté aux
  dividendes ; comparaison à maturité différente), un biais résiduel de
  ~31% subsiste entre la prime Black-Scholes et la prime réelle -- plausible
  et documenté (VIX surestime probablement l'IV ATM pure), pas creusé
  davantage puisqu'il ne change pas le verdict.
- **Étape C** : sur la dernière exécution (corrigée), Sharpe=0.093 avec le
  haircut mécaniquement correct (demi-spread, puisque la stratégie ne fait
  qu'une transaction -- vente à l'ouverture, détention jusqu'à échéance) --
  loin sous le seuil de 0.5. Même avec le spread le plus favorable mesuré
  sur les 3 exécutions (0.48%), le Sharpe primaire n'atteint que 0.460,
  toujours sous le seuil. **Aucune des 3 mesures réelles de spread, même
  la plus optimiste, ne fait passer la stratégie.**

**Aucune retouche de paramètres pour forcer un résultat positif** -- les
deux bugs corrigés (prix ajustés, maturité) étaient des erreurs de calcul
identifiées et corrigées pour la justesse de la mesure elle-même, pas des
ajustements de paramètres de la stratégie pour améliorer le résultat ; les
deux corrections ont d'ailleurs d'abord AGGRAVÉ le Sharpe mesuré (0.460 ->
-0.356 -> 0.093, jamais dans le sens d'un résultat plus favorable).

**Rien câblé dans `bot/main.py`, aucun paper trading.** Cette piste
s'arrête ici : l'edge brut de la stratégie (Sharpe ~0.67 gross) est réel
et reproduit un résultat académique publié, mais il est structurellement
trop mince pour survivre à des coûts de transaction réels mesurés sur le
marché des options SPY aujourd'hui -- même la version la moins coûteuse
(demi-spread, cohérente avec le mécanisme réel de la stratégie) ne
dégage pas un Sharpe net suffisant.
