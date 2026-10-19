# Rapport — First Candle Rule + FVG

**Date : 2026-10-19 · État : évaluation impossible dans cette session (aucune donnée de marché accessible).**

## A. Résumé exécutif

La stratégie n'a **pas** pu être évaluée. L'environnement d'exécution bloque l'accès à tous les fournisseurs de données
testés (Yahoo, Stooq, Alpaca, Polygon, Databento, Alpha Vantage, Tiingo, Dukascopy, Kaggle, Hugging Face : HTTP 403 du proxy),
et le dépôt ne contient aucune barre 1 minute réelle. Conformément à vos règles, **aucun résultat n'est fourni, estimé ou
simulé** : il n'existe aujourd'hui aucun win rate, profit factor, drawdown ou P&L pour SPY, QQQ, ES ou NQ.

Ce qui existe et est vérifié : un banc de test complet, exécutable sur vos fichiers CSV/Parquet, dont le comportement a été
contrôlé par 64 tests (scénarios calculés à la main, absence de lookahead, DST, calendrier, QC des données, P&L, sizing, invariants).
Contrôle négatif : sur des marches aléatoires avec coûts, le moteur donne une espérance négative (≈ −0,17 R pour SPY, −0,26 R pour ES,
~250 séances simulées) — il ne produit pas d'avantage artificiel. Ce n'est **pas** un résultat de marché.

## B. Résultats historiques

Aucun (voir A). Les tableaux seront produits par `python run_backtest.py ...` dans `results/<run>/report_auto.md`
(performance FULL / IS / OOS, risque, comportement, rejets, breakdowns, bootstrap, sensibilité), sans verdict automatique.

## C. Analyse du risque

Non évaluable sans données. Points structurels déjà identifiables **sans** chiffres de performance :

* Avec 0,5 % de risque sur 100 000 $ (= 500 $), un seul contrat dépasse le budget dès que le risque (stop + 1 tick de slippage + commissions) excède 500 $ : calcul `(ticks + 1) × valeur du tick + 4,50 $ > 500 $`, soit un stop ≳ 24,5 points sur NQ (5 $/tick) et ≳ 9,5 points sur ES (12,50 $/tick). Les signaux au-delà seront **rejetés** (jamais arrondis à 1 contrat), ce qui réduit l'échantillon ; le nombre de rejets est publié par le rapport.
* L'objectif à 3 R demande un win rate d'équilibre > 25 % avant coûts ; les coûts (≥ 2 ticks aller-retour) le relèvent d'autant plus que le stop est serré. Plus le stop est serré (ce qui peut arriver avec la variante A), plus 1 tick de slippage pèse en R ; la distribution réelle des distances de stop n'a pas pu être mesurée.
* Ce sont des raisons de **tester**, pas des conclusions.

## D. Robustesse

Prévue et implémentée, non exécutée sur données réelles : IS/OOS 70/30, tables annuelles et semestrielles, slippage 1/2/3/5 ticks,
durée de retest 5–30, fenêtre de confirmation 0–5, barre d'entrée testée ou non, levier ETF 1×, terciles de volatilité
(largeur d'OR, amplitude des 20 jours précédents), bootstrap par séance, concentration sur les meilleurs jours.

## E. Code source

`fcr_fvg_backtest/` : package Python `fcr_fvg/`, `run_backtest.py`, `pine/first_candle_fvg_strategy.pine` (Pine v6, **non compilé**),
`fetch/` (**non exécuté**), `tests/`. Détails et instructions : `README.md`.

## F. Données et reproductibilité

* Données utilisées : **aucune**. Provenance/période/qualité : à fournir par `qc/*.json` lors du premier vrai run.
* Fournisseurs recommandés : Alpaca SIP (SPY/QQQ, brut, ≥ 3 ans), Databento GLBX.MDP3 ohlcv-1m par contrat (ES/NQ). Yahoo (~30 jours) n'est qu'exploratoire.
* Paramètres principaux : `fcr_fvg/params.py` (valeurs par défaut = configuration principale, fixée a priori). Graine du bootstrap : 12345.
* Reproduire : voir `README.md` §0.

## G. Conclusion

Classification : **5 — Impossible à évaluer sérieusement** (données insuffisantes : aucune).

Cette classification décrit l'état de la preuve, **pas** la qualité de la stratégie. Elle n'implique ni qu'elle fonctionne ni
qu'elle échoue. Aucune recommandation de trading réel n'est possible, et un résultat positif sur un seul run ne suffirait pas à en
justifier une : il faudrait des effectifs suffisants (centaines de trades par instrument), un OOS positif après coûts avec un
intervalle de bootstrap excluant 0, une stabilité sur les périodes et sur le slippage, et une comparaison A/B sans sélection a posteriori.

Contexte (non probant pour cette stratégie) : une étude antérieure de ce dépôt (ORB-fade 15 min, `exploration_log_intraday_orb.md`) a
été rejetée, et son résultat s'effondrait dès que des coûts réalistes étaient appliqués — rappel que les coûts sont le facteur décisif en intraday.
