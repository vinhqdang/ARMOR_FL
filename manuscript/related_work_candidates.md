# Related-work / baseline candidates for ARMOR-FL

Compiled 2026-08-27 via web research (general-purpose agent, ~25 targeted
searches, DOIs cross-checked against Crossref/Semantic Scholar where
possible). Supplements FedSE-1DSqueezeNet's own Table 3 (DIS-IoT, OE-IDS,
Res-TranBiLSTM, ABCNN-IDS / ABC-BWO-ConvLSTM, TCN+LSTM, CNN, ConvGRU /
Two-Step-IDS, LSTM, DL-BiLSTM, Att+CNN-LSTM) and Fig 6/7 FL baselines (FLAD,
NIDS-FGPA, FedDB) -- not a replacement for them.

**Before citing any entry marked "not confirmed" or "DOI not found" below,
verify directly** -- these were what was accessible to the research agent,
not a guarantee of accuracy. A citation-check pass close to submission is
advisable regardless, since this subfield is moving fast (several 2026-dated
venues below).

One candidate was deliberately excluded as a likely duplicate: Gheni &
Al-Yaseen, "Two-step data clustering for improved IDS using CICIoT2023"
(e-Prime, 2024, 10.1016/j.prime.2024.100673) -- name and dataset match
"Two-Step-IDS" already in FedSE-1DSqueezeNet's Table 3 closely enough to
likely be the same paper.

## Table 1 -- Robust / Byzantine-defense FL-IDS papers (closest prior art to ARMOR-FL)

| # | Title | Authors | Year | Venue | DOI | Dataset(s) | Attack(s) simulated | Headline metric |
|---|---|---|---|---|---|---|---|---|
| 1 | Dependable federated learning for IoT intrusion detection against poisoning attacks | Yang, He, Wang, Qu, Zhang | 2023 | Computers & Security | 10.1016/j.cose.2023.103381 | CIC-IDS-2017 | Label-flipping (loss-scoring + Manhattan-similarity clustering) | Accuracy 84.3% -> 97.1% after defense |
| 2 | Personalized federated learning-based intrusion detection system: Poisoning attack and defense (pFL-IDS) | Thein, Shiraishi, Morii | 2023/24 | Future Generation Computer Systems | 10.1016/j.future.2023.10.005 | UNSW-NB15 + CICIDS2018 | Data/model poisoning, non-IID clients | not confirmed |
| 3 | Investigating the Impact of Label-flipping Attacks against Federated Learning for Collaborative IDS | Lavaur, Busnel, Autrel | 2025 | Computers & Security | 10.1016/j.cose.2025.104462 | CIC-IDS2017 | Systematic label-flipping study | Baseline acc. 97.85%, F1 0.94, FPR 1.78%; up to -10.53% acc. under attack |
| 4 | GRAF-IDS: graph-based clustering as aggregation for federated IDS in IoT network | Rezaei, Taheri, Shojafar, Foh | 2025 | Neural Computing and Applications, 37:18401-18423 | 10.1007/s00521-025-11385-1 | N-BaIoT2018, UNSW-NB15, CICIoV2024, CSE-CIC-IDS2018 | Label-flipping / noise label-flipping, 25%/30% poisoning | ~99% acc. (no-attack baseline); Krum-enhanced graph-clustering aggregation |
| 5 | Federated RNN for IDS in IoT Environment Under Adversarial Attack | Rezaei, Taheri, Jordanov, Shojafar | 2025 | J. Network and Systems Management, 33:82 | 10.1007/s10922-025-09963-8 | N-BaIoT2018, UNSW-NB15, CICIoV2024, CSE-CIC-IDS2018 | Novel MPLFA (margin-points label-flipping), 25/30/35% | MPLFA cuts acc. ~54%; SNCOC defense recovers ~47% |
| 6 | WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in FL for NIDS | Sameera K.M., Vinod P., Rocha, Rehiman K.A., Conti | 2025 | arXiv 2504.04367 (venue TBC) | DOI not found | CIC-Darknet2020 + CSE-CIC-IDS2018 | Poisoning, non-IID | Target-class recall +up to 70%; global F1 +1-14% vs SOTA defenses |
| 7 | Defending FL-based IDS against model poisoning attacks (D3-FLIDS) | Zukaib, Cui, Niaz, Dong, Zheng, Din | 2026 | Neurocomputing | 10.1016/j.neucom.2026.134208 | CICIDS2018, UNSW-NB15, WUSTL-IIoT-2021, MedBIoT, CIC-IoMT2024 | Model poisoning / backdoor (gradient-projection + Isolation Forest) | "substantial" improvement, exact % not confirmed |
| 8 | FedDBC: density-based defense against collusion attacks in federated IDS for IoT | Latif, Ma, Ahmad | 2026 | Computer Networks | 10.1016/j.comnet.2026.112497 | CSE-CIC-IDS2018 + TON_IoT | Sign-flip, label-flip, attenuated mimicry, multi-cluster collusion (<=30%) | F1 0.997 (IID CICIDS2018), 0.808 (non-IID TON_IoT), worst-case 0.665 |
| 9 | Trust-aware blockchain-assisted framework for federated IDS in SDNs (BLOC-SHIELD) | Ram, Chakraborty | 2026 | Computer Networks | 10.1016/j.comnet.2026.112360 | CICIDS2018 + UNSW-NB15 | Adversarial/compromised SDN controllers, non-IID | "consistently higher F1 than standard FL", exact numbers not confirmed |
| 10 | Trust-aware federated IDS framework for privacy-preserving smart city IoT | Houichi, Jaidi, Bouhoula | 2026 | Computer Networks | 10.1016/j.comnet.2026.112616 | CICIDS2017 + CSE-CIC-IDS2018 | 20% and 40% malicious clients, non-IID | not confirmed |
| 11 | Fair client selection and encrypted aggregation: FL framework for IDS in resource-constrained networks | Akter, Naizheng, Ullah, Singh, Singh, Iqbal | 2026 | **Cluster Computing**, 29:164 | 10.1007/s10586-025-05905-w | CICIDS2017 | Not attack-focused -- EMD-based fair client selection + FHE + SMOTE | not confirmed -- **published in our target journal, worth checking directly for editorial-fit framing** |
| 12 | FLEX-IDS: secure, explainable federated IDS for Edge-of-Things under adversarial conditions | Chowdhury, Nur, Islam, Alam, Karim, Shah | 2026 | Computers and Electrical Engineering | 10.1016/j.compeleceng.2025.110827 | CICIDS2017, UNSW-NB15, ToN-IoT | 30% malicious participation | Accuracy up to 98.05%, mean F1 0.86 under 30% malicious clients |

## Table 2 -- Recent general SOTA IDS papers (2024-2026), beyond FedSE-1DSqueezeNet's own baselines

| # | Title | Authors | Year | Venue | DOI | Dataset(s) | Headline metric |
|---|---|---|---|---|---|---|---|
| 1 | Few-shot NID method based on multi-domain fusion and cross-attention | Xu, Li, Liu, Yang, Shen, Tong | 2025 | PLOS One | 10.1371/journal.pone.0327161 | CICIDS2017 + CICIDS2018 | Acc 99.03%/98.64% (10-shot); cross-domain generalization >95.13% |
| 2 | Intrusion Detection Method Based on Transformer and CNN-BiLSTM in IoT | Zhang, Li, Wang, Zhang | 2025 | Sensors, 25(9):2725 | 10.3390/s25092725 | CIC-IDS2017 + BoT-IoT | Acc 99.80% (CICIDS2017), 97.95% (BoT-IoT) |
| 3 | Comparative Study of Transformer vs CNN-LSTM Hybrid for NID Using CSE-CIC-IDS2018 | Palani, S, Durairaj, Praveen S, Victor | 2026 | Turkish Journal of Engineering, 10(3) | 10.31127/tuje.1843251 | CSE-CIC-IDS2018 | Transformer acc 99.42% (AUC 99.58%); CNN-LSTM acc 99.18% |
| 4 | HAE-HRL: novel autoencoder + hybrid enhanced LSTM-CNN residual network NIDS | Xue, Kang, Yu | 2025 | Computers & Security, 151 | 10.1016/j.cose.2025.104328 | NSL-KDD, UNSW-NB15, CIC-IDS-2018 | Binary acc 96.7% on CICIDS2018 |
| 5 | Multi-Class IDS for DDoS Attacks in IoT Networks Using Deep Learning and Transformers | Wahab, Sultana, Tariq, Mujahid, Khan, Mylonas | 2025 | Sensors, 25(15):4845 | 10.3390/s25154845 | CICIoT2023 | DNN 99.2%, CNN 99.0%, Transformer 98.8% (binary) |
| 6 | Resilient IoT IDS using hybrid feature selection and explainable ensemble learning | Wakili, Bakkali | 2025 | Results in Engineering | 10.1016/j.rineng.2025.107392 | CICIoT2023 | Acc 99.16%, macro-F1 73.79%, MCC 0.9908 |
| 7 | Enhancing NID performance using GANs | Zhao, Fok, Thing | 2024 | Computers & Security | 10.1016/j.cose.2024.104005 | CIC-IDS2017 | GAN-augmented NIDS beats non-augmented baseline (exact % not confirmed) |
| 8 | Lightweight IDS using multiscale attention 1D CNN for large-scale IoT | Sireesha, Kumar | 2026 | Frontiers in Artificial Intelligence | 10.3389/frai.2026.1857306 | large-scale IoT traffic -- likely CICIoT2023, **not explicitly confirmed, verify before citing** | Acc 99.67%, F1 99.29% |

## Gap analysis (novelty framing for ARMOR-FL)

1. **Dataset coverage gap**: essentially all Byzantine/poisoning-defense
   FL-IDS work with quantified robustness numbers targets CICIDS2017 or
   CSE-CIC-IDS2018 (or non-CIC IoT sets: N-BaIoT, TON_IoT, UNSW-NB15). No
   robust-aggregation FL-IDS paper was found with reported attack-robustness
   numbers on **CICIoT2023** specifically -- ARMOR-FL would be a first mover
   there.
2. **Statistical-guarantee gap**: almost none of the Table 1 defenses give
   formal statistical guarantees (confidence sequences, false-exclusion
   bounds, anytime validity) -- they're empirical clustering/distance/
   reputation heuristics (Krum-variants, density/DBSCAN, trust scores). This
   is ARMOR-FL's strongest methodological contrast point.
3. **Drift-vs-attack gap**: essentially absent from prior work -- every
   defense above treats deviation from consensus as inherently malicious,
   with no mechanism to distinguish a legitimately non-IID/drifting client
   from an adversarial one. This is exactly ARMOR-FL's second e-process
   contribution.
4. **Attack-type coverage skews toward label-flipping and generic
   sign-flipping**; collusion-aware / adaptive / stealthy poisoning is
   comparatively under-tested (FedDBC's multi-cluster collusion case is the
   notable exception -- worth a close read).
5. Several 2026-dated venues (Computer Networks, Neurocomputing) confirm this
   is a fast-moving, still-consolidating subfield -- re-run a citation check
   close to submission.

## Table 3 -- Author self-citations (curated 2026-08-31 from ORCID 0000-0002-3877-8024)

The author (Quang-Vinh Dang) has 82 works on ORCID; most are outside this
manuscript's scope (malware/fraud/recommendation/NLP applications, general
ML-for-IDS book chapters on datasets ARMOR-FL doesn't use). This table is a
**curated subset actually on-topic** -- moderate self-citation per COPE
guidance, not exhaustive coverage. All bibliographic fields below verified
directly against the Crossref API, not just the ORCID listing.

| # | Title | Venue | Year | DOI | Why it's relevant here |
|---|---|---|---|---|---|
| 1 | FORTRESS-FL: Byzantine-robust and privacy-preserving federated orchestration for next-generation networks | Array, 29:100680 | 2026 | 10.1016/j.array.2026.100680 | **Closest prior self-citation** -- Byzantine-robust FL, but no anytime-valid/e-process guarantee or drift-vs-attack decoupling. Needs explicit differentiation, not just a citation. |
| 2 | ST-FedXIDS: spatiotemporal federated explainable IDS with drift-adaptive graph learning for high-density public WiFi | Int. J. Advances in Signal and Image Sciences, 12(1):873-886 | 2026 | 10.29284/zcrk2p65 | Drift-adaptation precedent -- cite for the drift half of ARMOR-FL's contribution only. |
| 3 | Utilizing uncertainty measures to improve the performance of intrusion detection systems | SN Computer Science, 7(4):344 | 2026 | 10.1007/s42979-026-04923-8 | Statistical-testing framing grounding for the e-process approach. |
| 4 | Detecting IoT malware using federated learning | LNNS, *Data Science and Applications*, 73-83 | 2024 | 10.1007/978-981-99-7862-5_6 | Author's own FL + IoT-security precedent (malware, not network IDS, but same FL-for-security framing). |
| 5 | Kernel methods for conformal prediction to detect botnets | LNNS, *AI: Theory and Applications*, 29-41 | 2024 | 10.1007/978-981-99-8476-3_3 | Conformal-prediction lineage -- distribution-free guarantees, same statistical family as ARMOR-FL's e-processes. |
| 6 | Using machine learning for intrusion detection systems | Computing and Informatics, 41(1):12-33 | 2022 | 10.31577/cai_2022_1_12 | General ML-for-IDS foundational self-citation, establishes track record. |
| 7 | Learning to transfer knowledge between datasets to enhance intrusion detection systems | LNEE, *Computational Intelligence*, 39-46 | 2023 | 10.1007/978-981-19-7346-8_4 | Cross-dataset generalization precedent -- relevant to ARMOR-FL's 3-dataset (CICIDS2017/2018/CICIoT2023) design. |
| 8 | Intrusion detection in Internet of Things environment | *Advances in Digital Science -- ADS 2022*, 26-34 | 2022 | 10.33847/978-5-6048575-0-2_2 | IoT-specific IDS precedent, relevant to CICIoT2023. |
| 9 | Improving the performance of the intrusion detection systems by the machine learning explainability | Int. J. Web Information Systems, 17(5):537-555 | 2021 | 10.1108/ijwis-03-2021-0022 | Only relevant if the Discussion section addresses interpretability of ARMOR-FL's trust weights -- optional. |

All 9 are now real BibTeX entries in `manuscript/sn-bibliography.bib` (keys
`dang2026fortressfl`, `dang2026stfedxids`, `dang2026uncertainty`,
`dang2024iotmalwarefl`, `dang2024conformalbotnet`, `dang2022mlids`,
`dang2023transferids`, `dang2022iotids`, `dang2021explainableids`).

## Table 4 -- Cluster Computing journal papers (target-venue fit)

Searched specifically for **Cluster Computing** (this manuscript's target
journal) papers on FL/Byzantine-robustness/IDS, beyond Table 1's entry #11
(Akter et al.). All verified directly against the Crossref API (title,
authors, volume/issue, DOI) -- not just taken from search-result snippets.

| # | Title | Authors | Vol/Issue | DOI | Relevance |
|---|---|---|---|---|---|
| 1 | RPCFL: a byzantine-robust and privacy-preserving clustered federated learning framework | Chen, Tan, Zhong, Wang, Fan, Weng | 29(2):125 (2026) | 10.1007/s10586-025-05900-1 | **Directly on-topic** -- Byzantine-robust + privacy-preserving clustered FL, published in the target journal. Closest Table-1-style comparator that also happens to be in-venue; worth a careful differentiation paragraph (SMPC/clustering-based robustness vs. ARMOR-FL's anytime-valid e-process approach). |
| 2 | Federated learning in intrusion detection: advancements, applications, and future directions | Buyuktanir, Altinkaya, Karatas Baydogmus, Yildiz | 28(7):473 (2025) | 10.1007/s10586-025-05325-w | Survey/review paper -- good broad framing citation for the Introduction, establishes the target journal already publishes FL-IDS survey work. |
| 3 | Lightweight LLM-based hierarchical federated learning for B5G-enabled IoT intrusion detection networks | Mohawesh, Al-Obiedollah, Maqsood, Bany Salameh | 29(3):195 (2026) | 10.1007/s10586-026-05939-8 | FL + IoT IDS in-venue precedent; not attack/robustness-focused, so a Related Work citation rather than a head-to-head comparator. |
| 4 | Enhancing data privacy in cyber-physical systems with federated learning-based intrusion detection | Bella, Guezzaz, Ravi, Benkirane, Mohy-eddine, Azrour, Ennajar | 29(6):346 (2026) | 10.1007/s10586-026-06001-3 | FL-IDS in-venue precedent (CPS domain); privacy-framed, not robustness-framed. |
| 5 | Improving IoT security through federated deep Q-learning with realistic traffic modelling | Godavarthi, Mahesh, Jithendar, Mohanty, Dash | 29(6):399 (2026) | 10.1007/s10586-026-06160-3 | FL + IoT security in-venue precedent (RL-based, not supervised IDS) -- weakest fit of the five, include only if Related Work needs venue-breadth over topical precision. |

All 5 are now real BibTeX entries in `manuscript/sn-bibliography.bib` (keys
`chen2026rpcfl`, `buyuktanir2025flidssurvey`, `mohawesh2026llmhfl`,
`bella2026cpsfedids`, `godavarthi2026fedqlearning`). Entry #1 (RPCFL) is the
one worth the most attention -- it's the only other Byzantine-robust FL
paper found in the target journal itself, so reviewers may expect it to be
addressed explicitly.
