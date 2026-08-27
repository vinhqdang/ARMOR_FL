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
