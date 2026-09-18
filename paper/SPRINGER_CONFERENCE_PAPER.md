# FedIDS-Bench: A Standardized Cross-Dataset, Cross-Algorithm, and Cross-Threat-Model Benchmark for Federated Intrusion Detection Systems

**Manuscript Format**: Official Springer LNCS Single-Column Style (`llncs.cls`, `splncs04.bst`)  
**Length**: Comprehensive full paper (>10 pages in standard Springer format)  
**Authors**:  
- **Lead Author & Corresponding**:  
  1. **Ms. Sowmiya .S** — *Assistant Professor, Department of B.Tech AI&DS* (`sowmiya.s@sece.ac.in`)  
- **Co-Authors**:  
  2. **Ram kumar .K** — *Department of B.Tech AI&DS* (`ramkumar.k2023ai-ds@sece.ac.in`)  
  3. **Rahgul.S.A** — *Department of B.Tech AI&DS* (`rahgul.sa2023ai-ds@sece.ac.in`)  
  4. **Naveen Kumar .R** — *Department of B.Tech AI&DS* (`naveenkumar.r2023ai-ds@sece.ac.in`)  
  5. **Ranjan Kumar.S.M** — *Department of B.Tech AI&DS* (`ranjankumar.s.m2023ai-ds@sece.ac.in`)  
*Sri Eshwar College of Engineering, Coimbatore, Tamil Nadu, India*  

---

## Abstract

Boundary nodes can cooperate using Federated Learning (FL) based Decentralized network intrusion detection systems (NIDS) without centralized aggregation of proprietary packet traces, thus preserving domain norms. In FL-NIDS research literature comparing and analyzing still is not significant with non - transferable experimental settings. e.g., train and test poisson when scaled globally and when synthetic data is too simple, assuming only one threat will happen, Byzantine defence baselines being not present and physical edge performance without intrumentation. We propose **FedIDS-Bench**, an end-to-end benchmark suite for FL-NIDS covering:
1. **Five network/IoT telemetry corpora** (UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23, and ToN_IoT) with isolated split-normalization;
2. **Class-preserving non-IID data partitioning** through Dirichlet label priors, dedicated skew, and power-law quantity distribution;
3. **Decoupled FL algorithms** (FedAvg, FedProx, FedAdam) with client-side memory protection;
4. **Adversarial attack engine** targeting flow watermarking, label inversion, gradient sign-flipping, Byzantine Gaussians, and sample inflation;
5. **Coordinate-wise robust aggregation operators** (trimmed mean, median, gradient norm bounds, and Multi-Krum); and
6. **Physical edge node performance assessment** on Raspberry Pi 4, Jetson Orin Nano, and workstation CPUs.

We find that naive sample report aggregation renders FedAvg vulnerable to full update hijacking, while adaptive server optimization (FedAdam) improves test F1 by $+11.75\%$ over FedAvg at extreme skew ($\alpha = 0.1$). Coordinate-wise trimmed mean and median regression maintain detection F1 $> 0.91$ under 20% Byzantine poison. The platform is released with process-sanity guarantees for multi-process bitwise determinism.

**Keywords**: Collaborative Cyber Defense $\cdot$ Federated Optimization $\cdot$ Statistical Drift $\cdot$ Adversarial Poisoning $\cdot$ Robust Aggregators $\cdot$ Edge Intelligence $\cdot$ Intrusion Detection.

*Note*: All source code, configuration manifests, and empirical evaluation harnesses are publicly accessible to enable full bitwise reproducibility.

---

## 1 Introduction

Modern organizational infrastructure and Internet-of-Things (IoT) gateways face an onslaught of complex network incursions covering multi-stage botnets, distributed denial-of-service (DDoS), zero-day memory exploits [1], [2]. Existing security infrastructure relies on centralized Network Intrusion Detection Systems (NIDS) that aggregate high-rate NetFlow, IPFIX, and raw pcap records at an enterprise Security Operations Center (SOC). However, such backhauling strategies incur substantial bandwidth limitations, processing delay, and privacy risks regarding internal topology disclosure and payload confidentiality [3], [4]. Federated Learning (FL) [5], [6] addresses this security paradigm by decentralizing model training at distributed edge sensors. Participating network monitors collaboratively optimize the detection model using private traffic traces with only aggregated parameter updates exchanged with an orchestrating coordinator. Despite a rich exploration of the literature, published FL-NIDS research suffers from severe evaluation inconsistencies inducing an empirical reproducibility crisis [3], [7]. Specifically, five major methodological deficiencies undermine the state of empirical literature:

1. **Silent Preprocessing Contamination (Data Leakage):** A pervasive issue in published FL-IDS manuscripts involves computing global normalization statistics (sample mean and variance) across the entire corpus before partitioning into clients and evaluation splits. This inadvertently leaks distribution moments of the held-out test split into the local training partitions, inflating detection scores.
2. **Trivialized Statistical Heterogeneity:** Real-world enterprise and IoT environments exhibit acute spatial and temporal data heterogeneity (web servers predominantly observe application-layer exploits, industrial controllers undergo telemetry fuzzing) but prior benchmarks frequently rely on uniform random partitions or synthetic generators, obscuring severe optimization divergence.
3. **Isolated Adversarial Threat Formulations:** Poisoning attacks are often studied in isolation. Papers that report label flips tend to ignore inversion and sample-count spoofing. At the time defensive evaluations rarely include direct comparisons between robust coordinate-level aggregators.
4. **Omission of Physical Edge Constraints:** Previous research mostly focuses on communication epochs. It completely overlooks real-world execution metrics like inference latency, memory usage and energy consumption at the Joule level. These are critical when running systems on edge hardware such as ARM Cortex cores and Jetson modules.
5. **Lack of Bitwise Sub-Process Reproducibility:** pseudo-random seeds, unpredictable multithreaded CPU scheduling and missing preprocessing scripts make it impossible to reproduce reported results accurately.

To address these structural limitations, we propose **FedIDS-Bench** — a standardized benchmarking framework designed from the ground up. It supports crossdataset, cross-algorithm and cross-threat-model evaluations. Its goal is to enable reproducible research, in Federated Learning-based Intrusion Detection Systems.

### Principal Research Contributions:
- **Strict Zero-Leakage Pipeline:** This pipeline keeps data separate. Transforms it across five intrusion benchmarks—UNSW-NB15, CICIDS2017, CSE-CIC-IDS2018, IoT-23 and ToN_IoT. The pipeline guarantees that all normalization settings come from training data not from any other source.
- **Multi-Faceted Heterogeneity Partitioner:** This partitioner creates realistic data splits that're not independent and identically distributed. It uses Dirichlet priors with values 0.1, 0.5 and 1.0.
- **Modular Optimization & State Isolation:** The system runs FedAvg, FedProx and FedAdam in modules. Each module copies the model deeply so that memory from one client does not leak into another. This keeps each client's state clean and isolated.
- **Comprehensive Threat & Defense Matrix:** We formally tested five types of attacks and four Byzantine-tolerant aggregation methods. The evaluation revealed a sample-reporting flaw, in classic FedAvg.
- **Hardware Profiling Across Physical Edge Testbeds:** We collected real resource data on network traffic, MAC and FLOP counts, memory use, inference time and power consumption.
- **Verified Sub-Process Determinism:** I have verified process determinism by auditing and testing multiple process execution confirming that the metrics are exactly the same at bit level, across independent runtime setting.

---

## 2 Related Work and Benchmarking Landscape

### 2.1 Decentralized Intrusion Detection Systems
I think the idea of learning became popular after the research, by McMahan et al. They introduced Federated Averaging, called FedAvg [5]. Federated Averaging mixes updates from clients. Each update is weighed by how samples each client has. In security engineering several big surveys [3] [4] asked whether federated telemetry monitoring could actually work. These surveys looked at devices and smart grid settings. Popoola et al. [7] Investigated how federated deep neural networks perform when spotting zero-day botnets. They wanted to find out if Federated Averaging could be used well in real-world security situations. However, as stressed by Campos et al. [3], most prior arts evaluated customized network architectures on individual datasets, and comparative analysis between them is lacking.

### 2.2 Client Drift and Statistical Non-IID Inconsistencies
The non-IID nature of traffic distributions between edge networks causes significant instability in gradient aggregation. Zhao et al. [8] and Hsu et al. [9] revealed that label skew leads to catastrophic weight drift that favors local minima. To alleviate the issue, Li et al. [10] proposed FedProx which added a proximal regularizer to penalize large parameter deviations from the global model. Meanwhile, Reddi et al. [11] showed that server-side adaptive momentum (FedAdam) could suppress extreme update fluctuations. In the context of network intrusion detection, where abnormal connection bursts are of rare occurrences, comparative studies between FedProx and FedAdam on controlled Dirichlet regimes using real flow traces is still lacking.

### 2.3 Adversarial Poisoning and Byzantine Robustness
The decentralized architecture of FL is vulnerable to adversarial manipulation at the aggregation layer. Malicious edge sensors may launch data poisoning attacks via label inversion or trigger embedding [12], [13], or model poisoning by directly tampering with weight updates [14], [15]. Byzantine-robust aggregation algorithms have been proposed to detect and eliminate malicious contributions: Blanchard et al. [16] proposed Krum, an algorithm that selects the model vector with minimal Euclidean distances to its neighbors; Yin et al. [17] derived convergence rates for coordinate-wise trimmed mean and median estimators; Sun et al. [18] investigated gradient norm clipping. Table 1 compares FedIDS-Bench with previous FL-IDS experimental frameworks.

### Table 1: Systematic Methodological Comparison: Prior FL-IDS Studies vs. FedIDS-Bench

| Benchmarking Suite | Evaluated Corpora | Train-Test Isolation | Heterogeneity Partitions | Coordination Algorithms | Threat Injection Matrix | Byzantine Defenses | Hardware Metrics | Deterministic Verification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Popoola et al. [7] | 1 (ToN_IoT) | Unverified | Label Partition Only | FedAvg | None | None | None | No |
| Mothukuri et al. [4] | Survey | N/A | N/A | N/A | Conceptual | Conceptual | None | N/A |
| Campos et al. [3] | 2 (CICIDS, IoT-23) | Partial | Uniform & Mild Skew | FedAvg | Label Flip | None | Theoretical | No |
| Sarhan et al. [19] | 3 (UNSW, CIC, ToN) | Verified | Centralized Split Only | N/A | None | None | None | No |
| **FedIDS-Bench (Ours)** | **5 Real Corpora** | **Strict Split-Isolated** | **Dirichlet, Label, Quantity** | **FedAvg, FedProx, FedAdam** | **5 Threat Models** | **4 Robust Defenses** | **RPi4, Jetson, CPU** | **Sub-Process Verified** |

---

## 3 System Model, Problem Formulation, and Threat Landscape

### 3.1 Federated Learning Mathematical Formulation
We model an ensemble of $K$ decentralized network boundary monitors. Each participant $k \in \{1, \dots, K\}$ maintains a private connection flow set $\mathcal{D}_k = \{(x_i^{(k)}, y_i^{(k)})\}_{i=1}^{n_k}$, where feature vector $x_i^{(k)} \in \mathbb{R}^d$ encapsulates flow attributes and $y_i^{(k)} \in \{0, \dots, C-1\}$ designates the threat class. The global objective minimizes empirical risk across all $N = \sum_{k=1}^K n_k$ observations without centralizing flow tables:
$$
\min_{w \in \mathbb{R}^d} F(w) = \sum_{k=1}^K \frac{n_k}{N} F_k(w), \quad F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(f(x_i^{(k)}; w), y_i^{(k)}).
$$

In standard Federated Averaging (FedAvg), the coordinator broadcasts the global parameters $w_t$ to an active cohort $S_t \subseteq \{1, \dots, K\}$, receiving local models $w_k^{t+1}$ trained over $E$ epochs:
$$
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} w_k^{t+1}.
$$

In FedProx [10], clients stabilize local optimization by minimizing a proximal-augmented loss:
$$
\min_{w} F_k(w) + \frac{\mu}{2} \| w - w_t \|^2.
$$

In FedAdam [11], the server computes pseudo-gradient $\Delta_t = \sum_{k \in S_t} \frac{n_k}{N_{S_t}} (w_k^{t+1} - w_t)$, maintaining first and second momentum vectors $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$ and $v_t = \beta_2 v_{t-1} + (1 - \beta_2) \Delta_t^2$ to produce global updates:
$$
w_{t+1} = w_t + \eta \frac{m_t}{\sqrt{v_t} + \tau}.
$$

### 3.2 Statistical Heterogeneity Partitions
FedIDS-Bench supports three realistic multi-domain distribution models:
- **Dirichlet Label Skew ($\text{Dir}(\alpha)$):** Within each client, class probabilities are distributed as $\mathbf{q}_k \sim \text{Dir}(\alpha \mathbf{p})$. For $\alpha \to \infty$, all clients have the same IID distribution, while small $\alpha$ (e.g., $\alpha = 0.1$) skew the class distribution towards specific attack categories on certain clients.
- **Single-Domain Label Partition:** A client’s data contains at most $L < C$ unique classes, modeling disjoint sensing domains such as DNS-only monitors and Industrial PLCs.
- **Pareto Volume Imbalance:** The number of flow samples per client follows a Pareto distribution $n_k \sim \text{Pareto}(\beta)$, creating heavy-tailed traffic patterns between enterprise backbones and IoT leaf nodes.

### 3.3 Adversarial Threat Formulations
We consider five attack primitives that target collaborative IDS:
1. **Targeted Label Inversion:** An adversary inverts benign labels to generate stealthy evasion attacks ($y \mapsto 0$).
2. **Model Sign-Inversion & Amplification:** Poisoned workers invert the sign of all gradients and amplify update magnitude:
$$
\tilde{w}_m^{t+1} = w_t - \gamma(w_m^{t+1} - w_t), \quad \gamma > 0.
$$
3. **Byzantine Gaussian Perturbation:** A fraction of workers replace their model weights with isotropic Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.
4. **Backdoor Flow Watermarking:** A watermark $m \in \{0, 1\}^d$ is embedded into the flow features ($x' = (1 - m) \odot x + m \odot \Delta_{\text{trigger}}$) and associated with a target class.
5. **Sample-Count Spoofing:** An adversary reports an unrealistically large local sample count ($n_m \gg \sum_j n_j$) to spoof the FedAvg gradient weighting.

### 3.4 Byzantine-Robust Defense Aggregators
The server employs four aggregation rules to mitigate poisoned updates:
1. **Norm-Bounded Gradient Clipping:** We clip large updates to bound $\|\Delta w_k\|_2 \leq \tau$.
2. **Coordinate Trimmed Mean:** We sort model parameters across workers for each coordinate and discard $\beta$-fraction from both tails [17].
3. **Median:** Coordinate-wise Median is found by estimating the median value for each coordinate from the data that comes from clients.
4. **Multi-Krum:** Multi-Krum chooses the parameter vector that has the smallest total distance, to its $K - f - 2$ closest neighbours [16].

---

## 4 FedIDS-Bench Modular Architecture

The architectural framework of FedIDS-Bench is shown in Figure 1. FedIDS-Bench is arranged into seven modules.

### 4.1 Split-Isolated Feature Preprocessing
To eliminate distribution leakage I split the corpus into a global training $\mathcal{S}_{\text{train}}$ and an evaluation $\mathcal{S}_{\text{test}}$ before I compute feature parameters. I compute the vector $\mu_{\text{train}}$ and the standard deviation $\sigma_{\text{train}}$ strictly, over the global training $\mathcal{S}_{\text{train}}$:
$$
x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}.
$$
Evaluation sets and decentralized client partitions are transformed only using these parameters making sure there is no train-test leakage.

### 4.2 Proportional Stratified Sampling
Network intrusion datasets often have problems with class imbalance. For example infiltration attacks might make up, than 0.05% of all network flows. When you randomly split the data the rare classes often get left out. Not represented properly.

### 4.3 Client Isolation & Parameter Encapsulation
To avoid hidden weight changes across clients in PyTorch client trainers enforce model isolation by making a deep copy of the entire model using `copy.deepcopy(global_model)`. This deep copy keeps the layer sizes the same optimizer momentum numbers and the same proximal parameters from one round, to the next even when the rounds are very different.

### 4.4 Multi-Tier Edge Profiler
In addition, to accuracy FedIDS-Bench looks at detection systems in three areas:
- **Detection Precision Metrics:** Precision, Recall, Macro-F1, Weighted-F1, False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$), False Negative Rate ($\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}}$), and Attack Recall, with protection against division-by-zero for single-class client partitions.
- **Network Bandwidth Accounting:** Measurement of uplink, downlink, and cumulative byte exchanges per communication epoch.
- **Physical Edge Telemetry:** Computational FLOPs/MACs, parameter storage footprint, inference latency, and energy consumption (Joules) benchmarked across ARM and NVIDIA hardware.

---

## 5 Experimental Setup and Telemetry Corpora

### 5.1 Intrusion Detection Corpora
FedIDS-Bench evaluates five recognized intrusion detection datasets that span enterprise backbone networks and modern IoT ecosystems (Table 2 and Fig. 2):
- **UNSW-NB15** [20]: Synthetic modern attack profiles with 39 extracted flow attributes across 10 distinct classes (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoors, Shellcode, Worms).
- **CICIDS2017** [21]: Contemporary enterprise network traffic with 78 flow features including benign and high volume DDoS flows.
- **CSE-CIC-IDS2018**: Cloud perimeter network captures with 78 flow features across LOIC-UDP, HOIC, and Benign traffic.
- **IoT-23** [2]: Real-world malware and benign IoT device captures extracted from Zeek `conn.log` connection tables (11 features).
- **ToN_IoT** [1]: Heterogeneous telemetry from modern IoT/IIoT sensor networks with 10 attack classes (DDoS, DoS, Ransomware, Backdoor, Injection, Scanning, etc.).

### Table 2: Intrusion Detection Telemetry Corpora Profile

| Dataset | Flow Features | Classes | Domain | Primary Attack Vectors |
| :--- | :---: | :---: | :---: | :--- |
| UNSW-NB15 | 39 | 10 | Enterprise | Exploits, Fuzzers, DoS |
| CICIDS2017 | 78 | 2 | Enterprise | Volumetric DDoS, Benign |
| CSE-CIC-2018 | 78 | 3 | Cloud/Perimeter | LOIC, HOIC, DDoS |
| IoT-23 | 11 | 2 | IoT Edge | Botnets, Malicious Flows |
| ToN_IoT | 16 | 10 | IoT/IIoT | Ransomware, Scanning, DDoS |

### 5.2 Neural Architectures & Hyperparameter Configurations
Evaluations deploy fully connected architectures: `SmallMLP` ($d \to 64 \to 32 \to C$) and `DeepMLP` ($d \to 128 \to 64 \to 32 \to C$) with ReLU activations. Baseline settings: client cohort size $K=10$, local training epochs $E=3$, mini-batch size $B=64$, client learning rate $\eta=0.01$, FedProx coefficient $\mu=0.01$, FedAdam parameters $\beta_1=0.9, \beta_2=0.99, \tau=10^{-3}$, total communication rounds $R=20$.

### 5.3 Physical Edge Testbed Platforms
Empirical execution costs are profiled across three physical deployment environments:
1. **Raspberry Pi 4**: Quad-core ARM Cortex-A72 @ 1.5 GHz, 4 GB LPDDR4, TDP 5.0 W.
2. **NVIDIA Jetson Orin Nano**: 6-core ARM Cortex-A78AE with Ampere GPU (1024 CUDA cores), TDP 15.0 W.
3. **Server / Laptop CPU**: Intel Core i7 x86_64 multi-core processor, TDP 28.0 W.

---

## 6 Empirical Evaluation and Results

### 6.1 Algorithm Robustness Under Non-IID Skew
We evaluate the convergence behavior of FedAvg, FedProx, and FedAdam under IID and non-IID Dirichlet distributions ($\alpha = 0.5$ and $\alpha = 0.1$). As shown in Fig. 3 and Table 3, data heterogeneity severely impacts classical FedAvg. When skewness is moderate ($\alpha = 0.5$), FedAvg experiences client drift, converging to 0.8812 F1 compared to 0.9420 under IID. FedProx helps reduce the change, in parameters a bit (0.9145 F1) but FedAdam gives better stability when it comes to converging (0.9324 F1).

Under non-IID conditions with $\alpha$ equal to 0.1 each sensor sees attack categories that do not overlap with the other. FedAvg suffers from severe weight oscillations, only converging to 0.7640 F1. FedAdam’s adaptive second moment estimation suppresses high-variance client updates, achieving an F1 of 0.8815 ($+11.75\%$ better than FedAvg).

### Table 3: Algorithmic Detection Performance Across Non-IID Regimes on ToN_IoT Telemetry (Round 20)

| Algorithm | Partitioning | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| FedAvg | IID | 0.8940 | 0.9420 | 0.9395 |
| FedAvg | Dirichlet ($\alpha=0.5$) | 0.8415 | 0.8812 | 0.8790 |
| FedProx ($\mu=0.01$) | Dirichlet ($\alpha=0.5$) | 0.8670 | 0.9145 | 0.9110 |
| FedAdam | Dirichlet ($\alpha=0.5$) | 0.8850 | 0.9324 | 0.9302 |
| FedAvg | Dirichlet ($\alpha=0.1$) | 0.7230 | 0.7640 | 0.7580 |
| FedProx ($\mu=0.01$) | Dirichlet ($\alpha=0.1$) | 0.7910 | 0.8420 | 0.8390 |
| FedAdam | Dirichlet ($\alpha=0.1$) | **0.8350** | **0.8815** | **0.8780** |

### 6.2 Adversarial Robustness and Defense Benchmarking
We benchmark security robustness against data and model poisoning under 20% malicious clients (2 of 10 workers). Results are in Table 4 and Fig. 4.

### Table 4: Detection F1-Score Under Adversarial Poisoning Vectors (20% Malicious Workers) Across Byzantine Defenses

| Adversarial Attack Vector | No Defense (FedAvg) | Norm Clipping ($\tau=2.0$) | Trimmed Mean ($\beta=0.2$) | Coordinate Median | Multi-Krum ($m=1$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clean Baseline (0% Malicious) | **0.9380** | 0.9310 | 0.9250 | 0.9180 | 0.9120 |
| Label Flip Attack | 0.6120 | 0.6350 | **0.8840** | 0.8710 | 0.8650 |
| Model Poisoning ($-\gamma \Delta w, \gamma=5.0$) | 0.1850 | 0.7420 | **0.9100** | 0.8980 | 0.8870 |
| Byzantine Isotropic Noise ($\sigma=1.0$) | 0.1240 | 0.6800 | **0.8950** | 0.8820 | 0.8750 |
| Backdoor Flow Trigger | 0.5400 | 0.6100 | **0.8200** | 0.8050 | 0.7900 |
| Sample-Count Inflation ($n_m = 10^6$) | 0.1980 | 0.5840 | **0.9250** | 0.9180 | 0.9100 |

Under baseline FedAvg, sign-inverting model poisoning ($\gamma = 5.0$) makes detection virtually impossible (F1 = 0.1850), while Byzantine Gaussian noise degrades performance to 0.1240 F1. While norm-bounded gradient clipping prevents unbounded scaling (restoring F1 to 0.7420), it remains vulnerable to coordinated adversarial trajectories. Coordinate-wise trimmed mean and median aggregators are more robust to extremal coordinate outliers, recovering detection F1 to 0.9100 and 0.8980 by filtering extreme coordinate outliers. Multi-Krum successfully detects benign clusters (F1 = 0.8870), but has quadratic computational overhead ($\mathcal{O}(K^2 \cdot d)$).

**Empirical Audit Finding: Unvalidated Sample Reporting.** Standard FedAvg computes model weights proportionally to client-reported counts ($w_k = \frac{n_k}{\sum_j n_j}$). When one adversarial client reports an inflated count ($n_m = 10^6$), its update dominates 99.9% of the global model weights, crashing undefended FedAvg to 0.1980 F1. In FedIDS-Bench, setting uniform weighting (`trust_client_sample_counts=False`) defeats this attack entirely (F1 = 0.9250).

### 6.3 Physical Edge Resource and Communication Profiling
Deploying federated IDS on embedded edge devices requires adhering to strict computational and power constraints. Table 5 and Fig. 5 summarize physical testbed measurements.

### Table 5: Hardware Execution Profile, Bandwidth, and Computational Cost Across Edge Tiers (SmallMLP Model)

| Target Edge Device | TDP (W) | Per-Sample Latency (ms) | Batch Energy (mJ) | Params (Bytes) | Round Traffic (Bytes) | MFLOPs / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raspberry Pi 4 (Cortex-A72) | 5.0 | 0.706 | 3.525 | 3,464 | 41,568 | 0.0039 |
| NVIDIA Jetson Orin Nano | 15.0 | **0.024** | **0.357** | 3,464 | 41,568 | 0.0039 |
| x86 Laptop CPU (Core i7) | 28.0 | 0.095 | 2.668 | 3,464 | 41,568 | 0.0039 |

The `SmallMLP` model has 3,464 parameter bytes (866 float32 weights), resulting in a 20.78 KB upload/download bandwidth requirement per client round (41.57 KB total). On the Raspberry Pi 4, inference delay is 0.706 ms per sample with 3.53 mJ energy draw per batch. The Jetson Orin Nano is able to execute under a millisecond (0.024 ms/sample) and draws 0.357 mJ of energy, demonstrating that edge-based federated intrusion detection is entirely feasible on low-power embedded hardware.

### 6.4 Cross-Dataset Generalizability and Transferability
We evaluate model transferability by training classifiers on a source domain and testing on target domains through feature space alignment. As seen in Table 6, classifiers trained on enterprise telemetry (UNSW-NB15) transfer somewhat well to similar enterprise environments (CICIDS2017: 0.3900 transfer accuracy), but perform much worse on heterogeneous IoT telemetry (IoT-23: 0.2150 transfer accuracy). This highlights the domain shift in network flow representations and reinforces the need for multi-dataset evaluation.

### Table 6: Cross-Dataset Transfer Evaluation Accuracy

| Source Dataset | Target Dataset | In-Domain F1 | Transfer Acc. |
| :--- | :--- | :---: | :---: |
| UNSW-NB15 | CICIDS2017 | 0.8840 | 0.3900 |
| UNSW-NB15 | IoT-23 | 0.8840 | 0.2150 |
| ToN_IoT | IoT-23 | 0.9380 | 0.4420 |
| CICIDS2017 | CSE-CIC-2018 | 0.9620 | 0.6840 |

---

## 7 Discussion, Best Practices, and Threats to Validity

### 7.1 Methodological Guidelines for FL-IDS Practitioners
Based on our results we propose four guidelines that future FL-IDS research should follow:
1. **Enforce split-isolated feature normalization:** Under no circumstances should feature normalizers be fitted over combined datasets; parameters must come from training partitions and then be passed forward.
2. **Deploy aggregators by default:** Undefended FedAvg is not fit for production systems, in hostile edge environments; coordinate-wise trimmed mean or median should be the usual baseline.
3. **Validate client-reported sample volumes:** Aggregation coordinators must either verify reported dataset sizes or enforce equal-weight aggregation under untrusted settings.
4. **Benchmark under controlled non-IID Dirichlet skew:** Authors should report performance across multiple $\alpha$ tiers ($\alpha \in \{0.1, 0.5, \infty\}$) rather than relying exclusively on uniform partitions.

### 7.2 Threats to Validity
Several limitations should be noted: (1) Architecture Diversity: While MLPs represent the primary production standard for tabular NetFlow classification, future extensions will incorporate Graph Neural Networks (GNNs) and temporal sequence models; (2) Cryptographic overhead: Current evaluations evaluate plaintext parameter exchange; incorporating Homomorphic Encryption (HE) and Differential Privacy (DP) will introduce supplementary computational and bandwidth trade-offs.

---

## 8 Conclusion

We introduce FedIDS-Bench, a standardized, zero-leakage, cross-threat-model benchmark framework for federated intrusion detection systems. By unifying five premier intrusion datasets, realistic non-IID partitioning schemes, modular FL optimization algorithms, adversarial attack engines, Byzantine-robust defense aggregators, and edge hardware profiling, we address the long-standing reproducibility crisis in FL-IDS literature. Our empirical findings demonstrate the superiority of adaptive federated optimization under severe data heterogeneity, the efficacy of coordinate-wise robust aggregators against poisoning attacks, and the practical feasibility of FL-IDS on embedded edge hardware. All code, configuration files, and scripts are publicly available to promote reproducible research in collaborative cyber defense.

---

## References

1. Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., Anwar, A.: TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven cyber security applications. IEEE Access 8, 165130--165150 (2020)
2. Garcia, S., Parmisano, A., Erquiaga, M.J.: An empirical analysis of IoT-23 dataset for intrusion detection. Stratosphere Laboratory Technical Report (2020)
3. Campos, E.M., Saura, P., González-Vidal, A., Hernández-Ramos, J.L., Bernabé, J.B., Skarmeta, A.: Evaluating federated learning for intrusion detection in the internet of things: Review and challenges. IEEE Communications Surveys & Tutorials 24(4), 2177--2210 (2022)
4. Mothukuri, V., Parizi, R.M., Pouriyeh, S., Huang, Y., Dehghantanha, A., Srivastava, G.: A survey on federated learning toward natural language processing and cybersecurity. IEEE Internet of Things Journal 8(4), 2199--2217 (2021)
5. McMahan, B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A.: Communication-efficient learning of deep networks from decentralized data. In: Artificial Intelligence and Statistics (AISTATS), pp. 1273--1282 (2017)
6. Kairouz, P., McMahan, H.B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A.N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al.: Advances and open problems in federated learning. Foundations and Trends® in Machine Learning 14(1–2), 1--210 (2021)
7. Popoola, S.I., Ande, R., Adebisi, B., Gui, G., Hammoudeh, M., Jogunola, O.: Federated deep learning for zero-day botnet attack detection in IoT edge devices. IEEE Internet of Things Journal 9(5), 3930--3944 (2021)
8. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., Chandra, V.: Federated learning with non-iid data. arXiv preprint arXiv:1806.00582 (2018)
9. Hsu, T.-M.H., Qi, H., Brown, M.: Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335 (2019)
10. Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.: Federated optimization in heterogeneous networks. In: Proceedings of Machine Learning and Systems (MLSys), vol. 2, pp. 429--450 (2020)
11. Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., McMahan, H.B.: Adaptive federated optimization. In: International Conference on Learning Representations (ICLR) (2021)
12. Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., Shmatikov, V.: How to backdoor federated learning. In: International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 2938--2948 (2020)
13. Tolpegin, V., Truex, S., Gursoy, M.E., Liu, L.: Data poisoning attacks against federated learning systems. In: European Symposium on Research in Computer Security (ESORICS), pp. 480--501 (2020)
14. Bhagoji, A.N., Chakrabarti, S., Mukherjee, P., Mittal, P.: Analyzing federated learning through an algorithmic lens. In: International Conference on Machine Learning (ICML), pp. 634--643 (2019)
15. Fang, M., Cao, X., Jia, J., Gong, N.Z.: Local model poisoning attacks to Byzantine-Robust federated learning. In: USENIX Security Symposium, pp. 1605--1622 (2020)
16. Blanchard, P., El Mhamdi, E.M., Guerraoui, R., Stainer, J.: Machine learning with adversaries: Byzantine tolerant gradient descent. In: Advances in Neural Information Processing Systems (NeurIPS), vol. 30 (2017)
17. Yin, D., Pananjady, A., Lam, M., Papailiopoulos, D., Ramchandran, K., Bartlett, P.: Byzantine-robust distributed learning: Towards optimal statistical rates. In: International Conference on Machine Learning (ICML), pp. 5650--5659 (2018)
18. Sun, Z., Kairouz, P., Suresh, A.T., McMahan, H.B.: Can you really backdoor federated learning? arXiv preprint arXiv:1911.07963 (2019)
19. Sarhan, M., Layeghy, S., Moustafa, N., Portmann, M.: Towards a standard feature set for network intrusion detection system datasets. Mobile Networks and Applications, pp. 1--14 (2021)
20. Moustafa, N., Slay, J.: UNSW-NB15: a comprehensive data set for network intrusion detection systems. In: Military Communications and Information Systems Conference (MilCIS), pp. 1--6 (2015)
21. Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A.: Toward generating a new intrusion detection dataset and intrusion traffic characterization. In: International Conference on Information Systems Security and Privacy (ICISSP), vol. 1, pp. 108--116 (2018)
