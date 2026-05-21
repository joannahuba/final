
regulatory-dna-optimization/
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── optimization_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│
├── models/
│   |──our models
│   ├── ensemble.py
│   ├── load_model.py
ocena po minimum 
ocena po minimum chyba że jakiś extremalnie duży
│
├── training/
│   all training code 
│
├── interpretation/
│   ├── saliency_maps.py
│   ├── in_silico_mutagenesis.py
│   ├── integrated_gradients.py
│   ├── motif_analysis.py
runner
│
├── optimization/
│   ├── seq_optimizer.py
│   ├── mutation_generator.py
│   ├── beam_search_optimizer.py
│   ├── evolutionary_optimizer.py
│   ├── constraint_handler.py
│
├── analysis/
│   ├── optimization_curves.py
│   ├── model_disagreement_analysis.py
│   ├── mutation_effect_heatmaps.py
│
├── scripts/
│   ├── run_training.sh
│   ├── run_optimization.sh
│   ├── generate_submission.py
│   ├── evaluation_script.py
│
├── outputs/
│   ├── plots/
│   ├── optimized_sequences.tsv
│   ├── logs/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_model_experiments.ipynb
│   ├── 03_interpretation.ipynb
│   ├── 04_optimization.ipynb
│
├── README.md
├── requirements.txt
└── report.pdf


# 1) Ensemble Models 
- grid search with our current models (one runner so if we modify models or want to test another ones it will be easy to evaluate)
- improvement of models:
Asia's part: improvements of CNN + Transformer (would it work better than current sol)
original deepstarr (apparently good performance)

# 2) interpretation

## `saliency_maps.py`

identifies which positions in a DNA sequence are most important for the model prediction.

The model predicts a regulatory activity score:

$$
f(X)
$$

where:

* $X$ = DNA sequence
* $f(X)$ = predicted activity

Saliency maps calculate how sensitive the prediction is to each nucleotide position.

For each position $i$:

$$
S_i =
\left|
\frac{\partial f(X)}{\partial x_i}
\right|
$$

where:

* $S_i$ = importance score
* large value = position strongly affects prediction


Gibbs sampling
więcej mutacji 
z 1 zadania sekwencje z najwyższą expresją

---

If changing one position strongly changes the prediction:

- that position is important.

---

## `in_silico_mutagenesis.py`

directly tests how mutations affect model predictions.

For every sequence position:

1. replace nucleotide
2. predict new activity
3. measure prediction change

For mutation:

$$
X^{(i \rightarrow b)}
$$

the effect is:

$$
\Delta_i^{(b)}=
f(X^{(i \rightarrow b)})
-
f(X)
$$

where:

* positive $\Delta$ = beneficial mutation
* negative $\Delta$ = harmful mutation

---

## `integrated_gradients.py`

computes smoother and more stable importance maps than standard saliency.

Instead of calculating gradients only once, integrated gradients accumulate gradients gradually from a baseline sequence to the real sequence.

Formula:

$$
IG_i(X)=

(x_i-x_i')
\int_0^1
\frac{\partial f(X'+\alpha(X-X'))}{\partial x_i}
d\alpha
$$

where:

* $X$ = original sequence
* $X'$ = baseline sequence


so move from neutral baselin eg random sequence into real one and check how model react.

---

## `motif_analysis.py`

analyzes recurring DNA patterns (motifs).

A motif is a short sequence pattern:

$$
M = (m_1,m_2,...,m_k)
$$

Examples:

* TATAAA
* CACGTG
* CCAAT

The file counts how often motifs appear in:

* active sequences
* inactive sequences
* optimized sequences

podmiany k-mer'ów optymalizacja po gradiencie rozkład probabilistyczny losujący nam że np weź 2 kmery takiej długości 3 takiej itp

baza motywów wyszukujemy wszystkie, które sie pojawiają w naszej sekwencji
lista z motywami-losujemy ile z tej listy wybieramy, potem ile zmieniamy, jakie długości, aby optymalizował po podmianach per model
problem z gradientem-klasa nazwa modelu: jaka architektury i loadery, podłączamy do tej klasy 


---


## Main analysis

The file extracts k-mers:

$$
k = 4,5,6,7...
$$

and measures enrichment:

$$
\text{Enrichment}=

\log_2
\left(
\frac{P(\text{motif}|\text{active})}
{P(\text{motif}|\text{inactive})}
\right)
$$

Positive enrichment:

* motif associated with active enhancers

Negative enrichment:

* motif associated with inactive sequences

---

# 3) Optimization Section (DNA Sequence Design System)

## What this section is about

responsible for improving DNA sequences using the trained machine learning model.

Instead of only predicting regulatory activity, the model is used as a scoring function:

$$
f(X) = \text{predicted RNA/DNA activity}
$$

The goal of this module is to find a new sequence $X^*$ that maximizes this score:

$$
X^* = \arg\max_X f(X)
$$

This is done by iteratively modifying DNA sequences and evaluating them using the model.

---

## `mutation_generator.py`

creates new DNA sequence variants by introducing small changes (mutations).

A DNA sequence is represented as:

$$
X = (x_1, x_2, ..., x_n), \quad x_i \in {A,C,G,T}
$$

A mutation replaces a nucleotide:

$$
x_i \rightarrow x_i'
$$

where:

$$
x_i' \in {A,C,G,T} \setminus x_i
$$

Two strategies:

### 1. Random mutation

* chooses positions randomly
* replaces nucleotides randomly
* encourages exploration

### 2. Guided mutation

* uses importance scores from interpretation methods (saliency / IG)
* focuses mutations on important positions

$$
P(\text{mutate position } i) \propto S_i
$$

where $S_i$ is importance of position $i$.

### Probably: hybrid strategy


$$
\text{mutation} =
\lambda \cdot \text{guided mutation}
+
(1-\lambda) \cdot \text{random mutation}
$$

where:

* $\lambda \in [0,1]$
* controls balance between guidance and exploration


For each λ value:

## 1. Optimization curve

$$
t \rightarrow f(X_t)
$$

* smooth increase
* stable convergence

---

## 2. Diversity of sequences

$$
\text{Hamming distance}(X_i, X_j)
$$

$$
d_H(X, Y) = \sum_{i=1}^{n} \mathbf{1}(x_i \neq y_i)
$$

If diversity is too low:

- guided mutation dominates too much

---

## 3. Biological plausibility

Check:

* GC content stability
* motif preservation
* repeat avoidance


## `constraint_handler.py`

ensures that generated sequences remain biologically realistic and do not exploit weaknesses in the model.

It filters sequences based on biological rules.

---

## Key constraint: GC content

GC content is defined as:

$$
GC(X) = \frac{N_G + N_C}{|X|}
$$

where:

* $N_G$ = number of G bases
* $N_C$ = number of C bases

The system enforces:

$$
0.2 \leq GC(X) \leq 0.8
$$

This prevents unrealistic sequences with extreme nucleotide composition.

---

## Repeat filtering

removes low-complexity sequences such as:

* AAAAAA
* CCCCC
* repetitive patterns

These sequences can artificially increase model predictions without biological meaning.

---

## `beam_search_optimizer.py`

**local, controlled optimization** of DNA sequences.

It starts from an initial sequence and iteratively improves it by exploring nearby mutations.

---

## Algorithm idea

At each step:

1. generate multiple mutated versions of each sequence
2. evaluate them using the model
3. keep only the best $k$ sequences

Formally:

$$
\text{Beam}_t = \text{Top-}k \left( f(X_t^{(1)}), ..., f(X_t^{(n)}) \right)
$$

---

## `evolutionary_optimizer.py`

performs global optimization inspired by biological evolution.

It maintains a population of sequences:

$$
P = {X_1, X_2, ..., X_m}
$$

Each sequence is evaluated using the model:

$$
f(X_i)
$$

---

## Evolution process

Each generation:

### 1. Selection

Keep best-performing sequences:

$$
\text{elite} = \text{Top-}k(P)
$$


### 2. Mutation

Generate new sequences:

$$
X' = \text{Mutate}(X)
$$


### 3. Replacement

Form new population and repeat.


* high diversity
* strong exploration power
* ability to discover new high-scoring patterns so it could be better for subtask B

---

## `seq_optimizer.py`

runner for optimization pipeline

It combines all components into one pipeline:

* mutation generation
* beam search
* evolutionary optimization
* constraint filtering
* model evaluation
