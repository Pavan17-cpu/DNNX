# DNN Programming Assignment — RNN, LSTM, GRU and Transformer
**Name:** YOUR NAME  |  **Student ID:** YOUR STUDENT ID  |  **Section:** YOUR SECTION  
**Chosen values:** h = 32; d = 64; heads = 4; seeds = 42, 123, 2026.

> Complete this report only after running both notebooks. Replace every bracketed item with measured results and insert the generated numbered figures/tables. Maximum six A4 pages for the report body; AI-use statement and references are outside the six-page limit.

## Problem 1 — Recurrent models
### Setup and data pipeline
UCI HAR contains 128 time steps and nine inertial channels. Explain the subject-wise validation split for seed [seed], list the four held-out subjects, and state that channel mean and standard deviation were fitted on the remaining training subjects only. Insert Figure 1 (class balance).

### Cell verification and parameter counts
Report maximum absolute error from `verify_cells()` for RNN, LSTM and GRU; all must be below 1e-5. For d=9 and h=32, the one-bias theoretical counts are: RNN = h(d+h+1) = 1,344; LSTM = 4h(d+h+1) = 5,376; GRU = 3h(d+h+1) = 4,032. The matching PyTorch modules (and scratch implementations retaining separate input/recurrent biases) have counts RNN = 1,376, LSTM = 5,504 and GRU = 4,128. Each is larger by h, 4h and 3h respectively because two bias vectors are stored. Explain that the one-bias formulas count a combined/effective bias, while the implementation keeps both vectors to copy PyTorch weights exactly.

### HAR classification
Insert Table 1 with each model's test accuracy and macro-F1 (mean ± sample std over three seeds), parameter count and seconds/epoch. Insert Figure 2 (best model confusion matrix). State epoch count and batch size.

### Adding problem
Insert Table 2 with test MSE mean ± std for T=50,100,200,400 and the solved-run counts (MSE < 0.05). Insert Figure 3 (T=200 learning curves) and Figure 4 (MSE versus sequence length). Compare with the constant prediction baseline 0.167.

### Gradients, sweep and ablation
Insert Figure 5 (gradient norm versus time step), Figure 6 (LSTM/GRU hidden-size sweep), and Figure 7 (forget-gate bias 0 versus 1 at T=200). The chosen ablation is an explicit assumption because the student-ID variant rule was excluded. Discuss memory, failure as T grows, accuracy/parameter/time trade-off, and wearable recommendation using measured evidence.

## Problem 2 — Transformer
### Attention and encoder verification
Use `three_token_hand_calculation.txt` to present the exact Q, K, V matrices, scaled score matrix, row-wise softmax weights and output. For the chosen matrices, Q = K = [[1,0],[0,1],[1,1]] and V = [[1,0],[0,2],[3,1]]. First compute QK^T = [[1,0,1],[0,1,1],[1,1,2]], then divide every entry by sqrt(2), giving scores [[0.7071,0,0.7071],[0,0.7071,0.7071],[0.7071,0.7071,1.4142]]. Apply softmax separately to each row to obtain approximately [[0.4011,0.1978,0.4011],[0.1978,0.4011,0.4011],[0.2483,0.2483,0.5035]]. Multiply these weights by V; the resulting attention output is approximately [[1.6044,0.7967],[1.4011,1.2033],[1.7587,1.0000]]. For example, output token 1 is [0.4011,0.1978,0.4011] @ V = [1.6044,0.7967]. Explain the hand calculation and state the maximum difference from the code result. Include the unit-test errors. For d=64 and d_ff=256, the weight-only count per block is 4d² + 2d·d_ff = 49,152. Add linear biases and LayerNorm parameters when reconciling with `sum(p.numel())`.

### HAR encoder classifier
Insert Table 3 comparing positional settings (none, sinusoidal, learned) with the best recurrent model: accuracy, macro-F1, parameters and seconds/epoch.

### Character language model
State vocabulary size, 90/10 sequential split, context 128, d=64, four heads and two layers. Insert Figure 8 (validation learning curve), validation loss in nats and bits/character. Include the three generated 500-character samples in `generated_samples.txt` as appendix or supplementary output and discuss temperature.

### Ablations and scaling
Insert Figures 9–10 (two attention heatmaps), Figure 11 (head-count ablation), Figure 12 (causal-mask ablation), and Figure 13 (time/memory versus context length). Explain the O(n²·d) attention cost and limitations of the measured benchmark.

## AI-use statement
[Describe specifically which code, debugging, explanation or editing assistance was obtained from AI. Confirm that you ran, checked and understand the submitted work, and that the reported results are your own experimental outputs.]

## References
1. Anguita et al., “A Public Domain Dataset for Human Activity Recognition Using Smartphones,” ESANN, 2013.
2. PyTorch documentation: https://pytorch.org/docs/stable/
3. Karpathy, Tiny Shakespeare dataset: https://github.com/karpathy/char-rnn
