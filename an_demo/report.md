# Statistical analysis (Section 5.6)

Records: 700; outer folds per dataset/model: 10-10


## Mean test RMSE (datasets x models)

| dataset     |   RBD-COG |   RBD-MOM |   RBD-WABL-F |   RBD-WABL-G |   RBD-WABL-R |     TS-0 |     TS-1 |
|:------------|----------:|----------:|-------------:|-------------:|-------------:|---------:|---------:|
| friedman1_0 |    2.2662 |    2.352  |       2.0725 |       2.0158 |       2.0237 |   2.5785 |   2.1513 |
| friedman1_1 |    2.0953 |    2.2163 |       1.9596 |       1.978  |       1.9727 |   2.3838 |   2.1136 |
| friedman1_2 |    2.2186 |    2.2969 |       2.0089 |       2.0388 |       2.0395 |   2.6126 |   2.2189 |
| friedman1_3 |    1.992  |    2.0206 |       1.8278 |       1.8406 |       1.8249 |   2.3466 |   1.9991 |
| friedman2_0 |   92.0233 |  100.297  |      60.6621 |      59.0018 |      57.0212 | 164.693  | 109.778  |
| friedman2_1 |   91.3059 |   94.6193 |      63.628  |      62.2816 |      61.4721 | 144.399  | 105.761  |
| friedman2_2 |  107.363  |  119.591  |      82.5858 |      82.4012 |      81.6382 | 166.034  | 121.281  |
| friedman3_0 |    0.1982 |    0.2072 |       0.194  |       0.1943 |       0.1917 |   0.2325 |   0.2124 |
| friedman3_1 |    0.28   |    0.2867 |       0.277  |       0.2768 |       0.2764 |   0.2985 |   0.2885 |
| friedman3_2 |    0.3313 |    0.334  |       0.3344 |       0.3336 |       0.3335 |   0.336  |   0.3412 |


## Family A: TS-0, TS-1, RBD-WABL-G, RBD-WABL-R

N = 10 datasets, k = 4 models

Average ranks: RBD-WABL-R 1.20, RBD-WABL-G 1.80, TS-1 3.10, TS-0 3.90

Friedman chi2_F = 27.000 (p = 5.887e-06); Iman-Davenport F_F = 81.000 (p = 1.283e-13)

Nemenyi q_0.05 = 2.569, CD = 1.483


Post-hoc rank tests:

| A          | B          |   rank_A |   rank_B |       z |      p |   p_holm |   p_finner |
|:-----------|:-----------|---------:|---------:|--------:|-------:|---------:|-----------:|
| RBD-WABL-G | TS-0       |      1.8 |      3.9 | -3.6373 | 0.0003 |   0.0008 |     0.0008 |
| RBD-WABL-G | TS-1       |      1.8 |      3.1 | -2.2517 | 0.0243 |   0.0487 |     0.0363 |
| RBD-WABL-G | RBD-WABL-R |      1.8 |      1.2 |  1.0392 | 0.2987 |   0.2987 |     0.2987 |


Wilcoxon signed-rank and Bayesian signed-rank tests (d = (RMSE_A - RMSE_B)/RMSE_B):

| A          | B          |   median_rel_diff |   W |      p | method   |    r_rb |   P(A better) |   P(rope) |   P(B better) |   p_holm | bayes_decision   |
|:-----------|:-----------|------------------:|----:|-------:|:---------|--------:|--------------:|----------:|--------------:|---------:|:-----------------|
| RBD-WABL-G | TS-0       |           -0.2169 |   0 | 0.002  | exact    | -1      |        0.9999 |    0.0001 |        0      |   0.0059 | A better         |
| RBD-WABL-G | TS-1       |           -0.0802 |   0 | 0.002  | exact    | -1      |        1      |    0      |        0      |   0.0059 | A better         |
| RBD-WABL-G | RBD-WABL-R |            0.0057 |   7 | 0.0371 | exact    |  0.7455 |        0      |    0.9133 |        0.0867 |   0.0371 | undecided        |


## Family B: RBD-COG, RBD-MOM, RBD-WABL-F, RBD-WABL-G, RBD-WABL-R

N = 10 datasets, k = 5 models

Average ranks: RBD-WABL-R 1.50, RBD-WABL-G 2.30, RBD-WABL-F 2.60, RBD-COG 3.70, RBD-MOM 4.90

Friedman chi2_F = 28.000 (p = 1.247e-05); Iman-Davenport F_F = 21.000 (p = 5.269e-09)

Nemenyi q_0.05 = 2.728, CD = 1.929


Post-hoc rank tests:

| A          | B          |   rank_A |   rank_B |       z |      p |   p_holm |   p_finner |
|:-----------|:-----------|---------:|---------:|--------:|-------:|---------:|-----------:|
| RBD-COG    | RBD-MOM    |      3.7 |      4.9 | -1.6971 | 0.0897 |   0.4484 |     0.145  |
| RBD-COG    | RBD-WABL-F |      3.7 |      2.6 |  1.5556 | 0.1198 |   0.4792 |     0.1666 |
| RBD-COG    | RBD-WABL-G |      3.7 |      2.3 |  1.9799 | 0.0477 |   0.2863 |     0.0932 |
| RBD-COG    | RBD-WABL-R |      3.7 |      1.5 |  3.1113 | 0.0019 |   0.013  |     0.0047 |
| RBD-MOM    | RBD-WABL-F |      4.9 |      2.6 |  3.2527 | 0.0011 |   0.0091 |     0.0038 |
| RBD-MOM    | RBD-WABL-G |      4.9 |      2.3 |  3.677  | 0.0002 |   0.0021 |     0.0012 |
| RBD-MOM    | RBD-WABL-R |      4.9 |      1.5 |  4.8083 | 0      |   0      |     0      |
| RBD-WABL-F | RBD-WABL-G |      2.6 |      2.3 |  0.4243 | 0.6714 |   0.6714 |     0.6714 |
| RBD-WABL-F | RBD-WABL-R |      2.6 |      1.5 |  1.5556 | 0.1198 |   0.4792 |     0.1666 |
| RBD-WABL-G | RBD-WABL-R |      2.3 |      1.5 |  1.1314 | 0.2579 |   0.5158 |     0.2821 |


Wilcoxon signed-rank and Bayesian signed-rank tests (d = (RMSE_A - RMSE_B)/RMSE_B):

| A          | B          |   median_rel_diff |   W |      p | method   |    r_rb |   P(A better) |   P(rope) |   P(B better) |   p_holm | bayes_decision   |
|:-----------|:-----------|------------------:|----:|-------:|:---------|--------:|--------------:|----------:|--------------:|---------:|:-----------------|
| RBD-COG    | RBD-MOM    |           -0.0357 |   0 | 0.002  | exact    | -1      |        0.9998 |    0.0002 |        0      |   0.0195 | A better         |
| RBD-COG    | RBD-WABL-F |            0.0917 |   1 | 0.0039 | exact    |  0.9636 |        0      |    0.0011 |        0.9989 |   0.0273 | B better         |
| RBD-COG    | RBD-WABL-G |            0.0852 |   1 | 0.0039 | exact    |  0.9636 |        0      |    0.0011 |        0.9989 |   0.0273 | B better         |
| RBD-COG    | RBD-WABL-R |            0.0897 |   1 | 0.0039 | exact    |  0.9636 |        0      |    0.0006 |        0.9994 |   0.0273 | B better         |
| RBD-MOM    | RBD-WABL-F |            0.1329 |   1 | 0.0039 | exact    |  0.9636 |        0      |    0.0001 |        0.9999 |   0.0273 | B better         |
| RBD-MOM    | RBD-WABL-G |            0.1236 |   0 | 0.002  | exact    |  1      |        0      |    0.0001 |        0.9999 |   0.0195 | B better         |
| RBD-MOM    | RBD-WABL-R |            0.1249 |   0 | 0.002  | exact    |  1      |        0      |    0.0001 |        0.9999 |   0.0195 | B better         |
| RBD-WABL-F | RBD-WABL-G |            0.0016 |  20 | 0.4922 | exact    |  0.2727 |        0.0003 |    0.8011 |        0.1986 |   0.4922 | undecided        |
| RBD-WABL-F | RBD-WABL-R |            0.007  |  11 | 0.1055 | exact    |  0.6    |        0.0002 |    0.535  |        0.4648 |   0.2109 | undecided        |
| RBD-WABL-G | RBD-WABL-R |            0.0057 |   7 | 0.0371 | exact    |  0.7455 |        0      |    0.9133 |        0.0867 |   0.1113 | undecided        |


## Per-dataset corrected resampled t-tests: win/tie/loss of A vs B (Holm over datasets)

|                           |   tie |   win |
|:--------------------------|------:|------:|
| ('RBD-WABL-G', 'RBD-COG') |     7 |     3 |
| ('RBD-WABL-G', 'RBD-MOM') |     7 |     3 |
| ('RBD-WABL-G', 'TS-0')    |     5 |     5 |
| ('RBD-WABL-G', 'TS-1')    |     7 |     3 |
