# Statistical analysis (Section 5.6)

Records: 3500; outer folds per dataset/model: 50-50


## Mean test RMSE (datasets x models)

| dataset     |   RBD-COG |   RBD-MOM |   RBD-WABL-F |   RBD-WABL-G |   RBD-WABL-R |   TS-0 |   TS-1 |
|:------------|----------:|----------:|-------------:|-------------:|-------------:|-------:|-------:|
| abalone     |    2.2264 |    2.2627 |       2.2001 |       2.182  |       2.1612 | 2.3617 | 2.2772 |
| airfoil     |    3.7022 |    3.7232 |       3.5251 |       3.4367 |       3.418  | 4.2246 | 3.9389 |
| auto_mpg    |    3.0134 |    3.0796 |       2.9864 |       2.9754 |       2.9956 | 3.8552 | 3.304  |
| ccpp        |    4.3813 |    4.3739 |       4.2807 |       4.2857 |       4.2546 | 6.5751 | 5.3106 |
| concrete    |    7.3449 |    7.5015 |       6.8294 |       6.9073 |       6.7019 | 9.1555 | 7.8211 |
| energy      |    1.515  |    1.5932 |       1.2067 |       1.2184 |       1.1007 | 3.3679 | 2.0995 |
| qsar_fish   |    0.97   |    0.9763 |       0.9687 |       0.9718 |       0.9732 | 1.01   | 0.9628 |
| real_estate |    8.1005 |    8.0676 |       8.1098 |       8.1024 |       8.1035 | 8.531  | 8.1627 |
| wine_red    |    0.6538 |    0.6554 |       0.655  |       0.6541 |       0.6535 | 0.6648 | 0.6549 |
| yacht       |    3.3382 |    2.9573 |       2.7101 |       2.6691 |       2.6252 | 6.5052 | 4.3685 |


## Family A: TS-0, TS-1, RBD-WABL-G, RBD-WABL-R

N = 10 datasets, k = 4 models

Average ranks: RBD-WABL-R 1.40, RBD-WABL-G 1.80, TS-1 2.80, TS-0 4.00

Friedman chi2_F = 24.240 (p = 2.226e-05); Iman-Davenport F_F = 37.875 (p = 8.151e-10)

Nemenyi q_0.05 = 2.569, CD = 1.483


Post-hoc rank tests:

| A          | B          |   rank_A |   rank_B |       z |      p |   p_holm |   p_finner |
|:-----------|:-----------|---------:|---------:|--------:|-------:|---------:|-----------:|
| RBD-WABL-G | TS-0       |      1.8 |      4   | -3.8105 | 0.0001 |   0.0004 |     0.0004 |
| RBD-WABL-G | TS-1       |      1.8 |      2.8 | -1.7321 | 0.0833 |   0.1665 |     0.1223 |
| RBD-WABL-G | RBD-WABL-R |      1.8 |      1.4 |  0.6928 | 0.4884 |   0.4884 |     0.4884 |


Wilcoxon signed-rank and Bayesian signed-rank tests (d = (RMSE_A - RMSE_B)/RMSE_B):

| A          | B          |   median_rel_diff |   W |      p | method   |    r_rb |   P(A better) |   P(rope) |   P(B better) |   p_holm | bayes_decision   |
|:-----------|:-----------|------------------:|----:|-------:|:---------|--------:|--------------:|----------:|--------------:|---------:|:-----------------|
| RBD-WABL-G | TS-0       |           -0.2074 |   0 | 0.002  | exact    | -1      |        0.9999 |    0.0001 |        0      |   0.0059 | A better         |
| RBD-WABL-G | TS-1       |           -0.1081 |   3 | 0.0098 | exact    | -0.8909 |        0.9911 |    0.0089 |        0      |   0.0195 | A better         |
| RBD-WABL-G | RBD-WABL-R |            0.0064 |   9 | 0.0645 | exact    |  0.6727 |        0      |    0.7562 |        0.2438 |   0.0645 | undecided        |


## Family B: RBD-COG, RBD-MOM, RBD-WABL-F, RBD-WABL-G, RBD-WABL-R

N = 10 datasets, k = 5 models

Average ranks: RBD-WABL-R 1.80, RBD-WABL-G 2.50, RBD-WABL-F 2.70, RBD-COG 3.60, RBD-MOM 4.40

Friedman chi2_F = 16.400 (p = 0.002527); Iman-Davenport F_F = 6.254 (p = 0.0006289)

Nemenyi q_0.05 = 2.728, CD = 1.929


Post-hoc rank tests:

| A          | B          |   rank_A |   rank_B |       z |      p |   p_holm |   p_finner |
|:-----------|:-----------|---------:|---------:|--------:|-------:|---------:|-----------:|
| RBD-COG    | RBD-MOM    |      3.6 |      4.4 | -1.1314 | 0.2579 |   1      |     0.315  |
| RBD-COG    | RBD-WABL-F |      3.6 |      2.7 |  1.2728 | 0.2031 |   1      |     0.315  |
| RBD-COG    | RBD-WABL-G |      3.6 |      2.5 |  1.5556 | 0.1198 |   0.7188 |     0.2252 |
| RBD-COG    | RBD-WABL-R |      3.6 |      1.8 |  2.5456 | 0.0109 |   0.0873 |     0.0359 |
| RBD-MOM    | RBD-WABL-F |      4.4 |      2.7 |  2.4042 | 0.0162 |   0.1135 |     0.04   |
| RBD-MOM    | RBD-WABL-G |      4.4 |      2.5 |  2.687  | 0.0072 |   0.0649 |     0.0355 |
| RBD-MOM    | RBD-WABL-R |      4.4 |      1.8 |  3.677  | 0.0002 |   0.0024 |     0.0024 |
| RBD-WABL-F | RBD-WABL-G |      2.7 |      2.5 |  0.2828 | 0.7773 |   1      |     0.7773 |
| RBD-WABL-F | RBD-WABL-R |      2.7 |      1.8 |  1.2728 | 0.2031 |   1      |     0.315  |
| RBD-WABL-G | RBD-WABL-R |      2.5 |      1.8 |  0.9899 | 0.3222 |   1      |     0.3509 |


Wilcoxon signed-rank and Bayesian signed-rank tests (d = (RMSE_A - RMSE_B)/RMSE_B):

| A          | B          |   median_rel_diff |   W |      p | method   |    r_rb |   P(A better) |   P(rope) |   P(B better) |   p_holm | bayes_decision   |
|:-----------|:-----------|------------------:|----:|-------:|:---------|--------:|--------------:|----------:|--------------:|---------:|:-----------------|
| RBD-COG    | RBD-MOM    |           -0.0061 |  14 | 0.1934 | exact    | -0.4909 |        0.4008 |    0.5125 |        0.0867 |   0.3867 | undecided        |
| RBD-COG    | RBD-WABL-F |            0.0177 |   4 | 0.0137 | exact    |  0.8545 |        0      |    0.0929 |        0.9071 |   0.0957 | undecided        |
| RBD-COG    | RBD-WABL-G |            0.0213 |   6 | 0.0273 | exact    |  0.7818 |        0      |    0.0469 |        0.9531 |   0.1367 | B better         |
| RBD-COG    | RBD-WABL-R |            0.03   |   4 | 0.0137 | exact    |  0.8545 |        0      |    0.0405 |        0.9595 |   0.0957 | B better         |
| RBD-MOM    | RBD-WABL-F |            0.0298 |   2 | 0.0059 | exact    |  0.9273 |        0      |    0.0118 |        0.9882 |   0.0586 | B better         |
| RBD-MOM    | RBD-WABL-G |            0.036  |   2 | 0.0059 | exact    |  0.9273 |        0      |    0.0118 |        0.9882 |   0.0586 | B better         |
| RBD-MOM    | RBD-WABL-R |            0.0375 |   3 | 0.0098 | exact    |  0.8909 |        0      |    0.0089 |        0.9911 |   0.0781 | B better         |
| RBD-WABL-F | RBD-WABL-G |            0.0011 |  21 | 0.5566 | exact    |  0.2364 |        0      |    0.9807 |        0.0193 |   0.5566 | equivalent       |
| RBD-WABL-F | RBD-WABL-R |            0.0121 |   7 | 0.0371 | exact    |  0.7455 |        0      |    0.3592 |        0.6408 |   0.1484 | undecided        |
| RBD-WABL-G | RBD-WABL-R |            0.0064 |   9 | 0.0645 | exact    |  0.6727 |        0      |    0.7562 |        0.2438 |   0.1934 | undecided        |


## Per-dataset corrected resampled t-tests: win/tie/loss of A vs B (Holm over datasets)

|                           |   tie |   win |
|:--------------------------|------:|------:|
| ('RBD-WABL-G', 'RBD-COG') |     9 |     1 |
| ('RBD-WABL-G', 'RBD-MOM') |     9 |     1 |
| ('RBD-WABL-G', 'TS-0')    |     3 |     7 |
| ('RBD-WABL-G', 'TS-1')    |     7 |     3 |
