# Undersampling overview tables

Columns: **us** = spatially undersampled band(s) (density factor < 0.5); **ns** = normal / well-sampled band(s) (density ≥ 0.5). First six numeric columns are variance-based; last six are moment-matched entropy.

Default sampling regions: `[((-5, 4), 1.0), ((4, 8), 0.05), ((8, 10), 1.0)]`.

## Linear, heteroscedastic

```text
Model AU_us AU_ns EU_us EU_ns rho_us rho_ns AU_us AU_ns EU_us EU_ns rho_us rho_ns
Deep Ensemble 0.576 0.233 0.549 0.217 0.824 0.234 0.831 0.535 0.131 0.136 0.359 -0.708
MC Dropout 0.120 0.299 0.308 0.181 0.769 -0.354 0.291 0.426 0.390 0.248 0.715 -0.481
BNN 0.631 0.218 0.056 0.122 0.406 -0.263 0.613 0.665 0.565 0.397 -0.935 -0.700
BAMLSS 0.611 0.137 0.114 0.057 0.021 -0.260 0.887 0.468 0.508 0.495 -0.028 -0.023
```

## Linear, homoscedastic

```text
Model AU_us AU_ns EU_us EU_ns rho_us rho_ns AU_us AU_ns EU_us EU_ns rho_us rho_ns
Deep Ensemble 0.530 0.354 0.345 0.050 0.682 -0.156 0.563 0.393 0.532 0.136 0.449 -0.534
MC Dropout 0.618 0.356 0.304 0.170 0.870 0.934 0.658 0.385 0.362 0.201 0.852 0.917
BNN 0.629 0.294 0.056 0.124 0.960 0.068 0.236 0.607 0.860 0.460 -0.982 -0.784
BAMLSS 0.653 0.224 0.106 0.063 -0.317 -0.061 0.669 0.238 0.492 0.485 0.107 -0.037
```

## Sinusoidal, heteroscedastic

```text
Model AU_us AU_ns EU_us EU_ns rho_us rho_ns AU_us AU_ns EU_us EU_ns rho_us rho_ns
Deep Ensemble 0.540 0.336 0.109 0.048 -0.683 0.055 0.789 0.587 0.236 0.174 -0.940 -0.247
MC Dropout 0.292 0.173 0.214 0.184 0.953 0.860 0.598 0.327 0.157 0.193 0.816 -0.014
BNN — — — — — — — — — — — —
BAMLSS 0.586 0.159 0.091 0.044 -0.015 -0.329 0.880 0.485 0.493 0.479 -0.007 -0.023
```

## Sinusoidal, homoscedastic

```text
Model AU_us AU_ns EU_us EU_ns rho_us rho_ns AU_us AU_ns EU_us EU_ns rho_us rho_ns
Deep Ensemble 0.469 0.195 0.164 0.054 -0.164 0.744 0.629 0.257 0.313 0.111 -0.729 0.707
MC Dropout 0.389 0.187 0.238 0.179 0.940 0.865 0.631 0.287 0.206 0.191 0.787 0.393
BNN 0.582 0.191 0.017 0.103 -0.900 -0.023 0.082 0.500 0.780 0.467 -0.945 -0.791
BAMLSS 0.654 0.228 0.092 0.047 -0.720 -0.130 0.668 0.241 0.499 0.502 -0.005 0.047
```
