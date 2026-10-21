# E3b — tables (same metric definitions as outputs/hybrid/tables.md)


## A — all clips (median; rate_* = failure rate)

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 51 | 0.592 | 0.976 | 1.23 | 0.0611 | 0.0458 | 0.972 | 0.809 | 0.618 | 4.82 | 11.1 | 0.098 | 0.107 | 0.286 | 0.0714 | 0.196 |
| pi3x_kgate | 51 | 0.652 | 1.14 | 1.19 | 0.0946 | 0.0456 | 0.973 | 0.813 | 0.67 | 4.06 | 7.03 | 0.0588 | 0.214 | 0.679 | 0.25 | 0.216 |
| pi3x_kgate@vipe_d | 51 | 0.652 | 1 | 1.19 | 0.0624 | 0.0456 | 0.973 | 0.813 | 0.67 | 4.06 | 7.03 | 0.0588 | 0 | 0.321 | 0.0714 | 0.216 |
| pi3x_kgate@vipe_t | 51 | 0.652 | 0.978 | 1.19 | 0.0782 | 0.0456 | 0.973 | 0.813 | 0.67 | 4.06 | 7.03 | 0.0588 | 0.0714 | 0.286 | 0.107 | 0.216 |
| pi3x_kgate@vipe_td | 51 | 0.652 | 0.99 | 1.19 | 0.0688 | 0.0456 | 0.973 | 0.813 | 0.67 | 4.06 | 7.03 | 0.0588 | 0.0714 | 0.25 | 0.107 | 0.216 |
| vipe | 51 | 0.592 | 0.976 | 1.23 | 0.114 | 0.138 | 0.819 | 0.493 | 0.544 | 5.62 | 11.2 | 0.098 | 0.107 | 0.286 | 0.143 | 0.196 |

## A — mean rank over 7scenes/tum/bonn/sintel/kitti among the methods of this experiment

| | geom | metric | cons | mean |
|---|---|---|---|---|
| pi3x_kgate@vipe_d | 3.05 | 2.5 | 2.83 | 2.79 |
| pi3x_kgate@vipe_t | 3.05 | 3.12 | 2.83 | 3 |
| pi3x_kgate@vipe_td | 3.05 | 3.5 | 2.83 | 3.13 |
| pi3x_kgate | 3.05 | 4.38 | 2.83 | 3.42 |
| H:vipe+pi3x+med | 4.01 | 2.88 | 4.62 | 3.84 |
| vipe | 4.79 | 4.62 | 5.04 | 4.82 |

### A / 7scenes

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 7 | 0.696 | 0.918 | 1.23 | 0.14 | 0.0406 | 0.974 | 0.89 | 0.618 | 5.59 | 13.3 | 0 | 0 | 0.429 | 0.143 | 0 |
| pi3x_kgate | 7 | 0.582 | 1.01 | 1.23 | 0.0396 | 0.034 | 0.974 | 0.923 | 0.736 | 5.22 | 10.6 | 0 | 0 | 0.286 | 0 | 0 |
| pi3x_kgate@vipe_d | 7 | 0.582 | 0.897 | 1.23 | 0.115 | 0.034 | 0.974 | 0.923 | 0.736 | 5.22 | 10.6 | 0 | 0 | 0.571 | 0.143 | 0 |
| pi3x_kgate@vipe_t | 7 | 0.582 | 0.918 | 1.23 | 0.119 | 0.034 | 0.974 | 0.923 | 0.736 | 5.22 | 10.6 | 0 | 0 | 0.429 | 0.143 | 0 |
| pi3x_kgate@vipe_td | 7 | 0.582 | 0.914 | 1.23 | 0.117 | 0.034 | 0.974 | 0.923 | 0.736 | 5.22 | 10.6 | 0 | 0 | 0.429 | 0.143 | 0 |
| vipe | 7 | 0.696 | 0.918 | 1.23 | 0.178 | 0.105 | 0.89 | 0.574 | 0.527 | 7.66 | 19.2 | 0 | 0 | 0.429 | 0.286 | 0 |

### A / bonn

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 8 | 0.956 | 1 | 0.213 | 0.0586 | 0.0267 | 0.991 | 0.878 | 0.263 | 4.47 | 6.25 | 0.125 | 0.25 | 0.25 | 0.125 | 0.125 |
| pi3x_kgate | 8 | 0.957 | 1.14 | 0.213 | 0.0721 | 0.0269 | 0.991 | 0.838 | 0.254 | 3.66 | 6.74 | 0 | 0.125 | 0.625 | 0 | 0.125 |
| pi3x_kgate@vipe_d | 8 | 0.957 | 1.03 | 0.213 | 0.0552 | 0.0269 | 0.991 | 0.838 | 0.254 | 3.66 | 6.74 | 0 | 0 | 0.25 | 0.125 | 0.125 |
| pi3x_kgate@vipe_t | 8 | 0.957 | 1 | 0.213 | 0.074 | 0.0269 | 0.991 | 0.838 | 0.254 | 3.66 | 6.74 | 0 | 0.125 | 0.25 | 0.125 | 0.125 |
| pi3x_kgate@vipe_td | 8 | 0.957 | 1.03 | 0.213 | 0.0646 | 0.0269 | 0.991 | 0.838 | 0.254 | 3.66 | 6.74 | 0 | 0.125 | 0.125 | 0.125 | 0.125 |
| vipe | 8 | 0.956 | 1 | 0.213 | 0.0873 | 0.0722 | 0.974 | 0.736 | 0.491 | 4.4 | 6.65 | 0.125 | 0.25 | 0.25 | 0.125 | 0.125 |

### A / kitti

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 5 | 0.79 | 1.09 | 1.58 | 0.0645 | 0.0402 | 0.984 |  |  |  |  | 0.2 | 0.2 | 0.6 | 0 | 0 |
| pi3x_kgate | 5 | 0.567 | 1.33 | 1.58 | 0.24 | 0.0356 | 0.988 |  |  |  |  | 0 | 1 | 1 | 0.8 | 0 |
| pi3x_kgate@vipe_d | 5 | 0.567 | 1.11 | 1.58 | 0.0777 | 0.0356 | 0.988 |  |  |  |  | 0 | 0 | 0.6 | 0 | 0 |
| pi3x_kgate@vipe_t | 5 | 0.567 | 1.09 | 1.58 | 0.0979 | 0.0356 | 0.988 |  |  |  |  | 0 | 0.2 | 0.6 | 0.2 | 0 |
| pi3x_kgate@vipe_td | 5 | 0.567 | 1.09 | 1.58 | 0.0979 | 0.0356 | 0.988 |  |  |  |  | 0 | 0.2 | 0.6 | 0.2 | 0 |
| vipe | 5 | 0.79 | 1.09 | 1.58 | 0.134 | 0.112 | 0.866 |  |  |  |  | 0.2 | 0.2 | 0.6 | 0 | 0 |

### A / sintel

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 23 | 0.287 |  | 2.64 |  | 0.12 | 0.834 | 0.105 |  | 3.94 | 12.5 | 0.13 |  |  |  | 0.391 |
| pi3x_kgate | 23 | 1.09 |  | 2.44 |  | 0.148 | 0.833 | 0.146 |  | 3.88 | 6.44 | 0.13 |  |  |  | 0.435 |
| pi3x_kgate@vipe_d | 23 | 1.09 |  | 2.44 |  | 0.148 | 0.833 | 0.146 |  | 3.88 | 6.44 | 0.13 |  |  |  | 0.435 |
| pi3x_kgate@vipe_t | 23 | 1.09 |  | 2.44 |  | 0.148 | 0.833 | 0.146 |  | 3.88 | 6.44 | 0.13 |  |  |  | 0.435 |
| pi3x_kgate@vipe_td | 23 | 1.09 |  | 2.44 |  | 0.148 | 0.833 | 0.146 |  | 3.88 | 6.44 | 0.13 |  |  |  | 0.435 |
| vipe | 23 | 0.287 |  | 2.64 |  | 0.359 | 0.493 | 0.0475 |  | 6.99 | 12.2 | 0.13 |  |  |  | 0.391 |

### A / tum

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 8 | 0.482 | 0.965 | 0.532 | 0.0497 | 0.0423 | 0.978 | 0.876 | 0.739 | 3.36 | 6.63 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate | 8 | 0.309 | 1.14 | 0.532 | 0.107 | 0.032 | 0.979 | 0.915 | 0.739 | 3.08 | 4.56 | 0 | 0 | 0.875 | 0.375 | 0 |
| pi3x_kgate@vipe_d | 8 | 0.309 | 0.99 | 0.532 | 0.0513 | 0.032 | 0.979 | 0.915 | 0.739 | 3.08 | 4.56 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate@vipe_t | 8 | 0.309 | 0.968 | 0.532 | 0.0563 | 0.032 | 0.979 | 0.915 | 0.739 | 3.08 | 4.56 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate@vipe_td | 8 | 0.309 | 0.984 | 0.532 | 0.0561 | 0.032 | 0.979 | 0.915 | 0.739 | 3.08 | 4.56 | 0 | 0 | 0 | 0 | 0 |
| vipe | 8 | 0.482 | 0.965 | 0.532 | 0.108 | 0.103 | 0.888 | 0.537 | 0.588 | 3.93 | 9.19 | 0 | 0 | 0 | 0.125 | 0 |

## B — all clips (median; rate_* = failure rate)

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 13 | 0.345 | 1.07 | 0.177 | 0.158 | 0.0557 | 0.97 | 0.0491 | 0.0407 | 7.05 | 15 | 0 | 0.308 | 0.692 | 0.385 | 0 |
| pi3x_kgate | 13 | 0.251 | 1.01 | 0.177 | 0.0775 | 0.0495 | 0.97 | 0.0644 | 0.0695 | 4.05 | 9.33 | 0.0769 | 0.0769 | 0.154 | 0.154 | 0 |
| pi3x_kgate@vipe_d | 13 | 0.251 | 1.15 | 0.177 | 0.165 | 0.0495 | 0.97 | 0.0644 | 0.0695 | 4.05 | 9.33 | 0.0769 | 0.462 | 0.769 | 0.385 | 0 |
| pi3x_kgate@vipe_t | 13 | 0.251 | 1.07 | 0.177 | 0.182 | 0.0495 | 0.97 | 0.0644 | 0.0695 | 4.05 | 9.33 | 0.0769 | 0.308 | 0.692 | 0.385 | 0 |
| pi3x_kgate@vipe_td | 13 | 0.251 | 1.09 | 0.177 | 0.159 | 0.0495 | 0.97 | 0.0644 | 0.0695 | 4.05 | 9.33 | 0.0769 | 0.385 | 0.615 | 0.385 | 0 |
| vipe | 13 | 0.345 | 1.07 | 0.177 | 0.213 | 0.2 | 0.665 | 0.0254 | 0.0241 | 12.5 | 52.3 | 0 | 0.308 | 0.692 | 0.692 | 0 |

### B / tartanair

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 6 | 0.414 | 0.895 | 0.168 | 0.151 | 0.0819 | 0.943 | 0.0162 | 0.0307 | 21.2 | 37.9 | 0 | 0.167 | 0.667 | 0.333 | 0 |
| pi3x_kgate | 6 | 1.19 | 0.935 | 0.168 | 0.158 | 0.0811 | 0.944 | 0.0448 | 0.0646 | 27.5 | 24.6 | 0.167 | 0.167 | 0.333 | 0.333 | 0 |
| pi3x_kgate@vipe_d | 6 | 1.19 | 1.07 | 0.168 | 0.125 | 0.0811 | 0.944 | 0.0448 | 0.0646 | 27.5 | 24.6 | 0.167 | 0.333 | 0.5 | 0.333 | 0 |
| pi3x_kgate@vipe_t | 6 | 1.19 | 0.896 | 0.168 | 0.206 | 0.0811 | 0.944 | 0.0448 | 0.0646 | 27.5 | 24.6 | 0.167 | 0.167 | 0.667 | 0.5 | 0 |
| pi3x_kgate@vipe_td | 6 | 1.19 | 0.985 | 0.168 | 0.173 | 0.0811 | 0.944 | 0.0448 | 0.0646 | 27.5 | 24.6 | 0.167 | 0.333 | 0.333 | 0.5 | 0 |
| vipe | 6 | 0.414 | 0.895 | 0.168 | 0.365 | 0.247 | 0.576 | 0.0234 | 0.0446 | 37.9 | 53.2 | 0 | 0.167 | 0.667 | 1 | 0 |

### B / vkitti2

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 7 | 0.126 | 1.16 | 0.204 | 0.182 | 0.041 | 0.978 | 0.052 | 0.0407 | 2.54 | 8.8 | 0 | 0.429 | 0.714 | 0.429 | 0 |
| pi3x_kgate | 7 | 0.151 | 1.02 | 0.204 | 0.0646 | 0.0404 | 0.978 | 0.0672 | 0.0695 | 2.52 | 5.16 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate@vipe_d | 7 | 0.151 | 1.24 | 0.204 | 0.184 | 0.0404 | 0.978 | 0.0672 | 0.0695 | 2.52 | 5.16 | 0 | 0.571 | 1 | 0.429 | 0 |
| pi3x_kgate@vipe_t | 7 | 0.151 | 1.16 | 0.204 | 0.148 | 0.0404 | 0.978 | 0.0672 | 0.0695 | 2.52 | 5.16 | 0 | 0.429 | 0.714 | 0.286 | 0 |
| pi3x_kgate@vipe_td | 7 | 0.151 | 1.17 | 0.204 | 0.159 | 0.0404 | 0.978 | 0.0672 | 0.0695 | 2.52 | 5.16 | 0 | 0.429 | 0.857 | 0.286 | 0 |
| vipe | 7 | 0.126 | 1.16 | 0.204 | 0.192 | 0.171 | 0.737 | 0.0254 | 0.0232 | 11.5 | 27.1 | 0 | 0.429 | 0.714 | 0.429 | 0 |

## scannetpp — all clips (median; rate_* = failure rate)

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 28 | 0.223 | 0.995 | 0.0706 | 0.0424 | 0.0235 | 0.986 | 0.965 | 0.767 | 0.811 | 1.91 | 0 | 0.0357 | 0.0357 | 0 | 0 |
| pi3x_kgate | 28 | 0.238 | 0.985 | 0.0706 | 0.0427 | 0.0227 | 0.986 | 0.982 | 0.942 | 1.4 | 1.9 | 0 | 0.0357 | 0.107 | 0.0357 | 0 |
| pi3x_kgate@vipe_d | 28 | 0.238 | 1 | 0.0706 | 0.045 | 0.0227 | 0.986 | 0.982 | 0.942 | 1.4 | 1.9 | 0 | 0 | 0.143 | 0 | 0 |
| pi3x_kgate@vipe_t | 28 | 0.238 | 0.996 | 0.0706 | 0.0406 | 0.0227 | 0.986 | 0.982 | 0.942 | 1.4 | 1.9 | 0 | 0.0357 | 0.0357 | 0.0357 | 0 |
| pi3x_kgate@vipe_td | 28 | 0.238 | 0.995 | 0.0706 | 0.0406 | 0.0227 | 0.986 | 0.982 | 0.942 | 1.4 | 1.9 | 0 | 0 | 0.0714 | 0 | 0 |
| vipe | 28 | 0.223 | 0.995 | 0.0706 | 0.108 | 0.104 | 0.913 | 0.5 | 0.494 | 1.32 | 3.63 | 0 | 0.0357 | 0.0357 | 0.107 | 0 |

### scannetpp unseen-3 (not in MapAnything train split)

| | n_ok | ate_sim3_pct | sim3_scale | fov_err_deg | depth_metric_absrel | depth_seq_absrel | depth_seq_delta1 | pc_fscore_5cm | pc_traj_fscore_5cm | g5_epe_med_px | g20_epe_med_px | rate_fail_ate_sim3 | rate_fail_ate | rate_fail_scale | rate_fail_depth | rate_fail_fov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 3 | 0.21 | 0.986 | 0.0537 | 0.0736 | 0.0269 | 0.982 | 0.973 | 0.888 | 0.54 | 1.47 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate | 3 | 0.233 | 0.998 | 0.0537 | 0.0374 | 0.025 | 0.981 | 0.983 | 0.948 | 1.82 | 2.51 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate@vipe_d | 3 | 0.233 | 1.02 | 0.0537 | 0.0617 | 0.025 | 0.981 | 0.983 | 0.948 | 1.82 | 2.51 | 0 | 0 | 0.333 | 0 | 0 |
| pi3x_kgate@vipe_t | 3 | 0.233 | 0.986 | 0.0537 | 0.0583 | 0.025 | 0.981 | 0.983 | 0.948 | 1.82 | 2.51 | 0 | 0 | 0 | 0 | 0 |
| pi3x_kgate@vipe_td | 3 | 0.233 | 1 | 0.0537 | 0.0684 | 0.025 | 0.981 | 0.983 | 0.948 | 1.82 | 2.51 | 0 | 0 | 0 | 0 | 0 |
| vipe | 3 | 0.21 | 0.986 | 0.0537 | 0.108 | 0.0961 | 0.927 | 0.559 | 0.505 | 1.78 | 5.31 | 0 | 0 | 0 | 0 | 0 |

## wild (C) — medians (no GT)

| | g5_selfcons | g20_selfcons | g5_photo | hfov_deg | depth_agree_absrel | metric_scale_dev | implaus_speed | implaus_depth |
|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 0.858 | 0.654 | 14.6 | 70 | 0.204 | 1.29 | 0 | 0 |
| pi3x_kgate | 0.889 | 0.792 | 11.5 | 70.7 | 0.209 | 1.5 | 0 | 0 |
| pi3x_kgate@vipe_d | 0.889 | 0.792 | 11.5 | 70.7 | 0.209 | 1.08 | 0 | 0 |
| pi3x_kgate@vipe_t | 0.889 | 0.792 | 11.5 | 70.7 | 0.209 | 1.43 | 0 | 0 |
| pi3x_kgate@vipe_td | 0.889 | 0.792 | 11.5 | 70.7 | 0.209 | 1.22 | 0 | 0 |
| vipe | 0.772 | 0.369 | 15.1 | 70 | 0.33 | 1.28 | 0 | 0 |

### wild by category: g5 self-consistency / depth agreement

| | ('g5_selfcons', 'driving') | ('g5_selfcons', 'embodied') | ('g5_selfcons', 'hard') | ('g5_selfcons', 'indoor_dynamic') | ('g5_selfcons', 'indoor_static') | ('g5_selfcons', 'outdoor_dynamic') | ('g5_selfcons', 'outdoor_static') | ('depth_agree_absrel', 'driving') | ('depth_agree_absrel', 'embodied') | ('depth_agree_absrel', 'hard') | ('depth_agree_absrel', 'indoor_dynamic') | ('depth_agree_absrel', 'indoor_static') | ('depth_agree_absrel', 'outdoor_dynamic') | ('depth_agree_absrel', 'outdoor_static') |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H:vipe+pi3x+med | 0.612 | 0.672 | 0.867 | 0.874 | 0.96 | 0.83 | 0.942 | 0.366 | 0.184 | 0.26 | 0.117 | 0.105 | 0.31 | 0.299 |
| pi3x_kgate | 0.726 | 0.889 | 0.909 | 0.871 | 0.965 | 0.839 | 0.946 | 0.364 | 0.184 | 0.31 | 0.127 | 0.108 | 0.325 | 0.289 |
| pi3x_kgate@vipe_d | 0.726 | 0.889 | 0.909 | 0.871 | 0.965 | 0.839 | 0.946 | 0.364 | 0.184 | 0.31 | 0.127 | 0.108 | 0.325 | 0.289 |
| pi3x_kgate@vipe_t | 0.726 | 0.889 | 0.909 | 0.871 | 0.965 | 0.839 | 0.946 | 0.364 | 0.184 | 0.31 | 0.127 | 0.108 | 0.325 | 0.289 |
| pi3x_kgate@vipe_td | 0.726 | 0.889 | 0.909 | 0.871 | 0.965 | 0.839 | 0.946 | 0.364 | 0.184 | 0.31 | 0.127 | 0.108 | 0.325 | 0.289 |
| vipe | 0.455 | 0.743 | 0.946 | 0.796 | 0.763 | 0.705 | 0.822 | 0.522 | 0.296 | 0.412 | 0.197 | 0.24 | 0.524 | 0.394 |