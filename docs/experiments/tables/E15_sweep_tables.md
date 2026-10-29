### A. Stage-1 replica (v0a) per clip: signals and gate

| clip | reg | pts/img | reproj | f/W | fx/fy | HFOV | planar | tri50 | gate | flags | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl_crowd_4437 | 60/60 | 21 | 0.78 | 1.031 | 4.216 | 51.7 | 0.12 | 7.8 | stop | points_per_image;fx_fy_mismatch |  |
| ctl_drive_42368 | 60/60 | 2 | 0.91 | 17.834 | 14.320 | 3.2 | 0.09 | 1.5 | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;low_parallax |  |
| ctl_drive_45312 | 60/60 | 118 | 0.42 | 1.117 | 1.114 | 48.2 | 0.43 | 3.7 | stop | fx_fy_mismatch |  |
| ctl_drive_52454 | 60/60 | 21 | 0.48 | 2.013 | 6.441 | 27.9 | 0.03 | 4.1 | stop | points_per_image;fx_fy_mismatch |  |
| ctl_misc_14774 | 60/60 | 70 | 0.56 | 0.679 | 1.011 | 72.7 | 0.18 | 4.2 | ok |  |  |
| ctl_misc_3498 | 23/60 | 106 | 0.23 | 0.709 | 0.969 | 70.4 | 0.57 | 2.8 | stop | registered_lt_half;planar_or_panoramic_pairs |  |
| ctl_misc_45623 | 60/60 | 103 | 0.27 | 0.397 | 0.065 | 103.1 | 1.00 | 2.3 | stop | fx_fy_mismatch;planar_or_panoramic_pairs |  |
| ctl_night_40640 | 60/60 | 4 | 0.34 | 1.450 | 1.701 | 38.0 | 1.00 | 1.2 | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier;low_parallax |  |
| ctl_night_40746 | 0/50 |  |  |  |  |  | 1.00 |  | stop | no_model |  |
| ctl_night_4451 | 0/60 |  |  |  |  |  | 1.00 |  | stop | no_model |  |
| ctl_pan_4448 | 60/60 | 0 | 0.00 | 1.119 | 0.363 | 48.2 | 1.00 |  | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier |  |
| ctl_pan_45414 | 60/60 | 64 | 0.23 | 1.072 | 1.004 | 50.0 | 1.00 | 2.3 | stop | planar_or_panoramic_pairs |  |
| ctl_pan_47821 | 60/60 | 1 | 0.72 | 15.207 | 1.356 | 3.8 | 0.50 | 2.8 | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;pose_step_outlier |  |
| ctl_static_3129 | 59/60 | 99 | 0.28 | 6.731 | 80.091 | 8.5 | 0.92 | 5.3 | stop | hfov_out_of_range;fx_fy_mismatch;planar_or_panoramic_pairs |  |
| ctl_static_4283 | 0/60 |  |  |  |  |  | 1.00 |  | stop | no_model |  |
| gov_grca_hd08_9cb7c2 | 60/60 | 184 | 0.30 | 1.164 | 0.759 | 46.5 | 0.07 | 3.8 | stop | fx_fy_mismatch;focal_at_colmap_default |  |
| gov_grca_hd10_0121cb | 60/60 | 208 | 0.30 | 1.114 | 1.253 | 48.3 | 0.04 | 7.6 | stop | fx_fy_mismatch |  |
| gov_grca_hd10_8a4df3 | 60/60 | 179 | 0.47 | 1.435 | 1.277 | 38.4 | 0.48 | 7.5 | stop | fx_fy_mismatch;pose_step_outlier |  |
| gov_grca_hd10_c32a46 | 60/60 | 138 | 0.24 | 1.329 | 0.891 | 41.2 | 0.20 | 4.4 | stop | fx_fy_mismatch |  |
| ia_glasswalk | 60/60 | 32 | 1.09 | 1.471 | 1.016 | 37.5 | 0.25 | 3.0 | stop | points_per_image;reproj_gt_1px |  |
| ia_telangana | 60/60 | 222 | 0.38 | 0.723 | 1.010 | 69.3 | 0.28 | 8.0 | ok |  |  |
| ia_victory | 60/60 | 149 | 0.60 | 1.382 | 0.996 | 39.8 | 0.16 | 3.7 | ok |  |  |
| mk_21589 | 60/60 | 53 | 0.42 | 0.787 | 0.837 | 64.9 | 0.65 | 3.5 | stop | fx_fy_mismatch;planar_or_panoramic_pairs |  |
| mk_2789 | 60/60 | 162 | 0.54 | 1.087 | 1.316 | 49.4 | 0.15 | 10.5 | stop | fx_fy_mismatch |  |
| mk_3110 | 60/60 | 3 | 0.43 | 0.998 | 1.288 | 53.2 | 1.00 | 1.3 | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;low_parallax |  |
| mk_41574 | 60/60 | 154 | 0.34 | 0.814 | 0.830 | 63.1 | 0.15 | 3.6 | stop | fx_fy_mismatch |  |
| mk_4206 | 60/60 | 121 | 0.34 | 0.717 | 1.081 | 69.8 | 0.09 | 6.8 | ok |  |  |
| mk_43622 | 60/60 | 177 | 0.35 | 0.863 | 0.990 | 60.2 | 0.41 | 5.0 | ok |  |  |
| mk_50847 | 60/60 | 42 | 0.56 | 1.129 | 0.984 | 47.8 | 0.33 | 2.3 | stop | points_per_image |  |
| mk_5290 | 0/60 |  |  |  |  |  | 0.01 |  | stop | no_model |  |
| px_19834200 | 60/60 | 73 | 0.54 | 0.473 | 1.002 | 93.2 | 0.21 | 5.1 | ok |  |  |
| px_4072162 | 0/60 |  |  |  |  |  | 0.66 |  | stop | no_model |  |
| px_8435627 | 60/60 | 123 | 0.35 | 1.461 | 1.018 | 37.8 | 0.33 | 3.7 | ok |  |  |
| px_854417 | 60/60 | 98 | 0.39 | 0.837 | 1.055 | 61.7 | 0.33 | 5.4 | ok |  |  |

### B. Camera-model / prior variants against the silver reference (focal error and ATE are vs `ref`)

| variant | runs | registered >= 90 % | gate ok/suspect | silver-accurate | silent errors | of which kdis <= 13.2 deg | false stops | median abs f err % | median ATE % |
|---|---|---|---|---|---|---|---|---|---|
| v0a | 16 | 14 | 7 | 3 | 4 | 2 | 0 | 42.5 | 0.28 |
| v0b | 16 | 14 | 7 | 3 | 4 | 2 | 0 | 41.6 | 0.28 |
| v1sp | 16 | 16 | 13 | 3 | 10 | 4 | 0 | 32.4 | 0.33 |
| v2prior | 14 | 14 | 7 | 3 | 4 | 2 | 0 | 40.7 | 0.28 |
| v3fixed | 14 | 14 | 13 | 6 | 7 | 7 | 0 | 11.4 | 0.29 |
| v0a_hq960 | 3 | 2 | 2 | 0 | 2 | 0 | 0 | 113.0 | 2.74 |

### C. Per clip: focal error % / ATE % vs reference / gate (stop = S, suspect = s, ok = .)

| clip | ref f/W | v0a | v0b | v1sp | v2prior | v3fixed | v0a_hq960 |
|---|---|---|---|---|---|---|---|
| gov_grca_hd08_9cb7c2 | 1.038 | 48 / 0.4 / S | 48 / 0.4 / S | 13 / 0.4 / s | 50 / 0.4 / S | -23 / 1.1 / . |  |
| gov_grca_hd10_0121cb | 0.384 | 190 / 6.8 / S | 190 / 6.8 / S | 195 / 6.7 / . | 184 / 6.8 / S | 118 / 6.1 / . |  |
| gov_grca_hd10_8a4df3 | 0.266 | 440 / 22.5 / S | 436 / 22.4 / S | 416 / 22.8 / s | 441 / 22.4 / S | 213 / 22.7 / s |  |
| gov_grca_hd10_c32a46 | reference failed (2/30 frames) |  /  / S (no reference) |  /  / S (no reference) |  /  / . (no reference) |  /  / . (no reference) |  /  / . (no reference) |  |
| ia_glasswalk | 0.913 | 61 / 0.3 / S | 61 / 0.3 / S | 60 / 0.3 / S | 61 / 0.3 / S | 1 / 0.3 / s | 60 / 0.3 / s |
| ia_telangana | 0.729 | -2 / 0.0 / . | -2 / 0.0 / . | -1 / 0.0 / . | -2 / 0.0 / . | 12 / 0.5 / . |  |
| ia_victory | 1.010 | 37 / 0.3 / . | 37 / 0.3 / . | 37 / 0.3 / . | 37 / 0.3 / . | -7 / 0.1 / . |  |
| mk_21589 | 1.544 | -49 / 0.2 / S | -46 / 0.2 / S | -38 / 0.1 / S | -44 / 0.2 / S | -55 / 0.3 / S |  |
| mk_2789 | 0.861 | 26 / 0.5 / S | 26 / 0.5 / S | 28 / 0.7 / . | 26 / 0.5 / S | 5 / 0.6 / . |  |
| mk_3110 | 1.208 | -36 / 4.7 / S | -36 / 4.7 / S | -18 / 20.6 / S |  |  |  |
| mk_41574 | 1.151 | -29 / 0.1 / S | -29 / 0.1 / S | -26 / 0.1 / . | -29 / 0.1 / S | -30 / 0.1 / . |  |
| mk_4206 | 0.855 | -22 / 0.1 / . | -22 / 0.1 / . | -19 / 0.1 / . | -22 / 0.1 / . | 3 / 0.1 / . |  |
| mk_43622 | 0.870 | -1 / 0.0 / . | -1 / 0.0 / . | -1 / 0.0 / . | -1 / 0.0 / . | 3 / 0.2 / . |  |
| mk_50847 | reference failed (2/30 frames) |  /  / S (no reference) |  /  / S (no reference) |  /  / S (no reference) |  |  |  /  / S (no reference) |
| mk_5290 | 0.892 |  /  / S |  /  / S | -41 / 0.6 / . |  |  |  /  / S |
| px_19834200 | 0.475 | -1 / 0.0 / . | -1 / 0.0 / . | -0 / 0.0 / . | -1 / 0.0 / . | -5 / 0.1 / . |  |
| px_4072162 | 0.360 |  /  / S |  /  / S | 4 / 11.7 / S |  |  | 166 / 5.1 / . |
| px_8435627 | 0.779 | 87 / 0.4 / . | 87 / 0.4 / . | 85 / 0.4 / . | 87 / 0.4 / . | 11 / 0.2 / . |  |
| px_854417 | 0.460 | 82 / 5.4 / . | 56 / 5.4 / . | 65 / 5.4 / . | 63 / 5.4 / . | 42 / 5.4 / . |  |

### D. Hard negative controls: filter decision and stage-1 gate

| clip | category | filter decision | reg | pts/img | planar | fx/fy | gate | flags |
|---|---|---|---|---|---|---|---|---|
| ctl_crowd_4437 | outdoor_walk |  | 60/60 | 21 | 0.12 | 4.216 | stop | points_per_image;fx_fy_mismatch |
| ctl_drive_42368 | other |  | 60/60 | 2 | 0.09 | 14.320 | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;low_parallax |
| ctl_drive_45312 | other |  | 60/60 | 118 | 0.43 | 1.114 | stop | fx_fy_mismatch |
| ctl_drive_52454 | other |  | 60/60 | 21 | 0.03 | 6.441 | stop | points_per_image;fx_fy_mismatch |
| ctl_misc_14774 | other |  | 60/60 | 70 | 0.18 | 1.011 | ok |  |
| ctl_misc_3498 | other |  | 23/60 | 106 | 0.57 | 0.969 | stop | registered_lt_half;planar_or_panoramic_pairs |
| ctl_misc_45623 | other |  | 60/60 | 103 | 1.00 | 0.065 | stop | fx_fy_mismatch;planar_or_panoramic_pairs |
| ctl_night_40640 | other |  | 60/60 | 4 | 1.00 | 1.701 | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier;low_parallax |
| ctl_night_40746 | other |  | 0/50 |  | 1.00 |  | stop | no_model |
| ctl_night_4451 | other |  | 0/60 |  | 1.00 |  | stop | no_model |
| ctl_pan_4448 | other |  | 60/60 | 0 | 1.00 | 0.363 | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier |
| ctl_pan_45414 | other |  | 60/60 | 64 | 1.00 | 1.004 | stop | planar_or_panoramic_pairs |
| ctl_pan_47821 | other |  | 60/60 | 1 | 0.50 | 1.356 | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;pose_step_outlier |
| ctl_static_3129 | other |  | 59/60 | 99 | 0.92 | 80.091 | stop | hfov_out_of_range;fx_fy_mismatch;planar_or_panoramic_pairs |
| ctl_static_4283 | other |  | 0/60 |  | 1.00 |  | stop | no_model |
| mk_3110 | other |  | 60/60 | 3 | 1.00 | 1.288 | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;low_parallax |

### E. Focal length (f / W) three ways: replica fx, fy | silver reference | GeoCalib median (post-hoc, exploratory)

| clip | v0a fx | v0a fy | ref | GeoCalib | v0a vs ref % | v0a vs GeoCalib % | ref vs GeoCalib % | who agrees (10 %) |
|---|---|---|---|---|---|---|---|---|
| gov_grca_hd08_9cb7c2 | 1.164 | 1.534 | 1.038 | 0.795 | 48 | 93 | 31 | no two agree |
| gov_grca_hd10_0121cb | 1.114 | 0.889 | 0.384 | 0.839 | 190 | 33 | 54 | no two agree |
| gov_grca_hd10_8a4df3 | 1.435 | 1.124 | 0.266 | 0.833 | 440 | 72 | 68 | no two agree |
| gov_grca_hd10_c32a46 | 1.329 | 1.491 | 2.867 | 0.829 | 54 | 80 | 246 | no two agree |
| ia_glasswalk | 1.471 | 1.448 | 0.913 | 0.921 | 61 | 60 | 1 | reference = GeoCalib, replica differs |
| ia_telangana | 0.723 | 0.716 | 0.729 | 0.815 | 2 | 12 | 11 | replica = reference, GeoCalib differs |
| ia_victory | 1.382 | 1.387 | 1.010 | 0.938 | 37 | 48 | 8 | reference = GeoCalib, replica differs |
| mk_21589 | 0.787 | 0.939 | 1.544 | 0.702 | 49 | 34 | 120 | no two agree |
| mk_2789 | 1.087 | 0.826 | 0.861 | 0.904 | 26 | 20 | 5 | reference = GeoCalib, replica differs |
| mk_41574 | 0.814 | 0.980 | 1.151 | 0.805 | 29 | 22 | 43 | no two agree |
| mk_4206 | 0.717 | 0.663 | 0.855 | 0.881 | 22 | 25 | 3 | reference = GeoCalib, replica differs |
| mk_43622 | 0.863 | 0.872 | 0.870 | 0.892 | 1 | 3 | 2 | all three agree |
| px_19834200 | 0.473 | 0.472 | 0.475 | 0.453 | 1 | 4 | 5 | all three agree |
| px_8435627 | 1.461 | 1.435 | 0.779 | 0.866 | 87 | 69 | 10 | reference = GeoCalib, replica differs |
| px_854417 | 0.837 | 0.793 | 0.460 | 0.654 | 82 | 28 | 30 | no two agree |
