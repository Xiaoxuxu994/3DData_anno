**easy** (18 clips): first stopped by rules 1, by the VLM 0, sent to the v1.3 annotator by the router 0, by the stage-1 gate 11; passed all 4 layers 6.

**control_negative** (16 clips): first stopped by rules 4, by the VLM 9, sent to the v1.3 annotator by the router 3, by the stage-1 gate 0; passed all 4 layers 0. Stopped by exactly one layer: rules 0, VLM 1, router 0, gate 0.

| clip | role | category | rules | rules_reason | vlm | vlm_reason | route | route_reason | gate | gate_flags | first_stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl_crowd_4437 | control_negative | too_dynamic | pass |  | stop | no_inaccurate | reject | filter | stop | points_per_image;fx_fy_mismatch | vlm |
| ctl_drive_42368 | control_negative | driving_low_parallax | pass |  | pass |  | v13 | vlm_driving | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;low_parallax | route |
| ctl_drive_45312 | control_negative | driving_low_parallax | pass |  | pass |  | v13 | dynamic;vlm_driving | stop | fx_fy_mismatch | route |
| ctl_drive_52454 | control_negative | driving_low_parallax | pass |  | pass |  | v13 | vlm_driving | stop | points_per_image;fx_fy_mismatch | route |
| ctl_misc_14774 | control_negative | fisheye | pass |  | stop | no_inaccurate | reject | filter | pass |  | vlm |
| ctl_misc_3498 | control_negative | cut | stop | cut | stop | no_unreconstructable | reject | filter | stop | registered_lt_half;planar_or_panoramic_pairs | rules |
| ctl_misc_45623 | control_negative | fisheye | pass |  | stop | no_unreconstructable | reject | filter | stop | fx_fy_mismatch;planar_or_panoramic_pairs | vlm |
| ctl_night_40640 | control_negative | dark_low_light | stop | dark | stop | no_inaccurate | reject | filter | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier;low_parallax | rules |
| ctl_night_40746 | control_negative | dark_low_light | pass |  | stop | no_inaccurate | reject | filter | stop | no_model | vlm |
| ctl_night_4451 | control_negative | static_camera | stop | text_overlay | stop | no_unreconstructable | reject | filter | stop | no_model | rules |
| ctl_pan_4448 | control_negative | no_parallax_rotation | pass |  | stop | no_unreconstructable | reject | filter | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;pose_step_outlier | vlm |
| ctl_pan_45414 | control_negative | no_parallax_rotation | pass |  | stop | no_unreconstructable | reject | filter | stop | planar_or_panoramic_pairs | vlm |
| ctl_pan_47821 | control_negative | no_parallax_rotation | pass |  | stop | no_unreconstructable | reject | filter | stop | points_per_image;hfov_out_of_range;fx_fy_mismatch;pose_step_outlier | vlm |
| ctl_static_3129 | control_negative | static_camera | stop | blur | stop | no_unreconstructable | reject | filter | stop | hfov_out_of_range;fx_fy_mismatch;planar_or_panoramic_pairs | rules |
| ctl_static_4283 | control_negative | static_camera | pass |  | stop | no_unreconstructable | reject | filter | stop | no_model | vlm |
| gov_grca_hd08_9cb7c2 | easy | drone_flythrough | pass |  | pass |  | glr |  | stop | fx_fy_mismatch;focal_at_colmap_default | gate |
| gov_grca_hd10_0121cb | easy | drone_orbit | pass |  | pass |  | glr |  | stop | fx_fy_mismatch | gate |
| gov_grca_hd10_8a4df3 | easy | drone_flythrough | pass |  | pass |  | glr |  | stop | fx_fy_mismatch;pose_step_outlier | gate |
| gov_grca_hd10_c32a46 | easy | drone_flythrough | pass |  | pass |  | glr |  | stop | fx_fy_mismatch | gate |
| ia_glasswalk | easy | outdoor_walk | pass |  | pass |  | glr |  | stop | points_per_image;reproj_gt_1px | gate |
| ia_telangana | easy | drone_orbit | pass |  | pass |  | glr |  | pass |  | passed_all |
| ia_victory | easy | indoor_walkthrough | pass |  | pass |  | glr |  | pass |  | passed_all |
| mk_21589 | easy | indoor_walkthrough | pass |  | pass |  | glr |  | stop | fx_fy_mismatch;planar_or_panoramic_pairs | gate |
| mk_2789 | easy | drone_orbit | pass |  | pass |  | glr |  | stop | fx_fy_mismatch | gate |
| mk_3110 | control_negative | no_parallax_rotation | pass |  | stop | no_unreconstructable | reject | filter | stop | points_per_image;fx_fy_mismatch;planar_or_panoramic_pairs;low_parallax | vlm |
| mk_41574 | easy | outdoor_walk | pass |  | pass |  | glr |  | stop | fx_fy_mismatch | gate |
| mk_4206 | easy | outdoor_walk | pass |  | pass |  | glr |  | pass |  | passed_all |
| mk_43622 | easy | drone_orbit | pass |  | pass |  | glr |  | pass |  | passed_all |
| mk_50847 | easy | outdoor_walk | pass |  | pass |  | glr |  | stop | points_per_image | gate |
| mk_5290 | easy | other | pass |  | pass |  | glr |  | stop | no_model | gate |
| px_19834200 | easy | indoor_walkthrough | stop | text_overlay | pass |  | reject | filter | pass |  | rules |
| px_4072162 | easy | drone_flythrough | pass |  | pass |  | glr |  | stop | no_model | gate |
| px_8435627 | easy | drone_flythrough | pass |  | pass |  | glr |  | pass |  | passed_all |
| px_854417 | easy | indoor_walkthrough | pass |  | pass |  | glr |  | pass |  | passed_all |
