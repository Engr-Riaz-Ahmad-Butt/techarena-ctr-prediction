# Calendar diagnosis and rejected candidates

The user confirms V36 and V39 have not been uploaded. The last confirmed official result remains V35 = 0.816623. V39 remains the best validated full ensemble: day6 ordinary AUC 0.8249578084, history-adjusted AUC 0.8166614879; day5 adjusted AUC 0.8105265996. These are diagnostic local results, not forecasts of official scores.

V40 recent-history posterior updates, V41 nonlinear expert replacement, and V44 three-day context adaptation all failed the frozen full-ensemble promotion gate. None was exported. V43's fixed unseen-calendar fallback improved the reused day5 ensemble, but failed the fresh component check (ordinary delta -0.0007207311, joint-adjusted delta -0.0000518170). Its exporter intentionally stopped before CSV creation. It is not an approved candidate.

The final V35/V36 context exporters fit only days5/6 (Saturday/Sunday), leaving the weekday context matrix exactly zero. Evaluation is 97.1987% weekdays, while both previous validation dates are weekends. This is a verified support mismatch; its effect on official AUC has not been established. Copying the observed matrix to the unseen level did not pass validation, so this diagnosis alone does not justify changing the submission.

V42 introduces a first-future-event calendar benchmark at cutoff4 using factors cut2 and fitting days2/3. It compares expert components only. `audit_v42_frozen_history.py` explicitly checks and removes any clicks after cutoff, then replays the fixed saved models. The audit found zero post-cutoff entries in selected histories and reproduced the original AUCs. Neither expert AUC represents the full submission model. The neural expert has higher raw AUC here, but its V41 full-ensemble replacement already failed the old validation gate.

Only one fit ran at a time, with four Torch threads. Failed candidates preserve reproducible source, input hashes and reports. No rank or 0.83/0.85 achievement is claimed.
