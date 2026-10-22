# Goal: local ROC AUC0.83

The user explicitly clarified that0.83 is a LOCAL target. Do not substitute an official leaderboard score or claim that a small improvement completes this goal. Current verified V19 AUC: January26 .8179481062613656, January25 .8106447159499837.

Next hypothesis: contextual item preferences estimated from warm clicked histories. CurrentV19 uses item/timebucket labelled context but lacks direct warm item/daytype and joint context priors. This may be particularly useful for weekend validation with few or no weekend labels in preceding classifier blocks.

For each cutoff collect only each observed user's earliest pre-cutoff history, deduplicate identical warm(user,item,timestamp)records, and append all real pre-cutoff labelled clicks. Warm timestamps must strictly precede their user's first event; therefore they cannot overlap labelled events. Preserve distinct labelled events even if timestamp resolution ties. Verify timestamp-inferred bucket/daytype matches provided context codes. No unobserved warm exposures are invented.

Append32features to V19's258: log item total/bucket/daytype/joint counts; log-lifts of P(context|itemclick) against global clicked-context priors with smoothing5,20,100; three current-history matching-context lifts with smoothing5;16event-relative features. Stats use labels and histories strictly before each block, while current history features use provided past-only information.

Fixed599trees, depth7, LR0.08, seed42, unchanged ensemble weights. Train day6 on blocks3,4,5 and choose replacement0%,50%,100% of V19's raw aligned component using history-weighted AUC. Freeze for day5 on blocks2,3,4. Baseline is V19. Require both ordinary/weighted improvements on both dates and independent audit before final refit. Do not claim target achieved until local0.83 is verified; retain both-date reporting and do not mask the lower confirmation result. Reused validation and model selection bias remain limitations.
