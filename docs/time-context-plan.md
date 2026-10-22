# Fallback: timestamp-conditioned preference profiles

Prepared before observing aligned experiment scores. Run only if aligned is rejected or its January26 ordinary gain is below0.001 AUC. Baseline remains V18, not a selectively changed blend.

## Explicit protocol amendment before any context validation result

Aligned day6 improved by0.001147812 ordinary and0.001071564 weighted AUC, exceeding the original compute cutoff. The experiment scope is expanded to test the distinct timestamp hypothesis anyway: the official gap to the screenshot leader is0.016784, so the first local gain does not resolve the user's competitive objective. This expansion was made after seeing aligned day6 and before any context validation AUC. The original cost gate is therefore not followed. No context AUC has been measured; only a10000-row feature-shape/finite/causality smoke check was run.

If aligned confirms on both dates and passes independent review, require context day6 weighted AUC to exceed the confirmed aligned day6 weighted score before spending a day5 refit. Otherwise the selection floor is V18. This floor is frozen before context training, and all model settings and mixture grids remain unchanged. Select the final variant by day6 weighted AUC among independently confirmed candidates; do not tune using day5. Report failed and skipped stages too. Expanded search further increases reused-validation selection bias.

The existing226features omit elapsed click timestamps. Add32 scalar features from observed histories only:8 activity/age statistics; four H-factor profiles weighted by6h decay,24h decay, matching6h time-of-day bucket and3h circular hour kernel; each profile supplies candidate dot product, cosine, log weight mass and weighted repeat count. Add event-relative mean/max differences for the four dot products. All timestamp entries must precede the scored event. Checkpoints remain strictly before each fitted block.

Use the same599trees/depth7/LR.08/seed42 and causal train blocks as aligned. Select only replacement0%,50%,100% of V18 decomposition component on day6 history-weighted AUC, freeze for day5. Require both ordinary and reweighted improvements on both dates and independent audit before final refit. If both feature variants pass, compare their frozen day6 weighted AUC to select one; report both dates and all trials honestly. This is further reused-validation model selection, not an unbiased holdout. Official AUC cannot be guaranteed.
