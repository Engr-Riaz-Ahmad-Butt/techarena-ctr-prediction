# Aligned preference interaction experiment

User confirmed on 2026-10-22 that submission_v18.csv received official AUC 0.809189 (rank16 in supplied screenshot). Screenshot leader: 0.825973. Local AUC is not an official score forecast.

Baseline: V18; local January26 .8168002939057617, January25 .8082967729281301.

Add32 normalized coordinate interactions to the existing226features. Align each historical warm16 factor checkpoint to the fixed cut2 checkpoint using orthogonal Procrustes on row-normalized item vectors. Apply the same rotation to U,V,H. Fit rotations using only pre-cutoff checkpoints. Reference cut2 precedes every fitted block. Verify orthogonality and preservation of dot products. The32 features are16 user/item coordinate products and16 current-history/item products after row normalization. Their sums equal the existing cosine features.

Fixed599trees/depth7/LR.08/seed42. Train day6 on blocks3,4,5. Select component replacement0%,50%,100% by day6 history-weighted AUC, freeze for day5 trained on blocks2,3,4. Keep all other V18 mixture weights unchanged. Require both ordinary and history-weighted AUC to improve on both dates and a positive day6 conditional paired event interval; independent audit required. Reused validation windows do not establish generalization, and intervals exclude repeated-selection bias.

If accepted, final refit uses blocks4,5,6 and eval cut7; prepare version19 only after independent final checks. Otherwise preserve V18. No website submission or target-score guarantee.
