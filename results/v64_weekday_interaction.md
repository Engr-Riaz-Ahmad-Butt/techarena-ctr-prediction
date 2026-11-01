# V64 weekday-interaction correction

- Protocol: train on day5 only, then evaluate the frozen day6 first-future-event cohort.
- Model: fixed CatBoost nonlinear correction over V53 features plus causal weekday/weekend, hour, bucket, and propensity interactions.
- Day6 ordinary AUC: V53 `0.824958`, V64 `0.825306`, gain `+0.000348`.
- Day6 history-weighted AUC: V53 `0.816661`, V64 `0.817005`, gain `+0.000344`.
- Day6 users: `35138`; day5/day6 user overlap `26404`; event-ID overlap `0`.
- 0.85 reached: `False`; official score: none.

## Validity notes

- This run preserves the previously frozen local day5-to-day6 benchmark; it is not an untouched new-date test because these dates have appeared in earlier repository experiments.
- Users can recur across the temporal boundary, but no day6 event IDs or labels enter model fitting; day6 features use only timestamps before the day6 cutoff.
- Day5 is Saturday and day6 Sunday in UTC; a single week provides little direct same-weekday history, so explicit weekday features do not eliminate calendar-distribution uncertainty.
- Day6 results are reported once under the fixed recipe and are not used to tune this experiment.
