
# Admission Target Definition

## Source
CDC NHAMCS 2022 Emergency Department Public-Use Dataset.

## Prediction outcome
Hospital admission during the emergency department visit.

- Positive: ADMITHOS = 1
- Negative: ADMITHOS = 0
- Excluded: NODISP = 1 or invalid admission/disposition flags

## Cohort counts
- Source records: 16,025
- Excluded (no disposition): 106
- Eligible records: 15,919
- Admissions: 2,121
- Non-admissions: 13,798

## Observation handling
- OBSHOS = 1 occurs within ADMITHOS = 1 in this dataset.
- OBSDIS = 1 without admission is a negative outcome.
- Observation is not counted as an additional admission.

## Limitations
- Retrospective survey data, not real-time hospital records.
- Cohort eligibility may be refined before modeling.
- Outcome variables must never be included as predictors.
- Counts are unweighted; national estimates require survey-aware methods.

## Reference
CDC 2022 NHAMCS ED Public-Use Data File Documentation,
page 52:
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/doc22-ed-508.pdf
