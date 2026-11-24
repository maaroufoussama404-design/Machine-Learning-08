# Data Card — UCI Adult (Census Income)

## Dataset identity

- **Name:** Adult / Census Income
- **Source:** UCI Machine Learning Repository, dataset 2
- **Download method:** `fetch_openml(name="adult", version=2, as_frame=True)`
- **Original data:** 1994 US Census-derived records
- **Expected size:** 48,842 rows
- **Inputs:** 14 features
- **Target:** `income`, representing income above USD 50,000 (`>50K`) or not (`<=50K`).

## Intended use

This dataset is used for educational experimentation with data ingestion,
validation, preprocessing, feature engineering and binary classification.

It is not appropriate for making real employment, lending, insurance,
or other consequential decisions about individuals.

## License and attribution

The UCI dataset page lists the dataset under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

Suggested citation:

Becker, B. and Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

Dataset page:
https://archive.ics.uci.edu/dataset/2/adult

## Features

The dataset contains numerical and categorical demographic and
socioeconomic attributes, including age, education, occupation,
relationship, race, sex, working hours and capital gains/losses.

The `fnlwgt` column is a census final weight and should not be
interpreted as an ordinary personal measurement.

## Known biases and limitations

- The data represents a historical US population and labor market
  from 1994, not the current population or economy.
- Income is influenced by structural, geographic, demographic and
  historical factors that are not fully represented by the features.
- The dataset contains demographic attributes such as race and sex.
  Models may reproduce or amplify historical disparities.
- The income threshold is historical and is not adjusted for inflation.
- `education` and `education-num` contain closely related information.
- Missing categorical values may appear as `?` in the source or as NaN after OpenML ingestion.
- The target classes are not equally represented. Accuracy alone can
  therefore hide poor performance on the less frequent class.

## Privacy and ethical considerations

The dataset contains demographic and socioeconomic information.
Although it is publicly available for research, combinations of
attributes can remain sensitive.

Do not attempt to identify individuals or link records to external
personal data. Do not use the resulting model to make decisions about
real people.

The 50-row manual-review sample should remain in the coursework
workspace unless submission requirements explicitly request it.

## Processing protocol

1. Raw data is ingested and saved without changing its values.
2. Schema, numeric types, ranges, missing tokens and duplicates are checked.
3. A stratified 70/30 train/test split is performed.
4. Exact duplicate records are removed from the training partition only.
5. Numeric median imputation and categorical mode imputation are fitted
   on training data only.
6. Numeric clipping thresholds at the 1st and 99th percentiles are
   learned from training data only.
7. Scaling and one-hot encoding are fitted on training data only.
8. Logistic Regression is evaluated with and without the engineered
   features `capital_net` and `hours_per_age`.
9. A SHA256 hash is recorded in `config.yaml`.

## Manual review

The pipeline exports 50 deterministic rows to
`results/manual_review_sample.csv`.

**Manual review is pending until a human has inspected all 50 rows**
against the UCI data dictionary and recorded the review status and notes.
The automated validation report is not a substitute for this review.


