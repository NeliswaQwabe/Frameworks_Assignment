# CORD-19 Analysis — Brief Report

## Summary
This project loads the CORD-19 metadata, performs cleaning and basic exploratory analysis, and generates visualizations to summarize publication activity and text features. Results and artifacts produced include time-series charts, top-journal rankings, word-frequency charts, and a word cloud of paper titles.

## Data cleaning & preparation
- Filled missing abstracts with empty strings; filled missing authors/journals with "Unknown".
- Removed rows with missing titles (treated as critical).
- Converted `publish_time` to datetime and extracted `publication_year`.
- Added `abstract_word_count` and `title_word_count` features.

## Key findings (high level)
- Publication counts vary by year; visualizations show clear temporal patterns (see publications-over-time chart).
- A relatively small set of journals account for a large share of records (top N journals output saved).
- Title word-frequency analysis highlights recurring thematic terms after basic stopword filtering.
- Word cloud visualizes the most prominent title terms.



## Limitations
- Metadata contains many missing and inconsistent values; results depend on available fields.
- Simple text preprocessing (basic stopwords, no stemming/lemmatization) limits NLP accuracy.
- Date parsing used a coercive approach — some publication dates may be lost (NaT).

## Recommendations / next steps
- Improve text preprocessing (tokenization, lemmatization, expanded stopwords).
- Deduplicate records and standardize journal names.
- Use sampling or incremental processing for very large datasets to reduce memory usage.


# Reflection

## What I did
- Loaded and inspected the CORD-19 metadata CSV.
- Implemented missing-value handling and basic cleaning rules.
- Converted dates and extracted publication years.
- Engineered simple text features (word counts).
- Performed exploratory analyses and created visualizations.
- Prepared a Streamlit app scaffold for interactive exploration.

## Challenges
- Heterogeneous metadata: many missing fields and inconsistent date formats required careful handling.
- Performance: reading and processing large CSVs can be memory intensive; caching and chunking may be needed.

## What I learned
- Small, consistent cleaning rules (fill vs drop) simplify downstream analysis.
- Visual checks (plots, word clouds) are effective for spotting data issues quickly.
- Streaming or incremental approaches help when working with large public datasets.

## Next learning goals
- Implement more robust NLP preprocessing (tokenization, stopword expansion, lemmatization).
- Add deduplication and normalization of journal/author names.
- Enhance Streamlit app with more interactive filters and downloadable summaries.
