# CAMS

Code accompanying *Attributable by Construction: Claim-Anchored Provenance for Multi-Document Summarization*.

## Contents

- `cams/` – main package
- `cams/evaluation/` – evaluation utilities
- `configs/` – settings used in the paper
- `scripts/` – entry points

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. AlignScore and SummaC are installed separately following their own repositories. Access to the backbone LLM and the NLI checkpoints listed in the configuration is required.

## Data

MultiNews, WCEP and DiverseSumm are publicly available. The localization experiments additionally rely on the alignments of Ernst et al. (2021). Inputs are expected in the format read by `cams/data.py`.

## Usage

The scripts in `scripts/` cover the stages of the pipeline and the evaluation. Behaviour is controlled by the configuration files and command-line options; values not listed in the paper may need adjusting to your setup.

## Annotations

The released claim–span records do not contain source text. `scripts/fill_quotes.py` restores the quote field from a local copy of MultiNews.

## Citation

Under review.
