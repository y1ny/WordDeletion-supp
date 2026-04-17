We provided the raw data (responses from LLMs) for all experiments, as well as trial data (the full prompt we gave to LLMs) for exp2 and exp5 (since other experiments rely on copyrighted treebank data). We also provided a script to generate the preprocessed data from the raw data, and compare the data with previously released data in our [Github repo](https://github.com/y1ny/WordDeletion) (also see our complete process and analysis scripts in that repo).

`./exp*/*/raw/*.csv` contains the raw data. Each CSV file includes at least the following columns: the demonstration `demonstration`, the test sentence `sentence`, and ChatGPT's raw output `response`. The column structure varies by experiment:

- **exp1, exp3, exp4**: columns are `demonstration`, `sentence`, `model_type`, `stop_type`, `response`. The prompt is not included because it requires syntactic annotations from treebanks, which are copyrighted materials. With access to the treebanks, prompts can be reproduced using our scripts in [Github repo](https://github.com/y1ny/WordDeletion) (e.g., `scripts/process_ptb.py` and `exp1/construct_test.py` for exp1).
- **exp5**: contains two sets of data. `english/` and `chinese/` correspond to the version released on GitHub. `english-new/` and `chinese-new/` correspond to the final published version (revised during peer review).
  - `english/` and `chinese/` use treebank data as demonstrations and meaningless sentences as test sentences, consistent with the [GitHub repo](https://github.com/y1ny/WordDeletion). Since the prompts rely on copyrighted treebank data, they are not included. Columns: `demonstration`, `sentence`, `model_type`, `stop_type`, `response`.
  - `english-new/` and `chinese-new/` use parallel sentences as demonstrations and meaningless sentences as test sentences, consistent with the final published paper. Full prompts are provided. Columns: `prompt`, `demonstration`, `sentence`, `model_type`, `stop_type`, `response`.
- **exp2**: columns are `prompt`, `demonstration`, `sentence`, `model_type`, `stop_type`, `response`.
- **exp6**: columns are `prompt`, `demonstration`, `sentence`, `is_plausible`, `structure`, `response`. Exp6 does not contain Chinese experiments.

`./exp*/*/github/*.csv` contains the data released in the [GitHub repo](https://github.com/y1ny/WordDeletion).

`./preprocess_and_verify.py` extracts preprocessed data from the raw data in `./exp*/*/raw/*.csv`, and verifies them against `./exp*/*/github/*.csv` to ensure consistency.

## Requirements

- Python 3
- pandas

Install dependencies:

```bash
pip install pandas
```

## Usage

Run the script:

```bash
python preprocess_and_verify.py
```
