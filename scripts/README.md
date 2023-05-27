# Installation

To install the package, run the following command in the root directory of your project:

```bash
python -m ven venv
./venv/Scripts/activate
pip install -r requirements.txt
```

# Usage

To run the sentence deletion unsupervised approach, run the following command in the root directory of your project:

```bash
python -m .\scripts\run_sentence_deletion.py
```

To observe all the available options, run the following command in the root directory of your project:

```bash
python -m .\scripts\run_sentence_deletion.py --help
```

You can run the sentence deletion on the provided datasets under `data/ang_processed`, `data/slo_processed` and 
`data/labeled`. To visualize the results, run the following command in the root directory of your project:

```bash
python -m .\scripts\visualize_results.py --input_file <path_to_input_file>
```