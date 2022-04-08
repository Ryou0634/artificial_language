# Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models

This repository contains code to generate data from the artificial languages described in our paper, [Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models](https://arxiv.org/abs/2203.10326).

## Installation
The required python versions are `>=3.8,<3.11`.
After setting up a python virtual environment, install the dependencies with [poetry](https://python-poetry.org/).

```bash
poetry install
```

## Usage
The artificial languages is configured by allennlp-like jsonnet files.
You can find the configuration files of some artificial languages under [`artificial_language/configs`](artificial_language/configs).

### Example
```bash
# Usage: python generate_artificial_corpus.py COFIG-PATH SAVE-PATH NUM-SENTENCES

# the Uniform language
poetry run python generate_artificial_corpus.py artificial_language/configs/no_structure_uniform.jsonnet data/uniform.txt 10000

# the Zipf Nesting-Dep language
poetry run python generate_artificial_corpus.py artificial_language/configs/dependency_nesting_zipf.jsonnet data/dependency_nesting_zipf.txt 10000
```