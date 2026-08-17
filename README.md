# Active Learning for Precise Event Spotting (ALPES)

## Installation

Create the Conda environment for the implementation you plan to use. Each
environment file includes that implementation's Python and third-party
dependencies.

For F3Set and the F3ED active-learning experiments:

```sh
conda env create --file src/F3Set/environment.yml --name alpes-f3set
conda activate alpes-f3set
```

For AdaSpot:

```sh
conda env create --file src/AdaSpot/environment.yml --name alpes-adaspot
conda activate alpes-adaspot
```

Then install the shared source tree from the repository root:

```sh
python -m pip install -e . --no-deps
```

The editable install exposes both implementations as `src.AdaSpot` and
`src.F3Set`. The `--no-deps` flag is intentional: dependencies are managed by
the selected Conda environment rather than the root package.

## Usage

Run repository scripts as modules, for example:

```sh
python -m src.F3Set.train_f3set_f3ed --help
python -m src.AdaSpot.main --help
```

See the [F3Set](src/F3Set/README.md) and
[AdaSpot](src/AdaSpot/README.md) documentation for dataset preparation,
training, and evaluation instructions.
