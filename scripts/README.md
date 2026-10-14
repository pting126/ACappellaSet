# Release utilities

Planned utilities:

- `download_models.py`: download a published checkpoint by registry ID and
  verify its SHA-256 checksum.
- `prepare_dataset.py`: validate the published dataset and prepare the inputs
  required by a selected training configuration.

These utilities will be added with the corresponding artifact registry and
dataset schema. There are no download or preparation commands in this initial
structure. The [Hugging Face model repository](https://huggingface.co/pting126/ACappellaSet-models)
exists, but checkpoint download URLs and the Zenodo dataset record remain pending.
