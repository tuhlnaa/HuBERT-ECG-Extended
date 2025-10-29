# REPRODUCIBILITY

Since for most dataset there isn't a pre-defined train-val-test split nor a fixed and known test set, we provide the splits we used to train, validate and test HuBERT-ECG to facilitate the replication of our results.
These splits are provided as csv files and are fully compatible with the underlying code.

## Notes on dataset labels
As described in the paper
- instances labelled with conditions appearing only once are dropped because these conditions are either unlearnable or untestable.
- instances labelled with conditions appearing only twice are placed only in the train and test sets to allow the model to try to learn and be evaluated on these conditions (within the limit of this setup).
- instances labelled with conditions appearing only three times are placed one in each split
- instances labelled with other conditions are placed in the train-val-test split following a randomly stratified selection using the functions in `utils.py`.

## Notes on folder contents
Train, test, and —optionally— val sets are reported in the form of `.csv` files and are easily usable with the provided code.
Files endings with {number}.csv indicate a specific folder to use in the cross-validation.
Details on how to fine-tune on specific dataset is provided in the paper.

## Notes on missing datasets
The folder for the Ribeiro dataset is not provided to respect non-disclosure terms of the data-user agreement. Nonetheless, the test set is predefined and publicly available. For this specific case, we can provide (upon request) fine-tuned models ready to be used for inference and evaluation.

## Cardio-Learning
Since Ribeiro dataset, a significant part of Cardio-Learning, cannot be disclosed, the results on the dataset are hardly to reproduce. However, to help use `hubert_ecg_{size}_cardio_learning`, we provide the ordered columns of the corresponding dataframe used to reference the Cardio-Learning examples.

## WHAT YOU HAVE TO DO
You simply have to download an ECG dataset and store its instances as .npy files with a name that matches the name provided in the .csv files in this folder.
Since labels are already provided, you don't have to 'find' them again within original .hea files.
Before use, make sure to preprocess your data following what's described in the paper.

## Example
You can find an example about how to fine-tune and evaluate a model on PTB-XL _All_ in the `.sh` scripts in the `code` folder.
Feel free to use the provided .csv files for this dataset.
