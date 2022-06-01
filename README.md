# Exploring the Effectiveness of Self-supervised Learning and Classifier Chains in Emotion Recognition of Nonverbal Vocalizations

This respository contains the code of the paper "Exploring the Effectiveness of Self-supervised Learning and Classifier Chains in Emotion Recognition of Nonverbal Vocalizations" submitted to the Few-Shot track of the ICML Expressive Vocalizations (ExVo) Competition.

## Setup
Please follow the following steps to setup your environment
1. clone this repo
2. install [poetry](https://python-poetry.org/)
3. run `poetry install`
4. enter the virtual env by `poetry shell`
5. prepare the data (wav files and the features provided by the organizer) in your local disk

## Preprocess
1. trim silence in wav files
```bash
python3 preprocess.py --src_dir /path/to/wav --tgt_dir /path/to/output/dir
```

2. create filelists
```bash
python3 create_splits.py --data_file_path=/path/to/data_info.csv --save_path=./filelists
```

3. create test filelists (optional)
If you want to fine-tune the model on test samples, please run:
```bash
python3 create_splits.py --data_file_path=/path/to/data_info.fewshot.csv --save_path=./filelists --test
```

4. create cross-validation filelists (optional)
If you want to do cross-validation, please run:
```bash
python3 create_cv.py
```

## Training
Run the following cmd to train the model.
```bash
python3 train.py
```
This will train the model with default setting using Wav2vec-base as feature extractor and linear classifier chain as the predictor.
If anything goes well, you should get a mean CCC of about 0.67 on the validation set.

If you want to train other models or modify the parameters, please refer the config files under the `config` dir.

## Cross-validation

```bash
git checkout cv
python3 cv.py
```
This will run cross-validation with default setting

## LICENCE
MIT
