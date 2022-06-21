# Exploring the Effectiveness of Self-supervised Learning and Classifier Chains in Emotion Recognition of Nonverbal Vocalizations

Official implementation of the paper "Exploring the Effectiveness of Self-supervised Learning and Classifier Chains in Emotion Recognition of Nonverbal Vocalizations" accepted by the Few-Shot track of the *ICML Expressive Vocalizations (ExVo) Competition 2022*.

>**Abstract:**<br>
We present an emotion recognition system for nonverbal vocalizations (NVs) submitted to the ExVo Few-Shot track of the ICML Expressive Vocalizations Competition 2022.
The proposed method uses self-supervised learning (SSL) models to extract features from NVs and uses a classifier chain to model the label dependency between emotions.
Experimental results demonstrate that the proposed method can significantly improve the performance of this task compared to several baseline methods.
Our proposed method obtained a mean concordance correlation coefficient (CCC) of $0.725$ in the validation set and $0.739$ in the test set, while the best baseline method only obtained $0.554$ in the validation set.

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

## Citation
Please kindly cite our work if you find our paper or code is helpful for you.


## LICENCE
MIT
