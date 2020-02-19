# LS-Tree: Feature Attribution and Interaction Detection for NLP Models

Code for replicating the experiments.

## Dependencies
The code runs with Python 2.7 and requires Tensorflow 1.11.0, Keras 2.2.4 and nltk 3.3. Please `pip install` the following packages:
- `numpy`
- `scipy`
- `pandas`
- `tensorflow` 
- `nltk`
- `keras`
- `scikit-learn`
- `networkx`


For experiments with LSTM and BERT, access to GPU is required. 

Download Stanford Parser is required for all experiments. The following is an example of how to run it (in the background).

```shell
###############################################
# Download from source.
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
# Unzip.
unzip stanford-corenlp-full-2018-02-27.zip -d ./

# Run the parser in the background.
cd stanford-corenlp-full-2018-02-27
java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000  >/dev/null 2>&1 &
###############################################
```

## Training of BERT.
For replicating experiments related to BERT, please follow the steps to download and train the model first:

Download and unzip the pre-trained BERT model.
```shell
###############################################
# Cd to the target directory. 
mkdir bert/models/
cd bert/models/
# Download
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
# Unzip.
unzip uncased_L-12_H-768_A-12.zip -d ./
###############################################
```

Train a BERT for the evaluation of average depth and linear correlation.
```shell
###############################################
cd bert/
export BERT_BASE_DIR=models/uncased_L-12_H-768_A-12
mkdir models/sst_output
python run_classifier.py   --task_name=SST-2   --do_train=true   --do_eval=true   --data_dir=glue_data/SST-2   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=models/sst_output/ 
###############################################
```


Train BERT and save every epoch for the experiment of overfitting.
```shell
###############################################
cd bert/ 
export BERT_BASE_DIR=models/uncased_L-12_H-768_A-12
mkdir models/sst_train_vs_test
python run_classifier.py   --task_name=SST-train  --do_train=true   --do_eval=true   --data_dir=glue_data/SST-train   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=5e-6   --num_train_epochs=9.0   --output_dir=models/sst_train_vs_test/ --save_checkpoints_steps 205
###############################################
```

## Evaluation of average depth and linear correlation
We provide as an example the source code to run experiments on SST in the paper. 

The evaluation of average depth and linear correlation can be carried out by the following steps:

Train BoW, CNN,and LSTM. (See the above separate section for the training of BERT.)
```shell
python bow.py
python explain.py --task train --model cnn
python explain.py --task train --model lstm
```

Explain each of the model and store scores.
```shell
python explain.py --task explain --model bow
python explain.py --task explain --model cnn
python explain.py --task explain --model lstm
python explain.py --task explain --model bert
```

Compute average correlation with BoW and average depth, and plot figures.
```shell
python nonlinearity.py
```
Figures will be stored in figs/

The experiment with CNN is the fastest to carry out and is expected to be finished around 60-120 min on a Tesla K80 GPU.

## Demo
Create demo for BoW, CNN, LSTM, BERT that visualizes interaction scores from the above experiment can be created with the following steps:
```shell
# A trained BoW / CNN / LSTM / BERT following the steps in the above sections is required.
python explain.py --task demo --model bow/cnn/lstm/bert
```
Trees colorized with interaction scores will be stored in figs/demo/


## Overfitting detection
Overfitting detection for CNN, LSTM and BERT can be carried out with the following steps:
Train CNN/LSTM for a fixed number of epochs. See the above separate section for the training of BERT.
```shell
python train_vs_test.py --task train --model_name cnn
python train_vs_test.py --task train --model_name lstm
```

Compute feature attribution and interaction scores for models at each epoch.
```shell
python train_vs_test.py --task generate_scores --model_name cnn
python train_vs_test.py --task generate_scores --model_name lstm
python train_vs_test.py --task generate_scores --model_name bert
```


Evaluate loss, calculate variance, carry out permutation tests, and plot the results. Figures will be stored in figs/
```shell
python train_vs_test.py --task compute_and_plot --model_name cnn
python train_vs_test.py --task compute_and_plot --model_name lstm
python train_vs_test.py --task compute_and_plot --model_name bert
```

The experiment with CNN is the fastest to carry out and is expected to be finished around 60-120min on a Tesla K80 GPU.















 
