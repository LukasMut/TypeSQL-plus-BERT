Most of the code is based on [TypeSQL](https://github.com/taoyds/typesql) and [SQLNet](https://github.com/xiaojunxu/SQLNet). 
Please cite both TypeSQL and SQLNet if you use this code.

## TypeSQL with BERT ensemble

BERT byte-pair encoded tokens had to be rejoined into WikiSQL tokens as otherwise the SQL generation task will not work (due to BERT tokenizer's sub-word splitting). See files `retokenizer.py` and `bert_utils.py` for implementation.


Moreover, I extended TypeSQL's single model approach with an ensemble method (according to soft-voting) - both TypeSQL nets are learning simultaneously (forward step and backprop are computed for both nets) but only one of the two will perform the final translation task (always the same network) given the averaged predictions of both sytems. 

## TypeSQL

Source code accompanying TypeSQL's NAACL 2018 paper:[TypeSQL: Knowledge-based Type-Aware Neural Text-to-SQL Generation
](https://arxiv.org/abs/1804.09769)

#### Environment Setup

1. The code uses Python 3.7, [Pytorch 1.1.0](https://pytorch.org/previous-versions/) and [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers).
2. Install Python dependency: `pip install -r requirements.txt`
3. Install Pytorch: `pip install pytorch`
4. Install Pytorch Transformers: `pip install pytorch-transformers`. Go to their repo for more information about requirements and dependencies.

#### Download Data and Embeddings

1. Download the zip data file at the [Google Drive](https://drive.google.com/file/d/1CGIRCjwf2bgmWl3UyjY1yJpP4nU---Q0/view?usp=sharing), and put it in the root dir.
2. Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and the [paraphrase embedding](https://drive.google.com/file/d/1iWTowxEG1-KZyq-fHP6cb6dNqMh4eHiN/view?usp=sharing) `para-nmt-50m/data/paragram_sl999_czeng.txt`. Put the unziped glove and para-nmt-50m folders in the root dir.

Neither do we use BERT embeddings to predict aggregate values in the SELECT clause nor to compute Type embeddings - as in both cases there is no context to disentangle. Thus, `GloVe` and `paraphrase` embeddings are still crucial to TypeSQL.

3. Use the pre-trained BERT model ('uncased') from [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers).

#### Train Models

1. To use knowledge graph types:
```
  mkdir saved_model_kg_single (if ensemble, also: mkdir saved_model_kg_second)
  python train.py
 --sd_1 saved_model_kg_single (set model save directory for single model)
 --sd_2 saved_model_kg_second (set save directory for second model, if ensemble computation)
 --BERT True (False, if you want to use GloVe)
 --types True (False, if you want to use BERT embeddings only - no concatenation with type embeddings)
 --merged (use max-pooled, averaged or summed bert embeddings)
 --ensemble (single model, mixed ensemble (GloVe and BERT), homogeneous ensemble (e.g., (GloVe and GloVe) XOR (BERT and BERT)))
```

2. To use DB content types:
```
   mkdir saved_model_con_single (if ensemble, also: mkdir saved_model_con_second)
   python train.py
  --sd_1 saved_model_con_single (set model save directory for single model)
  --sd_2 saved_model_con_second (set save directory for second model, if ensemble computation)
  --db_content 1
  --BERT True (False, if you want to use GloVe)
  --types True (False, if you want to use BERT embeddings only - no concatenation with type embeddings)
  --merged (use max-pooled, averaged or summed bert embeddings)
  --ensemble (single model, mixed ensemble (GloVe and BERT), homogeneous ensemble (e.g., (GloVe and GloVe) XOR (BERT and BERT)))
```
 
  
#### Test Models

1. Test Model with knowledge graph types:
```
python test.py --sd saved_model_kg
```
2. Test Model with knowledge graph types:
```
python test.py --sd saved_model_con --db_content 1
```

#### Get Data Types

1. Get a Google Knowledge Graph Search API Key by following the [link](https://developers.google.com/knowledge-graph/)
2. Search knowledge graph to get entities:
```
python get_kg_entities.py [Google freebase API Key] [input json file] [output json file]
```
3. Use detected knowledge graph entites and DB content to group questions and create type attributes in data files:
```
python data_process_test.py --tok [output json file generated at step 2] --table TABLE_FILE --out OUTPUT_FILE [--data_dir DATA_DIRECTORY] [--out_dir OUTPUT_DIRECTORY]

python data_process_train_dev.py --tok [output json file generated at step 2] --table TABLE_FILE --out OUTPUT_FILE [--data_dir DATA_DIRECTORY] [--out_dir OUTPUT_DIRECTORY]
```

#### Acknowledgement

The implementation is based on [TypeSQL](https://github.com/taoyds/typesql) and [SQLNet](https://github.com/xiaojunxu/SQLNet). Please cite it too if you use this code.
