#!/usr/bin/env python
# coding: utf-8

import json
import re
import torch
import unicodedata
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from pytorch_transformers import *
from typesql.utils import *
from retokenizer import Retokenizer

def concatenate_sql_data(sql_data_train, sql_data_val):
    sql_data_train.extend(sql_data_val)
    return sql_data_train

def count_context_toks(tok = 'the'):
    """
        Args: arbitrary token
        Return: number of times the token was used in different contexts
    """
    unique_toks = set()
    for sent_id in sent_idxs:
        string = tokenizer.decode(sent_id[0])
        string = string.split()
        if tok in string:
            idx = string.index(tok)
            unique_toks.add(sent_id[0][idx])
    return len(unique_toks)

def extract_questions(sql_data, tokenize = True):
    key = 'question_tok' if tokenize else 'question'
    return list(map(lambda el:el[key], sql_data))

def bert_preprocessing(questions, tok2ids_tuple = False, flatten = False):
    """
        Args: Raw natural language questions represented as strings. 
        Computation: Sentence preprocessing steps necessary for BERT model.
                    Each sentence is required to be preceded by a special [CLS] token
                    and followed by a special [SEP] token.
                    Token IDs arrays have to be converted into tensors 
                    before they can be passed to BERT. 
        Return: tokenized questions, token IDs, segment IDs (i.e., ones),
                tuples of (tokens, ids) either per token-id-pair or as a list per sentence.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    questions = list(map(lambda q: '[CLS]' + ' ' + ' '.join(q) + ' ' +'[SEP]', questions))
    tok_questions = [tokenizer.tokenize(q) for q in questions]
    indexed_tokens = [torch.tensor([tokenizer.convert_tokens_to_ids(tok_q)], dtype=torch.long) \
                      for tok_q in tok_questions]
    segment_ids = [torch.tensor([np.ones(len(q), dtype=int)],dtype=torch.long) \
                   for q in tok_questions]
    
    idx2word = {idx.item(): tok_w for tok_q, indexes in zip(tok_questions, indexed_tokens) \
                for tok_w, idx in zip(tok_q, indexes[0])}
    
    return tok_questions, indexed_tokens, segment_ids, idx2word
      

#TODO: investigate how you can make this function work properly 
#FOR NOW: just use token representations of last hidden layer (implemented in cell below)
def get_summed_embeddings(model, toks_ids, segment_ids):
    """
        Args: BertModel, token id tensors, segment id tensors
        Computation: Convert the hidden state embeddings into single token vectors
                     Holds the list of 12 layer embeddings for each token
                     Will have the shape: [# tokens, # layers, # features]
        Return: Bert context embedding for each token in the question.
                Final token embedding is the sum over the last four hidden layer representions.
    """
    # encoded_layers = model(toks_ids, segment_ids)[0]
    encoded_layers, _ = model(toks_ids)[-2:]
    token_embeddings = np.zeros((len(toks_ids[0]), 768), dtype=float)
    hidden_size = 12
    batch_i = 0
    for token_i in range(len(toks_ids[0])):
        # 12 layers of hidden states for each token
        hidden_layers = np.zeros((hidden_size, 768), dtype=float) 
        for layer_i in range(len(encoded_layers)):
            hidden_layers[layer_i] = encoded_layers[layer_i][batch_i][token_i]
        # each token's embedding is represented as a sum of the last four hidden layers
        token_embeddings[token_i] = torch.sum(torch.stack(hidden_layers[token_i])[-4:], 0).numpy()
    return token_embeddings

def bert_token_ids(sql_data, bert_questions, bert_ids, arbitrary_id = 99999):
    rejoined_toks = list()
    rejoined_ids = list()
    retokenizer = Retokenizer(merge=None, embeddings=False)
    for i, (question, bert_question, bert_id) in enumerate(zip(sql_data, bert_questions, bert_ids)):
        bert_id = list(bert_id[0].numpy())
        new_toks, new_ids, new_id = retokenizer.retokenize(question['question_tok'], bert_question[1:-1], bert_id[1:-1], arbitrary_id)
        arbitrary_id = new_id
        rejoined_toks.append(new_toks)
        rejoined_ids.append(new_ids)
    return rejoined_toks, rejoined_ids

def bert_embeddings(bert_questions, bert_ids, segment_ids, sql_data, merge, arbitrary_id = 99999):
    """
        Args: WordPiece tokenized questions, torch tensors of token ids and segment ids and SQL data.
        Computation: load pre-trained BERT model (weights),
                     put the model in "evaluation" mode, meaning feed-forward operation.
                     "torch.no_grad()" deactivates the gradient calculations, 
                     saves memory, and speeds up computation (we don't need gradients or backprop).
        Return: dictionary that maps token ids (keys) 
                to their corresponding BERT context embeddings (values).
    """
    #TODO: check what's necessary to ouput all hidden states
    #model = BertModel.from_pretrained('bert-base-uncased',
    #                              output_hidden_states=True,
    #                              output_attentions=True)
    
    model = BertModel.from_pretrained('bert-base-uncased')    
    model.eval()
    id2embed = dict()
    id2tok = dict()
    rejoined_toks = list()
    rejoined_ids = list()
    retokenizer = Retokenizer(merge=merge, embeddings=True)
    with torch.no_grad():
        for i, (question, bert_question, bert_id, segment_id) in enumerate(zip(sql_data, bert_questions, bert_ids, segment_ids)):
            bert_embeddings = model(bert_id, segment_id)[0][0]
            bert_embeddings = np.array(list(map(lambda embedding:embedding.numpy(), bert_embeddings)))
            
            #TODO: make "get_summed_embeddings" function work
            #token_embeddings = get_summed_embeddings(model, tok_id, segment_id)
            
            bert_id = list(bert_id[0].numpy())
            new_toks, new_ids, new_embeddings, new_id = retokenizer.retokenize(question['question_tok'], 
                                                                               bert_question[1:-1], 
                                                                               bert_id[1:-1], 
                                                                               arbitrary_id,
                                                                               bert_embeddings)
            arbitrary_id = new_id   
            try:
                assert len(new_ids) == len(new_toks) == len(new_embeddings) == len(question['question_tok'])
                for tok_id, bert_tok, bert_embedding in zip(new_ids, new_toks, new_embeddings):
                    if tok_id not in id2embed:
                        id2embed[tok_id] = bert_embedding
                    if tok_id not in id2tok:
                        id2tok[tok_id] = bert_tok
            except AssertionError:
                pass
            
            rejoined_toks.append(new_toks)
            rejoined_ids.append(new_ids)
            
    return rejoined_toks, rejoined_ids, id2embed, id2tok

def drop_data(tok_questions, tok_ids, sql_data, idx_to_drop):
    k = 0
    n_errors = len(idx_to_drop)
    n_questions = len(sql_data)
    for idx in idx_to_drop:
        tok_questions.pop(idx-k)
        tok_ids.pop(idx-k)
        sql_data.pop(idx-k)
        k += 1
    assert len(sql_data) == n_questions-n_errors, 'Incorrect number of erroneous questions was dropped'
    return tok_questions, tok_ids, sql_data

def update_sql_data(sql_data):
    """
        Args: SQL dataset
        Return: Updated SQL dataset with bert tokens and corresponding bert ids
                BERT tokens were rejoined into TypeSQL's gold standard tokens and
                hence are the same
    """
    bert_questions, bert_ids, _, _ = bert_preprocessing(extract_questions(sql_data))
    tok_questions, tok_ids = bert_token_ids(sql_data, bert_questions, bert_ids)
    idx_to_pop = list()
    n_original_questions = len(sql_data)
    print("Number of questions before computing BERT token representations:", n_original_questions)
    for i, (question, tok_id, tok_question) in enumerate(zip(sql_data, tok_ids, tok_questions)):
        try:
            assert len(question['question_tok']) == len(tok_id)  == len(tok_question)
        except:
            idx_to_pop.append(i)
        
    tok_questions, tok_ids, sql_data = drop_data(tok_questions, tok_ids, sql_data, idx_to_pop) 
    
    for i, (question, tok_id, tok_question) in enumerate(zip(sql_data, tok_ids, tok_questions)):
        assert len(sql_data[i]['question_tok']) == len(tok_id)  == len(tok_question), "Removing incorrectly rejoined questions did not work. Check function!"
        sql_data[i]['bert_tokenized_question'] = tok_question
        sql_data[i]['bert_token_ids'] = tok_id #list(tok_id[0].numpy())

    n_removed_questions = n_original_questions-len(sql_data)
    
    print("Number of questions in pre-processed dataset (after rejoining):", len(sql_data))
    print("Questions that could not be rejoined into TypeSQL tokens:", n_removed_questions)
    print("{}% of the original questions were removed".format(round((n_removed_questions / n_original_questions)*100, 2)))
    return sql_data

def remove_nonequal_questions(sql_data):
    count = 0
    for i, question in enumerate(sql_data):
        try:
            assert question['question_tok'] == question['bert_tokenized_question']
        except AssertionError:
            sql_data.pop(i)
            count += 1
    print("{} questions had different tokens and thus were removed from dataset".format(count))
    print("SQL data has been updated with BERT ids (tokens are the same as TypeSQL's tokens)")
    print()
    return sql_data

def reduce_dimensionality(id2embed, dims_to_keep=100):
    """
        Args: id2embedding dict; number of dimensions to keep (of original 768 BERT embeddings)
        Return: id2embdding dict with reduced dimensionality embeddings specified by dims-to-keep 
    """
    ids = []
    embeddings = np.zeros((len(id2embed), 768))
    id2embed = dict(sorted(id2embed.items(), key=lambda kv:kv[0], reverse=False))
    for i, (idx, embedding) in enumerate(id2embed.items()):
        ids.append(idx)
        embeddings[i] = embedding
        
    pca = PCA(n_components=dims_to_keep, svd_solver='auto', random_state=42)
    embeddings = pca.fit_transform(embeddings)
    
    id2embed_reduced = {idx:embedding for idx, embedding in zip(ids, embeddings)}
    return id2embed_reduced

def bert_pipeline(sql_data_train, sql_data_val, merge='max'):
    sql_data = concatenate_sql_data(sql_data_train, sql_data_val)
    tok_questions, tok_ids, segment_ids, _ = bert_preprocessing(extract_questions(sql_data))
    _, _, id2embed, id2tok = bert_embeddings(tok_questions, tok_ids, segment_ids, sql_data, merge)
    assert len(id2embed) == len(id2tok)
    id2embed = reduce_dims(id2embed)
    return id2tok, id2embed

def save_embeddings_as_json(id2tok, id2embed, merge, full=False):
    # np.arrays have to be converted into lists to be .json serializable
    id2embed = {int(idx):embedding.tolist() for idx, embedding in id2embed.items()}
    id2tok = {int(idx):tok for idx, tok in id2tok.items()}
    if merge == 'max':
        if dim == 'full':
            embeddings = './bert/id2embedMaxFull.json'
        elif dim == 600:
            embeddings = './bert/id2embedMax600.json'
        elif dim == 100:
            embeddings = './bert/id2embedMax100.json'
        ids = './bert/id2tokMax.json'
    elif merge == 'avg':
        if dim == 'full':
            embeddings = './bert/id2embedMeanFull.json'
        elif dim == 600:
            embeddings = './bert/id2embedMean600.json'
        elif dim == 100:
            embeddings = './bert/id2embedMean100.json'
        ids = './bert/id2tokMean.json'
    elif merge == 'sum':
        if dim == 'full':
            embeddings = './bert/id2embedSumFull.json'
        elif dim == 600:
            embeddings = './bert/id2embedSum600.json'
        elif dim == 100:
            embeddings = './bert/id2embedSum100.json'
        ids = './bert/id2tokSum.json' 
    else:
        raise Exception('Embeddings have to be max-pooled, averaged or summed')
    with open(embeddings, 'w') as json_file:
        json.dump(id2embed, json_file)
    with open(ids, 'w') as json_file:
        json.dump(id2tok, json_file)
        
def load_bert_dicts(file_tok, file_emb):
    with open(file_tok) as f:
        id2tok = json.loads(f.read())
    with open(file_emb) as f:
        id2embed = json.loads(f.read())
    id2tok = {int(idx):tok for idx, tok in id2tok.items()}
    id2embed = {int(idx):np.array(embedding) for idx, embedding in id2embed.items()}
    assert len(id2tok) == len(id2embed)
    return id2tok, id2embed

def plot_accs(n_epochs, train_accs, val_accs):
    plt.plot(n_epochs, train_accs, color='blue')
    max_train = np.argmax(train_accs)
    label = "Train: {:.2f}%, E: {}".format(train_accs[max_train]*100, n_epochs[max_train])
    plt.annotate(label, # text
                 (max_train+1, train_accs[max_train]),
                 textcoords="offset points",
                 xytext=(0,15),
                 ha='center')
    plt.plot(val_accs, color='orange')
    max_val = np.argmax(val_accs)
    label = "Dev: {:.2f}%, E: {}".format(val_accs[max_val]*100, n_epochs[max_val])
    plt.annotate(label,
                 (max_val+1, val_accs[max_val]),
                 textcoords="offset points",
                 xytext=(0,5),
                 ha='center')
    plt.title('TypeSQL learning curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show(block=False)

def plot_losses(n_epochs, losses):
    plt.plot(n_epochs, losses, color='blue')
    plt.title("TypeSQL's learning curve during training")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show(block=False)