#!/usr/bin/env python
# coding: utf-8

import calendar
import json
import re
import torch
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
      
def get_chunk(indexes, span, i):
    start = indexes[-1] if len(indexes) > 0 else 0
    if isinstance(span[i-1], int):
        chunk = [tok_id for tok_id in span[start:i+1]]
        if isinstance(span[start-1], list):
            if abs(span[start-1][-1]-span[start]) > 1:
                chunk.insert(0, min(chunk)-1)
        else:
            if abs(span[start-1]-span[start]) > 1 or span[start-1] == span[i]:
                chunk.insert(0, min(chunk)-1)
    else:
        if abs(span[start-1][-1]-span[start]) == 1:
            chunk = [span[i]]
        else:
            chunk = [span[i]-1, span[i]]
    return chunk

def chunk_indexes(span):    
    chunks = list()
    indexes = list()
    if len(span) == 1 and isinstance(span[0], int):
        span.insert(0, span[0]-1)
        return [span]
    elif len(span) == 1 and isinstance(span[0], list):
        span = span[0]
        span.insert(0, span[0]-1)
        return [span]
    else:
        for i, tok_id in enumerate(span): 
            if i > 0:
                if isinstance(tok_id, list) and i < len(span)-1:
                    chunks.append(tok_id)
                    indexes.append(i+1)
                elif isinstance(tok_id, list) and i == len(span)-1:
                    chunks.append(tok_id)
                else:
                    try:
                        if isinstance(span[i-1], int) and isinstance(span[i+1], int):
                            if (tok_id-span[i-1] == 1) and (abs(tok_id-span[i+1]) > 1):
                                chunk = get_chunk(indexes, span, i)
                                chunks.append(chunk)
                                indexes.append(i+1)
                            elif (abs(tok_id-span[i-1]) > 1) and (abs(tok_id-span[i+1]) > 1):
                                chunks.append([tok_id-1, tok_id])
                                indexes.append(i+1)
                        elif isinstance(span[i-1], int) and isinstance(span[i+1], list):
                            if (tok_id-span[i-1] == 1):
                                chunk = get_chunk(indexes, span, i)
                                chunks.append(chunk)
                                indexes.append(i+1)
                            else:
                                chunks.append([tok_id-1, tok_id])
                                indexes.append(i+1)
                        elif isinstance(span[i-1], list) and isinstance(span[i+1], int):
                            if (abs(tok_id-span[i+1]) > 1):
                                chunk = get_chunk(indexes, span, i)
                                chunks.append(chunk)
                                indexes.append(i+1)
                    except IndexError: # we are at the end of the list
                        if isinstance(span[i-1], int):
                            if (tok_id-span[i-1] == 1):
                                chunk = get_chunk(indexes, span, i)
                                chunks.append(chunk)
                            else:
                                chunks.append([tok_id-1, tok_id])
                        else:
                            chunk = get_chunk(indexes, span, i)
                            chunks.append(chunk)
            else:
                if isinstance(tok_id, list):
                    tok_id.insert(0, tok_id[0]-1)
                    indexes.append(i+1)
                else:
                    if isinstance(span[i+1], list) or abs(tok_id-span[i+1]) > 1:
                        chunks.append([tok_id-1, tok_id])
                        indexes.append(i+1)
        return chunks

def check_type(chunk_id):
    for idx in chunk_id:
        if idx == None:
            return False
    return True

def merge_token_ids_embed(bert_toks, bert_ids, arbitrary_id, merge, bert_embeddings=list()):
    month_names = list(map(lambda x: calendar.month_name[x].lower(), range(1,13)))
    bert_toks = bert_toks[1:-1]
    bert_ids = bert_ids[1:-1]

    if len(bert_embeddings) > 0:
        bert_embeddings = bert_embeddings[1:-1]
        retokenizer = Retokenizer(merge, embeddings=True)
    else:
        retokenizer = Retokenizer(merge, embeddings=False)
        
    ids_to_rejoin = list()
    for i, bert_tok in enumerate(bert_toks):  
        if i < len(bert_toks)-1:
            if re.search(r'#+\w+', bert_tok) and not re.search(r'#+\w+', bert_toks[i-1]) and re.search(r'#+\w+', bert_toks[i-2]) and not re.search(r'#+\w+', bert_toks[i+1]):
                ids_to_rejoin.append([i-1, i])
            elif re.search(r'#+\w+', bert_tok):
                ids_to_rejoin.append(i)
            if re.search(r"'", bert_tok) and (re.search(r'what|who|why|where|how', bert_toks[i-1]) or re.search(r'^s$', bert_toks[i+1])):
                ids_to_rejoin.append(i+1)
            elif re.search(r"\.", bert_tok) and (re.search(r"[a-zA-Z]+", bert_toks[i-1]) and re.search(r"\w+", bert_toks[i+1])):
                ids_to_rejoin.append(i)
            elif re.search(r"'", bert_tok) and not (re.search(r'what|who|why|where|how', bert_toks[i-1])  or re.search(r'^(s|\?)$', bert_toks[i+1])):
                ids_to_rejoin.append(i)
                ids_to_rejoin.append(i+1)
            elif re.search(r"\.|-|/", bert_tok) and not re.search(r'^\?$', bert_toks[i+1]):
                ids_to_rejoin.append(i)
                ids_to_rejoin.append(i+1)   
            try:
                if re.search(r",|:", bert_tok) and (re.search(r"[0-9]+", bert_toks[i-1]) and re.search(r"[0-9]+", bert_toks[i+1]) and re.search(r",", bert_toks[i+2]) and re.search(r"[0-9]+", bert_toks[i+3])):
                    ids_to_rejoin.append(i)
                    ids_to_rejoin.append(i+1)
                    ids_to_rejoin.append(i+2)
                    ids_to_rejoin.append(i+3)                                                                                       
                elif re.search(r",|:", bert_tok) and re.search(r"[0-9]+", bert_toks[i-1]) and re.search(r"[0-9]+", bert_toks[i+1]) and not ((re.search(r",", bert_toks[i+2]) and re.search(r"[0-9]+", bert_toks[i+3])) or (re.search(r",", bert_toks[i-2]) and re.search(r"[0-9]+", bert_toks[i-3])) or bert_toks[i-2] in month_names or (re.search(r'-', bert_toks[i-2]) and re.search(r'-', bert_toks[i+2]))):
                    ids_to_rejoin.append(i)
                    ids_to_rejoin.append(i+1)
            except IndexError:
                if re.search(r",|:", bert_tok) and re.search(r"[0-9]+", bert_toks[i-1]) and re.search(r"[0-9]+", bert_toks[i+1]) and not (bert_toks[i-2] in month_names or re.search(r'-', bert_toks[i-2])):
                    ids_to_rejoin.append(i)
                    ids_to_rejoin.append(i+1)
                elif re.search(r"\+", bert_tok) and re.search(r"[0-9]+", bert_toks[i+1]):
                    ids_to_rejoin.append(i+1)
        else:
            if re.search(r'#+\w+', bert_tok) and not re.search(r'#+\w+', bert_toks[i-1]) and re.search(r'#+\w+', bert_toks[i-2]):
                ids_to_rejoin.append([i-1, i])
            elif re.search(r'#+\w+', bert_tok):
                ids_to_rejoin.append(i)
            if re.search(r"'", bert_tok) and (re.search(r'what|who|why|where|how', bert_toks[i-1])):
                if i < len(bert_toks)-1:
                    ids_to_rejoin.append(i+1)
            elif re.search(r"-|/|'", bert_tok) and not (re.search(r'what|who|why|where|how', bert_toks[i-1])):
                ids_to_rejoin.append(i)
                if i < len(bert_toks)-1:
                    ids_to_rejoin.append(i+1)
    
    if len(ids_to_rejoin) > 0:
        chunk_ids = chunk_indexes(ids_to_rejoin)
        if len(bert_embeddings) > 0:
            new_ids, new_toks, new_embeddings, new_id = retokenizer.retokenize(bert_toks, 
                                                                               bert_ids, 
                                                                               bert_embeddings,
                                                                               chunk_ids,
                                                                               arbitrary_id)
            return new_ids, new_toks, new_embeddings, new_id
        else:
            
            new_ids, new_toks, new_id = retokenizer.retokenize(bert_toks, 
                                                               bert_ids, 
                                                               bert_embeddings,
                                                               chunk_ids,
                                                               arbitrary_id)
            return new_ids, new_toks, new_id
    else:
        if len(bert_embeddings) > 0:
            return bert_ids, bert_toks, bert_embeddings, arbitrary_id
        else:
            return bert_ids, bert_toks, arbitrary_id

## TODO: investigate how you can make this function work properly 
## FOR NOW: just use token representations of last hidden layer (implemented in cell below)
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

def bert_token_ids(tok_questions, tok_ids, sql_data, arbitrary_id = 99999):
    rejoined_toks = list()
    rejoined_ids = list()
    for i, (question, tok_q, tok_id) in enumerate(zip(sql_data, tok_questions, tok_ids)):
        tok_id = list(tok_id[0].numpy())
        new_ids, new_toks, new_id = merge_token_ids_embed(tok_q, tok_id, arbitrary_id, merge=None)
        arbitrary_id = new_id  
        rejoined_toks.append(new_toks)
        rejoined_ids.append(new_ids)
    return rejoined_toks, rejoined_ids

def bert_embeddings(tok_questions, tok_ids, segment_ids, sql_data, merge, arbitrary_id = 99999,
                    matrix = False, tensor = False):
    """
        Args: torch tensors of token ids and segment ids.
        Computation: load pre-trained BERT model (weights),
                     put the model in "evaluation" mode, meaning feed-forward operation.
                     "torch.no_grad()" deactivates the gradient calculations, 
                     saves memory, and speeds up computation (we don't need gradients or backprop).
        Return: dictionary that maps token ids (keys) 
                to their corresponding BERT context embeddings (values).
    """
    #TODO: check what's necessary to ouput all hidden states
    # model = BertModel.from_pretrained('bert-base-uncased',
    #                              output_hidden_states=True,
    #                              output_attentions=True)
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model.eval()
    id2embed = dict()
    id2tok = dict()
    #embeddings = list()
    rejoined_toks = list()
    rejoined_ids = list()
    with torch.no_grad():
        for i, (question, tok_q, tok_id, segment_id) in enumerate(zip(sql_data, tok_questions, tok_ids, segment_ids)):
            tok_embeddings = model(tok_id, segment_id)[0][0]
            
            #TODO: make "get_summed_embeddings" function work
            #token_embeddings = get_summed_embeddings(model, tok_id, segment_id)
            
            #NOTE: only use lines below, if you'd like to create an embedding matrix (or tensor)
            #if matrix:
            #    if tensor:
            #        embeddings.append(token_embeddings)
            #    else:
            #        for bert_embedding in token_embeddings:
            #            embeddings.append(bert_embedding.numpy())
            
            tok_id = list(tok_id[0].numpy())
            new_ids, new_toks, new_embeddings, new_id = merge_token_ids_embed(tok_q, 
                                                                              tok_id, 
                                                                              arbitrary_id,
                                                                              merge = merge,
                                                                              bert_embeddings = tok_embeddings)
            arbitrary_id = new_id   
            try:
                assert len(new_ids) == len(new_toks) == len(question['question_tok'])
                for tok_id, bert_tok, bert_embedding in zip(new_ids, new_toks, new_embeddings):
                    #tok_i = tok_i.item()
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
    tok_questions, tok_ids, _, _ = bert_preprocessing(extract_questions(sql_data))
    tok_questions, tok_ids = bert_token_ids(tok_questions, tok_ids, sql_data)
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
    Dimensionality reduction of BERT embeddings performed through PCA.
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

def bert_pipeline(sql_data_train, sql_data_val, merge='avg'):
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
        if full:
            embeddings = 'id2embedMaxFull.json'
        else:
            embeddings = 'id2embedMax100.json'
        ids = 'id2tokMax.json'
    elif merge == 'avg':
        if full:
            embeddings = 'id2embedMeanFull.json'
        else:
            embeddings = 'id2embedMean100.json'
        ids = 'id2tokMean.json'
    elif merge == 'sum':
        if full:
            embeddings = 'id2embedSumFull.json'
        else:
            embeddings = 'id2embedSum100.json'
        ids = 'id2tokSum.json' 
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
    plt.show()

def plot_losses(n_epochs, losses):
    plt.plot(n_epochs, losses, color='blue')
    plt.title("TypeSQL's learning curve during training")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()