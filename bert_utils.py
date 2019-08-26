#!/usr/bin/env python
# coding: utf-8

import calendar
import json
import re
import torch
import numpy as np

from pytorch_transformers import *
from typesql.utils import *

def concatenate_sql_data(sql_data_train, sql_data_val):
    sql_data_train.extend(sql_data_val)
    return sql_data_train

def count_context_toks(tok = 'the'): 
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
        Input: Raw natural language questions represented as strings. 
        Computation: Sentence preprocessing steps necessary for BERT model.
                    Each sentence is required to be preceded by a special [CLS] token
                    and followed by a special [SEP] token.
                    Token IDs arrays have to be converted into tensors 
                    before they can be passed to BERT. 
        Output: tokenized questions, token IDs, segment IDs (i.e., ones),
                tuples of (tokens, ids) either per token-id-pair or as a list per sentence.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    questions = list(map(lambda q: '[CLS]' + ' ' + ' '.join(q) + ' ' +'[SEP]', questions))
    tok_questions = [tokenizer.tokenize(q) for q in questions]
    #tok_questions = [rejoin_toks(q) for q in tok_questions]
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

def remove_elements(token_ids, tokens, embeddings, idx, chunk_ids):
    token_ids.pop(idx)
    tokens.pop(idx)
    # remove element at position "idx" in torch tensor - 
    # equivalent to classic Python "vector.pop(idx)"
    if len(embeddings) > 0:
        embeddings = torch.cat([embeddings[:idx], embeddings[idx:]])
    if len(chunk_ids) > 0:
        chunk_ids = [[el-1 for el in chunk] for chunk in chunk_ids]
    return token_ids, tokens, embeddings, chunk_ids

def check_type(chunk_id):
    for idx in chunk_id:
        if idx == None:
            return False
    return True

## function only important for updating sql data (no embeddings) to compute token rejoining step faster ##
def rejoin_token_ids(bert_toks, bert_ids, bert_embeddings, chunk_ids, arbitrary_id, merge):
    new_toks = list()
    new_ids = list()
    for i, (bert_tok, bert_id) in enumerate(zip(bert_toks, bert_ids)):
        if len(chunk_ids) > 0:
            for j, chunk_id in enumerate(chunk_ids):
                if i in chunk_id:
                    new_ids.append(arbitrary_id)
                    arbitrary_id += 1
                    chunk_ids.pop(j)
                    new_tok = ''.join([bert_toks[idx] for idx in chunk_id])
                    new_tok = re.sub(r'#+','', new_tok)
                    new_toks.append(new_tok)
                    tok_span = len(chunk_id)
                    if tok_span > 2:
                        pop_idx = 0
                        for k in range(tok_span-1):
                            bert_ids, bert_toks, _, chunk_ids = remove_elements(\
                                bert_ids,bert_toks,bert_embeddings,chunk_id[k]-pop_idx,chunk_ids)
                            pop_idx += 1
                    elif tok_span == 2:
                        bert_ids, bert_toks, _, chunk_ids = remove_elements(\
                            bert_ids,bert_toks,bert_embeddings,chunk_id[0],chunk_ids)
                    break
                else:
                    new_ids.append(bert_id)    
                    new_toks.append(bert_tok)
                    break
        else:
            new_ids.append(bert_id)    
            new_toks.append(bert_tok)
    return new_ids, new_toks, arbitrary_id

def rejoin_token_ids_embed(bert_toks, bert_ids, bert_embeddings, chunk_ids, arbitrary_id, merge):
    """
        Input: BERT tokens pre-processed according to WordPiece-Model, corresponding BERT ids,
               BERT context embeddings, indexes in token list to be rejoined, arbitrary id that 
               will be used instead of original BERT id for rejoined token, 
               string that denotes whether BERT context vectors should be summed or averaged after
               rejoining BERT tokens into TypeSQL tokens.
        Output: Rejoined tokens, corresponding ids, BERT embeddings and new arbitrary id to start at
                next iteration. 
    """
    new_toks = list()
    new_ids = list()
    new_embeddings = list()
    for i, (bert_tok, bert_id, bert_embedding) in enumerate(zip(bert_toks, bert_ids, bert_embeddings)):
        if len(chunk_ids) > 0:
            for j, chunk_id in enumerate(chunk_ids):
                if i in chunk_id:
                    new_ids.append(arbitrary_id)
                    arbitrary_id += 1
                    chunk_ids.pop(j)
                    new_tok = ''.join([bert_toks[idx] for idx in chunk_id])
                    new_tok = re.sub(r'#+','', new_tok)
                    new_toks.append(new_tok)
                    if merge == 'sum':
                        new_embedding = np.sum(np.array([bert_embeddings[idx].numpy() for 
                                                         idx in chunk_id]), axis=0)
                    elif merge == 'avg':
                        new_embedding = np.mean(np.array([bert_embeddings[idx].numpy() for 
                                                         idx in chunk_id]), axis=0)
                    else:
                        raise Exception("Bert embeddings should be summed or averaged.")
                    new_embeddings.append(new_embedding)
                    tok_span = len(chunk_id)
                    if tok_span > 2:
                        pop_idx = 0
                        for k in range(tok_span-1):
                            bert_ids, bert_toks, bert_embeddings, chunk_ids = remove_elements(\
                                bert_ids,bert_toks,bert_embeddings,chunk_id[k]-pop_idx,chunk_ids)
                            pop_idx += 1
                    elif tok_span == 2:
                        bert_ids, bert_toks, bert_embeddings, chunk_ids = remove_elements(\
                            bert_ids,bert_toks,bert_embeddings,chunk_id[0],chunk_ids)
                    break
                else:
                    new_ids.append(bert_id)    
                    new_toks.append(bert_tok)
                    new_embeddings.append(bert_embedding.numpy())
                    break
        else:
            new_ids.append(bert_id)    
            new_toks.append(bert_tok)
            new_embeddings.append(bert_embedding.numpy())
    return new_ids, new_toks, new_embeddings, arbitrary_id

def merge_token_ids_embed(bert_toks, bert_ids, arbitrary_id, merge, bert_embeddings = list()):
    month_names = list(map(lambda x: calendar.month_name[x].lower(), range(1,13)))
    bert_toks = bert_toks[1:-1]
    bert_ids = bert_ids[1:-1]
    if len(bert_embeddings) > 0:
        bert_embeddings = bert_embeddings[1:-1]
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
            new_ids, new_toks, new_embeddings, new_id = rejoin_token_ids_embed(bert_toks, 
                                                                               bert_ids, 
                                                                               bert_embeddings,
                                                                               chunk_ids,
                                                                               arbitrary_id,
                                                                               merge)
            return new_ids, new_toks, new_embeddings, new_id
        else:
            new_ids, new_toks, new_id = rejoin_token_ids(bert_toks, 
                                                         bert_ids, 
                                                         bert_embeddings,
                                                         chunk_ids,
                                                         arbitrary_id,
                                                         merge)
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
        Input: BertModel, token id tensors, segment id tensors
        Computation: Convert the hidden state embeddings into single token vectors
                     Holds the list of 12 layer embeddings for each token
                     Will have the shape: [# tokens, # layers, # features]
        Output: Bert context embedding for each token in the question.
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
        new_ids, new_toks, new_id = merge_token_ids_embed(tok_q, tok_id, arbitrary_id, merge = 'sum')
        arbitrary_id = new_id  
        rejoined_toks.append(new_toks)
        rejoined_ids.append(new_ids)
    return rejoined_toks, rejoined_ids

def bert_embeddings(tok_questions, tok_ids, segment_ids, sql_data, dim = 100, arbitrary_id = 99999,
                    matrix = False, tensor = False):
    """
        Input: torch tensors of token ids and segment ids.
        Computation: load pre-trained BERT model (weights),
                     put the model in "evaluation" mode, meaning feed-forward operation.
                     "torch.no_grad()" deactivates the gradient calculations, 
                     saves memory, and speeds up computation (we don't need gradients or backprop).
        Output: dictionary that maps token ids (keys) 
                to their corresponding BERT context embeddings (values).
    """
    ## TO DO: check what's necessary to ouput all hidden states
    # model = BertModel.from_pretrained('bert-base-uncased',
    #                              output_hidden_states=True,
    #                              output_attentions=True)
    
    model = BertModel.from_pretrained('bert-base-uncased')
    #model = RobertaModel.from_pretrained('roberta-base')
    
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
                                                                              merge = 'sum',
                                                                              bert_embeddings = tok_embeddings)
            arbitrary_id = new_id   
            try:
                assert len(new_ids) == len(new_toks) == len(question['question_tok'])
                for tok_id, bert_tok, bert_embedding in zip(new_ids, new_toks, new_embeddings):
                    #tok_i = tok_i.item()
                    if tok_id not in id2embed:
                        id2embed[tok_id] = bert_embedding[:dim]
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
    assert len(sql_data) == n_questions-n_errors # check whether correct number of erroneous questions was dropped
    return tok_questions, tok_ids, sql_data

def update_sql_data(sql_data):
    """
        Input: SQL dataset
        Output: Updated SQL dataset with bert tokens and corresponding bert ids
                BERT tokens were rejoined into TypeSQL's gold standard tokens and
                hence are the same
    """
    tok_questions, tok_ids, _, _ = bert_preprocessing(extract_questions(sql_data))
    tok_questions, tok_ids = bert_token_ids(tok_questions, tok_ids, sql_data)
    idx_to_pop = list()
    n_original_questions = len(sql_data)
    print("Number of questions before computing BERT token representations:", n_original_questions)
    print()
    for i, (question, tok_id, tok_question) in enumerate(zip(sql_data, tok_ids, tok_questions)):
        try:
            assert len(question['question_tok']) == len(tok_id)  == len(tok_question)
        except:
            idx_to_pop.append(i)
        
    tok_questions, tok_ids, sql_data = drop_data(tok_questions, tok_ids, sql_data, idx_to_pop) 
    
    for i, (question, tok_id, tok_question) in enumerate(zip(sql_data, tok_ids, tok_questions)):
        try:
            assert len(sql_data[i]['question_tok']) == len(tok_id)  == len(tok_question)
            sql_data[i]['bert_tokenized_question'] = tok_question
            sql_data[i]['bert_token_ids'] = tok_id #list(tok_id[0].numpy())
        except:
            raise Exception("Removing incorrectly rejoined questions did not work. Check function!")
    
    n_removed_questions = n_original_questions-len(sql_data)
    
    print("Number of questions in pre-processed dataset (after rejoining):", len(sql_data))
    print()
    print("Questions that could not be rejoined into TypeSQL tokens:", n_removed_questions)
    print()
    print("{}% of the original questions were removed".format(round((n_removed_questions / n_original_questions)*100, 2)))
    print()
    print("SQL data has been updated with BERT ids (tokens are the same as TypeSQL's tokens)")
    print()
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
    print()
    return sql_data

def bert_pipeline(sql_data_train, sql_data_val):
    sql_data = concatenate_sql_data(sql_data_train, sql_data_val)
    tok_questions, tok_ids, segment_ids, _ = bert_preprocessing(extract_questions(sql_data))
    _, _, id2embed, id2tok = bert_embeddings(tok_questions, tok_ids, segment_ids, sql_data)
    assert len(id2embed) == len(id2tok)
    return id2tok, id2embed

def save_embeddings_as_json(id2tok, id2embed):
    # save token id : bert embeddings dictionary as .json
    # np.arrays have to be converted into lists to be .json serializable
    id2embed = {int(idx):embedding.tolist() for idx, embedding in id2embed.items()}
    id2tok = {int(idx):tok for idx, tok in id2tok.items()}
    with open('id2embed.json', 'w') as json_file:
        json.dump(id2embed, json_file)
    # save token id : bert token dictionary as .json
    with open('id2tok.json', 'w') as json_file:
        json.dump(id2tok, json_file)
        
def load_bert_dicts(file_tok, file_emb):
    with open(file_tok) as f:
        id2tok = json.loads(f.read())
    with open(file_emb) as f:
        id2embed = json.loads(f.read())
    id2tok = {int(idx):tok for idx, tok in id2tok.items()}
    id2embed = {int(idx):np.array(embedding) for idx, embedding in id2embed.items()}
    return id2tok, id2embed