import json
import torch
import datetime
import argparse
import numpy as np

from typesql.utils import *
from typesql.model.sqlnet import SQLNet
from bert_utils import update_sql_data, remove_nonequal_questions, load_bert_dicts
from pos_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--sd_1', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--sd_2',type=str, default='',
            help='set save directory for second model if ensemble.')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    parser.add_argument('--BERT', action='store_true',
            help='If provided: Use BERT context embeddings, Else: use GloVe embeddings')
    parser.add_argument('--merged', type=str, default='avg',
            help='max: use max-pooled bert embeddings, avg: use averaged bert embeddings, sum: use summed bert embeddings')
    parser.add_argument('--POS', action='store_true',
            help='Add Part-of-Speech embeddings')  
    parser.add_argument('--types', action='store_true',
            help='If provided: concatenate BERT with Type embeddings, Else: use BERT context embeddings only')
    parser.add_argument('--ensemble', type=str, default='single',
            help='single: single model, mixed: mixed ensemble (GloVe and BERT), homogeneous: homogeneous ensemble (e.g., (GloVe and GloVe) XOR (BERT and BERT))')
    
    args = parser.parse_args()
    print(args)
    
    # N_word = 600
    N_word=100
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=False # set to True for running on Cluster
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
    print()
    
    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    
    if args.db_content == 0:
        
        if N_word == 100:
            word_emb = load_word_and_type_emb('./glove/glove.6B.50d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt',val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
        elif N_word == 600:
            word_emb = load_word_and_type_emb('./glove/glove.42B.300d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt',val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        if N_word == 100:
            word_emb = load_concat_wemb('./glove/glove.6B.50d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt')
        elif N_word == 600:
            word_emb = load_concat_wemb('./glove/glove.42B.300d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt')
            
    
    if args.BERT:
        print()
        sql_data_updated = update_sql_data(sql_data)
        sql_data = remove_nonequal_questions(sql_data_updated)
        val_sql_data_updated = update_sql_data(val_sql_data)
        val_sql_data = remove_nonequal_questions(val_sql_data_updated)
        test_sql_data_updated = update_sql_data(test_sql_data)
        test_sql_data = remove_nonequal_questions(test_sql_data_updated)
        print("SQL data has been updated and now consists of bert-preprocessed tokens and corresponding IDs")
        print()
        print("Loading bert embeddings...")
        if args.merged == 'max':
            if N_word == 100:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokMax.json", "./bert/id2embedMax100.json")
            elif N_word == 600:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokMax.json", "./bert/id2embedMax600.json")
        elif args.merged == 'avg':
            if N_word == 100:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokMean.json", "./bert/id2embedMean100.json")
            elif N_word == 600:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokMean.json", "./bert/id2embedMean600.json")
        elif args.merged == 'sum':
            if N_word == 100:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokSum.json", "./bert/id2embedSum100.json")
            elif N_word == 600:
                id2tok, word_emb_bert = load_bert_dicts("./bert/id2tokSum.json", "./bert/id2embedSum600.json")
        else:
            raise Exception('Only max-pooled, averaged or summed bert embeddings can be loaded into memory')
        print("Bert embeddings have been loaded into memory")
        bert_tuple = (id2tok, word_emb_bert)
    
    else:
        bert_tuple = None
        
    if args.POS:
        sql_data = update_sql_data_pos(sql_data)
        val_sql_data = update_sql_data_pos(val_sql_data)
        test_sql_data = update_sql_data_pos(test_sql_data)
        print("SQL data has been updated with POS tags for each token")
        print()
        
    if args.ensemble != 'single':
        
        agg_m1, sel_m1, cond_m1, agg_e1, sel_e1, cond_e1, agg_m2, sel_m2, cond_m2, agg_e2, sel_e2, cond_e2 = best_model_name(args)
        
        if args.ensemble == 'mixed':
            
            model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=None, BERT=False, types=args.types, POS=args.POS)
            model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=args.POS)
            
        elif args.ensemble == 'homogeneous' and args.BERT:
            
            model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=self.BERT, types=args.types, POS=args.POS)
            model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=False)

        elif args.ensemble == 'homogeneous' and not args.BERT:
            
            model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=self.BERT, types=args.types, POS=args.POS)
            model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=False)
       
        model = [model_1, model_2]

        print("Loading from %s"%agg_m1)
        model[0].agg_pred.load_state_dict(torch.load(agg_m1))
        print("Loading from %s"%sel_m1)
        model[0].selcond_pred.load_state_dict(torch.load(sel_m1))
        print("Loading from %s"%cond_m1)
        model[0].op_str_pred.load_state_dict(torch.load(cond_m1))
        
        print("Loading from %s"%agg_m2)
        model[1].agg_pred.load_state_dict(torch.load(agg_m2))
        print("Loading from %s"%sel_m2)
        model[1].selcond_pred.load_state_dict(torch.load(sel_m2))
        print("Loading from %s"%cond_m2)
        model[1].op_str_pred.load_state_dict(torch.load(cond_m2))
        
        #only for loading trainable embedding
        print("Loading from %s"%agg_e1)
        model[0].agg_type_embed_layer.load_state_dict(torch.load(agg_e1))
        print("Loading from %s"%sel_e1)
        model[0].sel_type_embed_layer.load_state_dict(torch.load(sel_e1))
        print("Loading from %s"%cond_e1)
        model[0].cond_type_embed_layer.load_state_dict(torch.load(cond_e1))

        #only for loading trainable embedding
        print("Loading from %s"%agg_e2)
        model[1].agg_type_embed_layer.load_state_dict(torch.load(agg_e1))
        print("Loading from %s"%sel_e2)
        model[1].sel_type_embed_layer.load_state_dict(torch.load(sel_e1))
        print("Loading from %s"%cond_e2)
        model[1].cond_type_embed_layer.load_state_dict(torch.load(cond_e1))

        
    elif args.ensemble == 'single':
        
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
        model = [SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=args.POS)]
   
        print("Loading from %s"%agg_m)
        model[0].agg_pred.load_state_dict(torch.load(agg_m))
        print("Loading from %s"%sel_m)
        model[0].selcond_pred.load_state_dict(torch.load(sel_m))
        print("Loading from %s"%cond_m)
        model[0].op_str_pred.load_state_dict(torch.load(cond_m))

        #only for loading trainable embedding
        print("Loading from %s"%agg_e)
        model[0].agg_type_embed_layer.load_state_dict(torch.load(agg_e))
        print("Loading from %s"%sel_e)
        model[0].sel_type_embed_layer.load_state_dict(torch.load(sel_e))
        print("Loading from %s"%cond_e)
        model[0].cond_type_embed_layer.load_state_dict(torch.load(cond_e))

    accs = dict()
    
    dev_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
    dev_exec_acc= epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
    
    accs['dev']=dev_acc[0]
    accs['dev_exec']=dev_exec_acc
    
    print("Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"% dev_acc)
    print("Dev execution acc: %s"% dev_exec_acc)
    
    test_acc = epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
    test_exec_acc = epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
    
    accs['test']=test_acc[0]
    accs['test_exec']=test_exec_acc
    
    print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"% test_acc)
    print("Test execution acc: %s"% test_exec_acc )
    
    RESULTS = 'results'
    POS = '_pos' if args.POS else ''
    BERT = '_bert' if args.BERT else ''
    ENSEMBLE = '_single' if args.ensemble=='single' else '_ensemble'
    TYPES = '_types' if args.types else ''
    DB = '_kg' if args.db_content==0 else '_db'
    
    with open(RESULTS+POS+BERT+ENSEMBLE+TYPES+DB+'.json', 'w') as f:
        json.dump(accs, f)
