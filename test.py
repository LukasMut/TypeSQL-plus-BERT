import json
import torch
import datetime
import argparse
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet
#from BERT_context_embeddings import update_sql_data, load_bert_dicts

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
    parser.add_argument('--BERT', type=bool, default=True,
            help='False: use GloVe "no context" embeddings, True: use BERT context embeddings')
    parser.add_argument('--ensemble', type=bool, default=False,
            help='False: load single model, True: load ensemble')
    args = parser.parse_args()
    
    # N_word = 600
    N_word=100
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
    
    # FOR BERT implementation: SQL data must be updated with tokens pre-processed by BERT's WordPiece model
    # and corresponding BERT ids
    #sql_data = update_sql_data(sql_data)
    #val_sql_data = update_sql_data(val_sql_data)
    #test_sql_data = update_sql_data(test_sql_data)
    #print("SQL data has been updated and now consists of bert-preprocessed tokens and their corresponding IDs") 
    
    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    if args.db_content == 0:
        word_emb = load_word_and_type_emb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt',\
                                            val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        word_emb = load_concat_wemb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt')
    
    if args.BERT:
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
            id2tok, word_emb_bert = load_bert_dicts("./id2tokMax.json", "./id2embedMax.json")
        elif args.merged == 'avg':
            id2tok, word_emb_bert = load_bert_dicts("./id2tokMean.json", "./id2embedMean.json")
        elif args.merged == 'sum':
            id2tok, word_emb_bert = load_bert_dicts("./id2tokSum.json", "./id2embedSum.json")
        else:
            raise Exception('Only max-pooled, averaged or summed bert embeddings can be loaded into memory')
        print("Bert embeddings have been loaded into memory")
        bert_tuple = (id2tok, word_emb_bert)
    else:
        bert_tuple = None

    #TODO: add ensemble
    if args.ensemble:
        
        agg_m_1, sel_m_1, cond_m_1, agg_e_1, sel_e_1, cond_e_1, agg_m_2, sel_m_2, cond_m_2, agg_e_2, sel_e_2, cond_e_2 = best_model_name(args)
        
        print "Loading from %s"%agg_m
        model[0].agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.selcond_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.op_str_pred.load_state_dict(torch.load(cond_m))

        #only for loading trainable embedding
        print "Loading from %s"%agg_e
        model.agg_type_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_type_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_type_embed_layer.load_state_dict(torch.load(cond_e))

        
    else: 
        model = [SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content, word_emb_bert=bert_tuple, BERT=args.BERT)]
        agg_m_1, sel_m_1, cond_m_1, agg_e_1, sel_e_1, cond_e_1 = best_model_name(args)
   
    print "Loading from %s"%agg_m
    model[0].agg_pred.load_state_dict(torch.load(agg_m))
    print "Loading from %s"%sel_m
    model[0].selcond_pred.load_state_dict(torch.load(sel_m))
    print "Loading from %s"%cond_m
    model[0].op_str_pred.load_state_dict(torch.load(cond_m))

    #only for loading trainable embedding
    print "Loading from %s"%agg_e
    model[0].agg_type_embed_layer.load_state_dict(torch.load(agg_e))
    print "Loading from %s"%sel_e
    model[0].sel_type_embed_layer.load_state_dict(torch.load(sel_e))
    print "Loading from %s"%cond_e
    model[0].cond_type_embed_layer.load_state_dict(torch.load(cond_e))


    print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY, args.db_content)
    print "Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, args.db_content)
    
    print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, args.db_content)
    print "Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, args.db_content)
