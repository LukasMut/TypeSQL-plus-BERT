import json
import torch
import datetime
import argparse
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet
from typesql.lib.dbengine import DBEngine
from bert_utils import update_sql_data, remove_nonequal_questions, load_bert_dicts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')
    parser.add_argument('--BERT', type=bool, default=True,
            help='False: use GloVe "no context" embeddings, True: use BERT context embeddings')
    parser.add_argument('--types', type=bool, default=False,
            help='False: only use BERT context embeddings, True: concatenate BERT with Type embeddings')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')

    args = parser.parse_args()

    #N_word=600
    N_word = 100
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=False #True #set to True, if you have access to a GPU
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=False #True #set to True, if you have access to a GPU
        BATCH_SIZE=64
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-3

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
    print()
    
    ## FOR BERT implementation: SQL data must be updated with (rejoined)
    ## tokens pre-processed by BERT's WordPiece model and corresponding BERT ids
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
        id2tok, word_emb_bert = load_bert_dicts("./id2tok.json", "./id2embed.json")
        print("Bert embeddings have been loaded into memory")
        bert_tuple = (id2tok, word_emb_bert)
    else:
        bert_tuple = None
        
    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    
    if args.db_content == 0:
        word_emb = load_word_and_type_emb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt',\
                                            val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        word_emb = load_concat_wemb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt')
    

    # Lines below are for BERT implementations
    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                  word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types)

    #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)

    if args.train_emb: # Load pretrained model.
        agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
        print("Loading from %s"%agg_lm)
        model.agg_pred.load_state_dict(torch.load(agg_lm))
        print("Loading from %s"%sel_lm)
        model.selcond_pred.load_state_dict(torch.load(sel_lm))
        print("Loading from %s"%cond_lm)
        model.cond_pred.load_state_dict(torch.load(cond_lm))


    #initial accuracy
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc)
    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
        torch.save(model.agg_type_embed_layer.state_dict(), agg_e)
    if TRAIN_SEL:
        torch.save(model.selcond_pred.state_dict(), sel_m)
        torch.save(model.sel_type_embed_layer.state_dict(), sel_e)
    if TRAIN_COND:
        torch.save(model.op_str_pred.state_dict(), cond_m)
        torch.save(model.cond_type_embed_layer.state_dict(), cond_e)

    for i in range(100):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        print(' Loss = %s'%epoch_train(
                model, optimizer, BATCH_SIZE,
                sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT))
        print(' Train acc_qm: %s\n breakdown result: %s'%epoch_acc(
                model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT))

        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, False, BERT=args.BERT) #for detailed error analysis, pass True to the end (second last argument before BERT)
        print(' Dev acc_qm: %s\n breakdown result: %s'%val_acc)
        if TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i+1
                torch.save(model.agg_pred.state_dict(),
                    args.sd + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model.agg_pred.state_dict(), agg_m)

            torch.save(model.agg_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.agg_embed%s'%(i+1, args.suffix))
            torch.save(model.agg_type_embed_layer.state_dict(), agg_e)

        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model.selcond_pred.state_dict(),
                    args.sd + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model.selcond_pred.state_dict(), sel_m)

                torch.save(model.sel_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                torch.save(model.sel_type_embed_layer.state_dict(), sel_e)

        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model.op_str_pred.state_dict(),
                    args.sd + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model.op_str_pred.state_dict(), cond_m)

                torch.save(model.cond_type_embed_layer.state_dict(),
                                args.sd + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                torch.save(model.cond_type_embed_layer.state_dict(), cond_e)

        print(' Best val acc = %s, on epoch %s individually'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))
