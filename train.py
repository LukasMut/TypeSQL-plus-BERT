import argparse
import datetime
import json
import logging
import matplotlib
import torch

import numpy as np
import matplotlib.pyplot as plt

from typesql.utils import *
from typesql.model.sqlnet import SQLNet
from typesql.lib.dbengine import DBEngine
from bert_utils import update_sql_data, remove_nonequal_questions, load_bert_dicts, plot_accs, plot_losses
from pos_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--sd_1', type=str, default='',
            help='set model save directory for single model.')
    parser.add_argument('--sd_2', type=str, default='',
            help='set save directory for second model, if ensemble computation.')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding.')
    parser.add_argument('--dim', type=int, default=100,
            help='Dimensionality of word vectors used to represent natural language questions.')
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

    N_word=args.dim
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=False #set to True, if you have access to a GPU
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True #set to True, if you have access to a GPU
        BATCH_SIZE=64
    TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-3

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
    print()
    
    ## FOR BERT implementation: SQL data must be updated with (rejoined)
    ## tokens tokenized by BERT's WordPiece model and corresponding BERT ids
    if args.BERT:    
        sql_data = update_sql_data(sql_data)
        #sql_data = remove_nonequal_questions(sql_data)
        
        val_sql_data = update_sql_data(val_sql_data)
        #val_sql_data = remove_nonequal_questions(val_sql_data)
        
        test_sql_data = update_sql_data(test_sql_data)
        #test_sql_data = remove_nonequal_questions(test_sql_data_updated)
        
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
        print()
        bert_tuple = (id2tok, word_emb_bert)
    else:
        bert_tuple = None
        
    if args.POS:
        sql_data = update_sql_data_pos(sql_data)
        val_sql_data = update_sql_data_pos(val_sql_data)
        test_sql_data = update_sql_data_pos(test_sql_data)
        print("SQL data has been updated with POS tags for each token")
        print()
    
    if args.db_content == 0:
        
        if N_word == 100:
            word_emb = load_word_and_type_emb('./glove/glove.6B.50d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt',val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False, N_word=int(N_word/2))
            print("Using GloVe 50d")
            print()
        elif N_word == 600:
            word_emb = load_word_and_type_emb('./glove/glove.42B.300d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt',val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False, N_word=int(N_word/2))
            print("Using GloVe 300d")
            print()
    else:
        if N_word == 100:
            word_emb = load_concat_wemb('./glove/glove.6B.50d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt', dim=int(N_word/2))
            print("Using GloVe 50d")
            print()
        elif N_word == 600:
            word_emb = load_concat_wemb('./glove/glove.42B.300d.txt', './para-nmt-50m/data/paragram_sl999_czeng.txt', dim=int(N_word/2))
            print("Using GloVe 300d")
            print()
    
    if args.ensemble == 'mixed':
        model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=None, BERT=False, types=args.types, POS=args.POS)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=args.POS)
        
        #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate, weight_decay = 0)
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate, weight_decay = 0)
        
        model = [model_1, model_2]
        optimizer = [optimizer_1, optimizer_2]
        print()
        print("Mixed ensemble was initialized...")
        print()
        
    elif args.ensemble == 'homogeneous' and args.BERT:
        model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
              word_emb_bert=bert_tuple, BERT=args.BERT, types=True, POS=True)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=True, POS=False)
        
        #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate, weight_decay = 0)
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate, weight_decay = 0)
        
        model = [model_1, model_2]
        optimizer = [optimizer_1, optimizer_2]
        print()
        print("Homogeneous BERT ensemble was initialized...")
        print()
        
    elif args.ensemble == 'homogeneous' and not args.BERT:
        model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
              word_emb_bert=bert_tuple, BERT=args.BERT, types=True, POS=True)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=True, POS=False)
        
        #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate, weight_decay = 0)
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate, weight_decay = 0)
        
        model = [model_1, model_2]
        optimizer = [optimizer_1, optimizer_2]
        print()
        print("Homogeneous GloVe ensemble was initialized...")
        print()
        
    elif args.ensemble == 'single':
        model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types, POS=args.POS)
       
        #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
        model = [model]
        print()
        print("Single model was initialized...")
        print()
    
    else:
        raise Exception('Model must be single, mixed ensemble or homogeneous ensemble')
        
    assert isinstance(model, list), 'models have to be stored in list'
    
    if args.ensemble == 'single':
        agg_m1, sel_m1, cond_m1, agg_e1, sel_e1, cond_e1 = best_model_name(args)
    else:
        agg_m1, sel_m1, cond_m1, agg_e1, sel_e1, cond_e1, agg_m2, sel_m2, cond_m2, agg_e2, sel_e2, cond_e2 = best_model_name(args)
        

    if args.train_emb: # Load pretrained model.
        agg_lm, sel_lm, cond_lm = best_model_name(args, for_load=True)
        print("Loading from %s"%agg_lm)
        model.agg_pred.load_state_dict(torch.load(agg_lm))
        print("Loading from %s"%sel_lm)
        model.selcond_pred.load_state_dict(torch.load(sel_lm))
        print("Loading from %s"%cond_lm)
        model.cond_pred.load_state_dict(torch.load(cond_lm))

    #initial accuracy
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc)
        
    if TRAIN_AGG:
        if args.ensemble == 'single':
            torch.save(model[0].agg_pred.state_dict(), agg_m1)
            torch.save(model[0].agg_type_embed_layer.state_dict(), agg_e1)
        else:
            torch.save(model[0].agg_pred.state_dict(), agg_m1)
            torch.save(model[0].agg_type_embed_layer.state_dict(), agg_e1)
            torch.save(model[1].agg_pred.state_dict(), agg_m2)
            torch.save(model[1].agg_type_embed_layer.state_dict(), agg_e2)        
            
    if TRAIN_SEL:
        if args.ensemble == 'single':               
            torch.save(model[0].selcond_pred.state_dict(), sel_m1)
            torch.save(model[0].sel_type_embed_layer.state_dict(), sel_e1)
        else:
            torch.save(model[0].selcond_pred.state_dict(), sel_m1)
            torch.save(model[0].sel_type_embed_layer.state_dict(), sel_e1)
            torch.save(model[1].selcond_pred.state_dict(), sel_m2)
            torch.save(model[1].sel_type_embed_layer.state_dict(), sel_e2)
            
    if TRAIN_COND:
        if args.ensemble == 'single':
            torch.save(model[0].op_str_pred.state_dict(), cond_m1)
            torch.save(model[0].cond_type_embed_layer.state_dict(), cond_e1)
        else:
            torch.save(model[0].op_str_pred.state_dict(), cond_m1)
            torch.save(model[0].cond_type_embed_layer.state_dict(), cond_e1)   
            torch.save(model[1].op_str_pred.state_dict(), cond_m2)
            torch.save(model[1].cond_type_embed_layer.state_dict(), cond_e2)
            
    losses = list()
    train_accs = list()
    val_accs = list()
    for i in range(100):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        loss = epoch_train(
                model, optimizer, BATCH_SIZE,
                sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
        losses.append(loss)
        print(' Loss = %s'%loss)
        train_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble)
        train_accs.append(train_acc[0])
        print(' Train acc_qm: %s\n breakdown result: %s'% train_acc)

        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, False, BERT=args.BERT, POS=args.POS, ensemble=args.ensemble) 
        
        #for detailed error analysis, pass True to the end (argument before BERT)
        
        val_accs.append(val_acc[0])
        print(' Dev acc_qm: %s\n breakdown result: %s'%val_acc)
        
        if TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i+1
                torch.save(model[0].agg_pred.state_dict(),
                           args.sd_1 + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model[0].agg_pred.state_dict(), agg_m1)
                torch.save(model[0].agg_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.agg_embed%s'%(i+1, args.suffix))
                torch.save(model[0].agg_type_embed_layer.state_dict(), agg_e1)

                if args.ensemble != 'single':
                    torch.save(model[1].agg_pred.state_dict(),
                                args.sd_2 + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                    torch.save(model[1].agg_pred.state_dict(), agg_m2)
                    torch.save(model[1].agg_type_embed_layer.state_dict(),
                               args.sd_2 + '/epoch%d.agg_embed%s'%(i+1, args.suffix)) 
                    torch.save(model[1].agg_type_embed_layer.state_dict(), agg_e2)
                
        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model[0].selcond_pred.state_dict(),
                           args.sd_1 + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model[0].selcond_pred.state_dict(), sel_m1)

                torch.save(model[0].sel_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                torch.save(model[0].sel_type_embed_layer.state_dict(), sel_e1)
                
                if args.ensemble != 'single':
                    torch.save(model[1].selcond_pred.state_dict(),
                               args.sd_2 + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                    torch.save(model[1].selcond_pred.state_dict(), sel_m2)
                    torch.save(model[1].sel_type_embed_layer.state_dict(),
                                args.sd_2 + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                    torch.save(model[1].sel_type_embed_layer.state_dict(), sel_e2)

        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model[0].op_str_pred.state_dict(),
                           args.sd_1 + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model[0].op_str_pred.state_dict(), cond_m1)
                torch.save(model[0].cond_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                torch.save(model[0].cond_type_embed_layer.state_dict(), cond_e1)
                
                if args.ensemble != 'single':
                    torch.save(model[1].op_str_pred.state_dict(),
                               args.sd_2 + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                    torch.save(model[1].op_str_pred.state_dict(), cond_m2)
                    torch.save(model[1].cond_type_embed_layer.state_dict(),
                                args.sd_2 + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                    torch.save(model[1].cond_type_embed_layer.state_dict(), cond_e2)

                    

        print(' Best val acc = %s, on epoch %s individually'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))
    
    
    #SAVE RESULTS
    
    accs = dict()
    accs['train']=dict()
    accs['dev']=dict()
    
    accs['train']['max']=max(train_accs)
    accs['train']['agg']=train_acc[1][0]
    accs['train']['sel']=train_acc[1][1]
    accs['train']['where']=train_acc[1][2]
    accs['dev']['max']=max(val_accs)
    accs['dev']['agg']=best_agg_acc
    accs['dev']['sel']=best_sel_acc
    accs['dev']['where']=best_cond_acc
    
    DIMS = '100d' if N_word==100 else '600d'
    POS = '_pos' if args.POS else ''
    BERT = '_bert' if args.BERT else ''
    TYPES = '_types' if args.types else ''
    if args.ensemble=='single':
        ENSEMBLE = '_single'
    elif args.ensemble=='mixed':
        ENSEMBLE = '_mixed'
    elif args.ensemble=='homogeneous':
        ENSEMBLE = '_homogeneous'
    DB = '_kg' if args.db_content==0 else '_db'
    
    DATETIME = datetime.datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
    
    if args.BERT:
        if args.merged=='max':
            MERGED='_max-pool'
        elif args.merged=='avg':
            MERGED='_avg'
        LOG_FILENAME = DIMS+BERT+MERGED+POS+TYPES+ENSEMBLE+DB+DATETIME+'.log'
        with open('./results/'+DIMS+BERT+MERGED+POS+TYPES+ENSEMBLE+DB+'.json', 'w') as f:
            json.dump(accs, f)
    else:
        LOG_FILENAME = DIMS+POS+TYPES+ENSEMBLE+DB+DATETIME+'.log'
        with open('./results/'+DIMS+POS+TYPES+ENSEMBLE+DB+'.json', 'w') as f:
            json.dump(accs, f)
    
    accs = json.dumps(accs)
 
    logging.basicConfig(filename='./log_files/'+LOG_FILENAME, level=logging.INFO)
    logging.info(accs)

    plt.clf()
    plot_accs(list(range(1,101)), train_accs, val_accs)
    if args.BERT:
        plt.savefig('./plots/accs/'+DIMS+BERT+MERGED+POS+TYPES+ENSEMBLE+DB+'.png')
    else:
        plt.savefig('./plots/accs/'+DIMS+POS+TYPES+ENSEMBLE+DB+'.png')
    plt.close()
    plot_losses(list(range(1,101)), losses)
    if args.BERT:
        plt.savefig('./plots/losses/'+DIMS+BERT+MERGED+POS+TYPES+ENSEMBLE+DB+'.png')
    else:
        plt.savefig('./plots/losses/'+DIMS+POS+TYPES+ENSEMBLE+DB+'.png')
    plt.close('all')