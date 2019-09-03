import json
import torch
import datetime
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typesql.utils import *
from typesql.model.sqlnet import SQLNet
from typesql.lib.dbengine import DBEngine
from bert_utils import update_sql_data, remove_nonequal_questions, load_bert_dicts, plot_accs, plot_losses

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
    parser.add_argument('--BERT', type=bool, default=True,
            help='False: use GloVe "no context" embeddings, True: use BERT context embeddings')
    parser.add_argument('--merged', type=str, default='avg',
            help='max: use max-pooled bert embeddings, avg: use averaged bert embeddings, sum: use summed bert embeddings')
    parser.add_argument('--types', type=bool, default=False,
            help='False: use BERT context embeddings only, True: concatenate BERT with Type embeddings')
    parser.add_argument('--ensemble', type=str, default='single',
            help='single: single model, mixed: mixed ensemble (GloVe and BERT), homogeneous: homogeneous ensemble (e.g., (GloVe and GloVe) XOR (BERT and BERT))')
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
        
    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    
    if args.db_content == 0:
        word_emb = load_word_and_type_emb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt',\
                                            val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        word_emb = load_concat_wemb('glove/glove.6B.50d.txt', 'para-nmt-50m/data/paragram_sl999_czeng.txt')
    
    if args.ensemble == 'mixed':
        model_1 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=None, BERT=False, types=args.types)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types)
        
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
              word_emb_bert=bert_tuple, BERT=self.BERT, types=args.types)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=False)
        
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
              word_emb_bert=bert_tuple, BERT=self.BERT, types=args.types)
        model_2 = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=args.train_emb, db_content=args.db_content,
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types)
        
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
                      word_emb_bert=bert_tuple, BERT=args.BERT, types=args.types)
       
        #TODO: Change optimizer to RAdam as soon as there is an implementation available in PyTorch
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
        model = [model]
        print("Single model was initialized...")
    
    else:
        raise Exception('Model must be single, mixed ensemble or homogeneous ensemble')
        
    assert isinstance(model, list), 'models have to be stored in list'
    
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
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, ensemble=args.ensemble)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc)
        
    if TRAIN_AGG:
        if args.ensemble != 'single':
            for net in model:
                torch.save(net.agg_pred.state_dict(), agg_m)
                torch.save(net.agg_type_embed_layer.state_dict(), agg_e)
        else:
            torch.save(model[0].agg_pred.state_dict(), agg_m)
            torch.save(model[0].agg_type_embed_layer.state_dict(), agg_e)
    if TRAIN_SEL:
        if args.ensemble != 'single':
            for net in model:
                torch.save(net.selcond_pred.state_dict(), sel_m)
                torch.save(net.sel_type_embed_layer.state_dict(), sel_e)         
        else:
            torch.save(model[0].selcond_pred.state_dict(), sel_m)
            torch.save(model[0].sel_type_embed_layer.state_dict(), sel_e)
    if TRAIN_COND:
        if args.ensemble != 'single':
            for net in model:
                torch.save(net.op_str_pred.state_dict(), cond_m)
                torch.save(net.cond_type_embed_layer.state_dict(), cond_e)    
        else:
            torch.save(model[0].op_str_pred.state_dict(), cond_m)
            torch.save(model[0].cond_type_embed_layer.state_dict(), cond_e)
            
    losses = list()
    train_accs = list()
    val_accs = list()
    for i in range(100):
        print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        loss = epoch_train(
                model, optimizer, BATCH_SIZE,
                sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, ensemble=args.ensemble)
        losses.append(loss)
        print(' Loss = %s'%loss)
        train_acc = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY, args.db_content, BERT=args.BERT, ensemble=args.ensemble)
        train_accs.append(train_acc[0])
        print(' Train acc_qm: %s\n breakdown result: %s'% train_acc)

        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, args.db_content, False, BERT=args.BERT, ensemble=args.ensemble) #for detailed error analysis, pass True to the end (second last argument before BERT)
        val_accs.append(val_acc[0])
        print(' Dev acc_qm: %s\n breakdown result: %s'%val_acc)
        
        if TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i+1
                torch.save(model[0].agg_pred.state_dict(),
                           args.sd_1 + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model[0].agg_pred.state_dict(), agg_m)
                torch.save(model[0].agg_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.agg_embed%s'%(i+1, args.suffix))
                torch.save(model[0].agg_type_embed_layer.state_dict(), agg_e)

                if args.ensemble != 'single':
                    torch.save(model[1].agg_pred.state_dict(),
                                args.sd_2 + '/epoch%d.agg_model%s'%(i+1, args.suffix))
                    torch.save(model[1].agg_type_embed_layer.state_dict(),
                               args.sd_2 + '/epoch%d.agg_embed%s'%(i+1, args.suffix))                
                
        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model[0].selcond_pred.state_dict(),
                           args.sd_1 + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model[0].selcond_pred.state_dict(), sel_m)

                torch.save(model[0].sel_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                torch.save(model[0].sel_type_embed_layer.state_dict(), sel_e)
                
                if args.ensemble != 'single':
                    torch.save(model[1].selcond_pred.state_dict(),
                               args.sd_2 + '/epoch%d.sel_model%s'%(i+1, args.suffix))
                    torch.save(model[0].sel_type_embed_layer.state_dict(),
                                args.sd_2 + '/epoch%d.sel_embed%s'%(i+1, args.suffix))
                    
    
        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model[0].op_str_pred.state_dict(),
                           args.sd_1 + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model[0].op_str_pred.state_dict(), cond_m)

                torch.save(model[0].cond_type_embed_layer.state_dict(),
                           args.sd_1 + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                torch.save(model[0].cond_type_embed_layer.state_dict(), cond_e)
                
                if args.ensemble != 'single':
                    torch.save(model[1].op_str_pred.state_dict(),
                               args.sd_2 + '/epoch%d.cond_model%s'%(i+1, args.suffix))
                    torch.save(model[1].cond_type_embed_layer.state_dict(),
                                args.sd_2 + '/epoch%d.cond_embed%s'%(i+1, args.suffix))
                    

        print(' Best val acc = %s, on epoch %s individually'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))
   

    plt.clf() # clear current figure, but leave window opened
    plot_accs(list(range(1,101)), train_accs, val_accs)
    plot_losses(losses)