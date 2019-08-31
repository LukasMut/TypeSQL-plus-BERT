import numpy as np
import re
import torch

class Retokenizer:
    
    def __init__(self, merge, embeddings: bool):
        """
            Input: init method takes a string for specifing the merging computation of embeddings 
            and a boolean value that indicates whether embeddings are passed or not.
        """
        self.merge = merge
        # set embeddings to False to compute token rejoining step faster (for updating sql data)
        self.embeddings = embeddings
    
    @staticmethod
    def remove_elements(token_ids, tokens, embeddings, idx, chunk_ids):
        token_ids.pop(idx)
        tokens.pop(idx)
        if isinstance(embeddings, list):
            embeddings.pop(idx)
        else:
            embeddings = torch.cat([embeddings[:idx], embeddings[idx:]])
        if len(chunk_ids) > 0:
            chunk_ids = [[el-1 for el in chunk] for chunk in chunk_ids]
        return token_ids, tokens, embeddings, chunk_ids
    
    @staticmethod
    def max_pooling(mat):
        #max_vec = np.zeros(mat.shape[1])
        #transpose matrix to loop over column vectors
        #for j, col in enumerate(mat.T):
        #    max_vec[j] += max(col)
        return torch.max(mat,dim=0)[0].numpy()
    
    def retokenize(self, bert_toks, bert_ids, bert_embeddings, chunk_ids, arbitrary_id):
        """
            Input: BERT tokens pre-processed according to WordPiece-Model, corresponding BERT ids,
                   BERT context embeddings, indexes in token list to be rejoined, arbitrary id that 
                   will be used instead of original BERT id for rejoined token, 
                   string that denotes whether BERT context vectors should be summed or averaged after
                   rejoining BERT tokens into TypeSQL tokens.
            Output: Rejoined tokens, corresponding ids, BERT embeddings and new arbitrary id to start at
                    next iteration. 
        """
        assert isinstance(arbitrary_id, int), 'token ids must be integers'
        new_toks = list()
        new_ids = list()
        if self.embeddings:
            new_embeddings = list()
        else:
            # create placeholder values for embeddings to compute loop over all arrays simultaneously
            bert_embeddings = [0 for _ in range(len(bert_toks))]
            assert len(bert_toks) == len(bert_embeddings), 'arrays must have same number of elements'
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
                        if self.embeddings:
                            if self.merge == 'sum':
                                new_embedding = np.sum(np.array([bert_embeddings[idx].numpy() for 
                                                                 idx in chunk_id]), axis=0)
                            elif self.merge == 'avg':
                                new_embedding = np.mean(np.array([bert_embeddings[idx].numpy() for 
                                                                 idx in chunk_id]), axis=0)
                            elif self.merge == 'max':
                                stacked_embeddings = np.vstack([bert_embeddings[idx].numpy() for 
                                                                 idx in chunk_id])
                                new_embedding = self.max_pooling(stacked_embeddings)
                            else:
                                raise Exception('Embeddings have to be summed, averaged or max-pooled')
                            new_embeddings.append(new_embedding)
                        tok_span = len(chunk_id)
                        if tok_span > 2:
                            pop_idx = 0
                            for k in range(tok_span-1):
                                bert_ids, bert_toks, bert_embeddings, chunk_ids = self.remove_elements(\
                                    bert_ids,bert_toks,bert_embeddings,chunk_id[k]-pop_idx,chunk_ids)
                                pop_idx += 1
                        elif tok_span == 2:
                            bert_ids, bert_toks, bert_embeddings, chunk_ids = self.remove_elements(\
                                bert_ids,bert_toks,bert_embeddings,chunk_id[0],chunk_ids)                  
                        break
                    else:
                        new_ids.append(bert_id)    
                        new_toks.append(bert_tok)
                        if self.embeddings:
                            new_embeddings.append(bert_embedding.numpy())
                        break
            else:
                new_ids.append(bert_id)    
                new_toks.append(bert_tok)
                if self.embeddings:
                    new_embeddings.append(bert_embedding.numpy())
                    
        if self.embeddings:            
            return new_ids, new_toks, new_embeddings, arbitrary_id
        else:
            return new_ids, new_toks, arbitrary_id
