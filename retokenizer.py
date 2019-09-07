import numpy as np
import re
import torch
import unicodedata

class Retokenizer:
    
    def __init__(self, merge, embeddings: bool):
        """
            Input: init method takes a string for specifing the merging computation of embeddings 
            and a boolean value that indicates whether embeddings are passed or not.
        """
        self.merge = merge
        # set embeddings to False to compute token rejoining step faster (for updating sql data)
        self.embeddings = embeddings
    
    def rejoin(self, bert_toks, i, typesql_tok):
        indexes=[]
        str_length=0
        for idx, bert_tok in enumerate(bert_toks[i:]):
            bert_tok_str = bert_tok.strip('##') if re.search('#{2,}', bert_tok) else bert_tok
            try:
                #TODO: fix Chinese and Japanese characters - this is a hack and not clean (we should not just join them)
                if not re.match(typesql_tok, bert_tok) and self.is_cjk(bert_tok_str):
                    indexes.append(i+idx)
                    str_length += len(bert_tok_str)
                    continue
            except:
                pass
            if not re.match(typesql_tok, bert_tok) and bert_tok_str == typesql_tok[str_length:str_length+len(bert_tok_str)]:
                indexes.append(i+idx)
                str_length += len(bert_tok_str)
            #TODO: fix UNKs - this is a hack and not clean (we should not join UNKs)
            elif not re.match(typesql_tok, bert_tok) and bert_tok == '[UNK]':
                indexes.append(i+idx)
            else:
                break
        try:
            rejoined_tok = ''.join([tok.strip('##') for tok in bert_toks[i:indexes[-1]+1]])
        except:
            return False
        return rejoined_tok, indexes


    def retokenize(self, typesql_toks, bert_toks, bert_ids, arbitrary_id, bert_embeddings=None):
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
        if not self.embeddings:
            # create placeholder values for embeddings to compute loop over all arrays simultaneously
            bert_embeddings = [0 for _ in range(len(bert_toks))]
            assert len(bert_toks) == len(bert_ids) == len(bert_embeddings), 'all arrays must have same number of elements'
        j = 0
        for i, (bert_tok, bert_id, bert_embedding) in enumerate(zip(bert_toks, bert_ids, bert_embeddings)):
            for typesql_tok in typesql_toks[j:]:
                typesql_tok = typesql_tok.strip()
                # remove all diacritics (since they are automatically removed by WordPiece tokenizer)
                typesql_tok = self.remove_accents(typesql_tok)
                # handle special regex tokens
                if typesql_tok == '?':
                    typesql_tok = '\?'
                elif typesql_tok[0] == '?':
                    typesql_tok = '\?' + typesql_tok[1:]
                elif typesql_tok == '+':
                    typesql_tok = '\+'
                elif typesql_tok[0] == '+':
                    typesql_tok = '\+' + typesql_tok[1:]
                elif typesql_tok == '*':
                    typesql_tok = '\*'
                elif typesql_tok[0] == '*':
                    typesql_tok = '\*' + typesql_tok[1:]
                elif typesql_tok == '$':
                    typesql_tok = '\$'
                elif typesql_tok == '.':
                    typesql_tok = '\.'
                elif typesql_tok == '^':
                    typesql_tok = '\^'
                elif typesql_tok == '(':
                    typesql_tok = '\('
                elif typesql_tok == ')':
                    typesql_tok = '\)'
                elif typesql_tok == '()':
                    typesql_tok = '\()'
                elif typesql_tok == '[':
                    typesql_tok = '\['
                elif typesql_tok == '[]':
                    typesql_tok = '\['
                elif typesql_tok == '\\':
                    typesql_tok = '\\' + typesql_tok
                bert_tok = self.remove_accents(bert_tok) # necessary for cases such as cm³
                try:
                    if re.match(typesql_tok, bert_tok):
                        j += 1
                        break
                except:
                    print("TypeSQL token:", typesql_tok)
                    print("BERT token:", bert_tok)
                    raise Exception
                if not re.match(typesql_tok, bert_tok):
                    if not self.rejoin(bert_toks, i, typesql_tok):
                        # rejoining did not work (most probably due to Tamil, Korean or Japanese characters
                        #                         which cannot be handled by WordPiece tokenizer)
                        if self.embeddings:
                            return bert_toks, bert_ids, bert_embeddings, arbitrary_id
                        else:
                            return bert_toks, bert_ids, arbitrary_id
                    rejoined_tok, indexes = self.rejoin(bert_toks, i, typesql_tok)
                    if self.embeddings:
                        if self.merge == 'sum':
                            merged_embedding = np.sum(np.array([bert_embeddings[idx].numpy() for 
                                                             idx in indexes]), axis=0)
                        elif self.merge == 'avg':
                            merged_embedding = np.mean(np.array([bert_embeddings[idx].numpy() for 
                                                             idx in indexes]), axis=0)
                        elif self.merge == 'max':
                            stacked_embeddings = np.vstack([bert_embeddings[idx] for 
                                                             idx in indexes])
                            merged_embedding =  np.amax(stacked_embeddings, axis=0)
                        else:
                            raise Exception('Embeddings have to be summed, averaged or max-pooled')
                    for _ in range(len(indexes)):
                        bert_toks.pop(indexes[0])
                        bert_ids.pop(indexes[0])
                        bert_embeddings = np.delete(bert_embeddings, indexes[0], axis=0)
                    bert_toks.insert(indexes[0], rejoined_tok)
                    bert_ids.insert(indexes[0], arbitrary_id)
                    if self.embeddings:
                        bert_embeddings = np.insert(bert_embeddings, indexes[0], merged_embedding, axis=0)
                    else:
                        # insert placeholder value
                        bert_embeddings.insert(indexes[0], 0)
                    arbitrary_id += 1
                    j += 1
                    break
        if self.embeddings:            
            return bert_toks, bert_ids, bert_embeddings, arbitrary_id
        else:
            return bert_toks, bert_ids, arbitrary_id

    #def find_cjk(self, char):
    #    if ord(char) > 10000:
    #        return True
    #    else:
    #        return False
    
    def is_cjk(self, char):
        """
            Handling of Chinese characters.
        """
        # list of cjk codepoint ranges
        # tuples indicate the bottom and top of the range, inclusive
        cjk_ranges = [
        ( 0x4E00,  0x62FF),
        ( 0x6300,  0x77FF),
        ( 0x7800,  0x8CFF),
        ( 0x8D00,  0x9FCC),
        ( 0x3400,  0x4DB5),
        (0x20000, 0x215FF),
        (0x21600, 0x230FF),
        (0x23100, 0x245FF),
        (0x24600, 0x260FF),
        (0x26100, 0x275FF),
        (0x27600, 0x290FF),
        (0x29100, 0x2A6DF),
        (0x2A700, 0x2B734),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F)
        ]
        char = ord(char)
        for bottom, top in cjk_ranges:
            if char >= bottom and char <= top:
                return True
        return False
    
    def remove_accents(self, input_str):
        nfkd_form = unicodedata.normalize('NFKD', str(input_str))
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])