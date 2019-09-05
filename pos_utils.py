import re
import nltk

def concatenate_sql_data(sql_data_train, sql_data_val):
    sql_data_train.extend(sql_data_val)
    return sql_data_train

def annotate_pos(question_tok_concol, pos_tags):
    pos_concol = list()
    tok_span = 0
    for tok_concol in question_tok_concol:
        if len(tok_concol) > 1:
            tok_span += len(tok_concol)
            pos_concol.append([pos for pos in pos_tags[tok_span-len(tok_concol):tok_span]])
        else:
            pos_concol.append(pos_tags[tok_span].split())
            tok_span += 1
    return pos_concol

def update_sql_data_pos(sql_data):
    """
    Update SQL data with POS tags for each question.
    """
    for i, question in enumerate(sql_data):
        tok_pos_tags = nltk.pos_tag(question['question_tok'])
        _, pos_tags = zip(*tok_pos_tags)
        assert len(question['question_tok']) == len(pos_tags)
        sql_data[i]['question_tok_pos'] = list(pos_tags)
        pos_tags_concol = annotate_pos(question['question_tok_concol'], list(pos_tags))
        assert len(question['question_tok_concol']) == len(pos_tags_concol)
        sql_data[i]['question_tok_concol_pos'] = pos_tags_concol
    return sql_data