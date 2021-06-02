import pandas as pd
import os
import pickle
import collections
import random
import tensorflow as tf
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.datasets import load_from_csv
from tqdm import tqdm

filedir = os.path.dirname(__file__)
Input_dir = os.path.join(filedir,'INPUT')
code_desc_filename = os.path.join(Input_dir,"code_desc.csv")
code_relation_filename = os.path.join(Input_dir,"code_relation_code.csv") #columns = ['code_1', 'relation', 'code_2']
#code_desc_filename = "./INPUT/code_desc_100.csv"
#code_relation_filename = "./INPUT/code_relation_code_100.csv" #columns = ['code_1', 'relation', 'code_2']


text_out_filename = os.path.join(Input_dir,"codes_input_full.txt")
text_out_filename_train = os.path.join(Input_dir,"codes_input_train.txt")
text_out_filename_test = os.path.join(Input_dir,"codes_input_test.txt")

desc_out_filename = os.path.join(Input_dir,"desc_input_train.txt")
kg_out_filename = os.path.join(Input_dir,"kg_input_train.txt")

code_index_filename = os.path.join(Input_dir,"codes_vocab.pkl")
rel_index_filename = os.path.join(Input_dir,"rel_vocab.pkl")
node_rel_index_filename = os.path.join(Input_dir,"node_rel_vocab.pkl")

#text_out_filename = "./INPUT/codes_input_train.txt"

code_column = "code"
desc_column = "description"

test_size = .2

def _parse(line):
    """Parse train data."""
    cols_types = [[1], [1], [1]]
    return tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')
    
# if not os.path.isfile(text_out_filename):
if True:
    df_desc = pd.read_csv(code_desc_filename)
    df_desc = df_desc.sample(frac=1).reset_index(drop=True)
    df_desc[[desc_column]].to_csv(desc_out_filename, header=False, index=False, sep='\t')
    # create input text filename for training
    #with open(text_out_filename, "w", encoding='utf-8') as text_file:
    #    for ind, row in df_desc.iterrows():
    #        text_file.write(row[desc_column] + "\n")
    #    text_file.close()

    num_codes = df_desc.shape[0]
    # save dict of code, index and desc (if required)
    code_desc  = dict(zip(df_desc[code_column], df_desc[desc_column]))

    code_index = dict(zip(df_desc[code_column], range(df_desc.shape[0])))

    index_code = dict(zip(range(df_desc.shape[0]), df_desc[code_column]))

    df_rel = pd.read_csv(code_relation_filename)
    print("df_rel Shape" , df_rel.shape)

    # save relations tuple in a dict { key : code_index , value: list of list([code_1, relation, code_2])}
    num_rel = 0
    rel_index = dict()

    #rel_tuple = dict(zip(index_code.keys(), [{'t':[],'r':[]} for _ in range(len(index_code))]))
    rel_tuple = dict(zip(index_code.keys(), [[] for _ in range(len(index_code))]))
    
    node_rel_dict = dict()
    
    l_h = []
    l_t = []
    l_r = []
    
    for ind, row in tqdm(df_desc.iterrows()):
        df_desc.loc[ind, code_column] = code_index[row[code_column]]
        
    for ind, row in tqdm(df_rel.iterrows()):
    
        try:
            code_1 = code_index[row['code_1']]
            code_2 = code_index[row['code_2']]
            if row['relation'] in rel_index:
                relation = rel_index[row['relation']]
            else:
                rel_index[row['relation']] = num_rel
                relation = rel_index[row['relation']]
                num_rel += 1
                
            #rel_tuple[code_1]['t'].append(code_2)
            #rel_tuple[code_1]['r'].append(relation)
            rel_tuple[code_1].append([code_1, relation, code_2]) 
            if code_1 not in node_rel_dict:
                node_rel_dict[code_1] = {}
                
            if relation not in node_rel_dict[code_1]:
                node_rel_dict[code_1][relation] = [code_2]
            else:
                node_rel_dict[code_1][relation].append(code_2)
            
            l_h.append(code_1)
            l_t.append(code_2)
            l_r.append(relation)
        except:
            pass
            
    new_rel = pd.DataFrame()
    new_rel['code_1'] = l_h
    new_rel['code_2'] = l_t
    new_rel['relation'] = l_r
    
    df_desc = pd.merge(df_desc, new_rel, how='left', left_on=[code_column], right_on=['code_1'])
    df_desc['code_1'] = df_desc.apply(lambda x: str(int(x.code_1)) if pd.notnull(x.code_1) else str(int(x[code_column])), axis=1)
    df_desc['code_2'] = df_desc.apply(lambda x: str(int(x.code_2)) if pd.notnull(x.code_2) else str(int(x[code_column])), axis=1)
    df_desc['relation'] = df_desc['relation'].apply(lambda x: str(int(x)) if pd.notnull(x) else str(1))
    
    df_desc['code_1'] = df_desc['code_1'].astype(str)
    df_desc['code_2'] = df_desc['code_2'].astype(str)
    df_desc['relation'] = df_desc['relation'].astype(str)
    
    df_desc = df_desc.dropna()
    
    df_desc[[desc_column, 'code_1', 'code_2', 'relation']].to_csv(text_out_filename, header=False, index=False, sep='\t')
    
    pickle.dump(code_index, open(code_index_filename, 'wb'))
    pickle.dump(rel_index, open(rel_index_filename, 'wb'))
    pickle.dump(node_rel_dict, open(node_rel_index_filename, 'wb'))
    
    #df_desc = df_desc.drop_duplicates(subset=['code_1']).reset_index(drop=True)
    print ("Splitting into train and test for KG validation")
    X = load_from_csv(directory_path="",file_name=text_out_filename, sep='\t')
    X_train, X_test = train_test_split_no_unseen(X, test_size=int(test_size*X.shape[0]))
    X_train = pd.DataFrame(X_train, columns=[desc_column, 'code_1', 'code_2', 'relation'])
    X_test = pd.DataFrame(X_test, columns=[desc_column, 'code_1', 'code_2', 'relation'])
    X_train[[desc_column, 'code_1', 'code_2', 'relation']].to_csv(text_out_filename_train, header=False, index=False, sep='\t')
    X_test[[desc_column, 'code_1', 'code_2', 'relation']].to_csv(text_out_filename_test, header=False, index=False, sep='\t')
    
    new_rel.to_csv(kg_out_filename, sep='\t', index=False, header=False)
    
    # save count of relations for each code in dict { key : code_index , value: #relation with head as code_index }
    rel_count = dict()

    for key in rel_tuple:
        rel_count[key] = len(rel_tuple[key])

    #print("rel_count dict = ", rel_count)
    #print("___rel_tuple = ", rel_tuple)
    
    print ("Building TF KG dataset")
    tf_kg_dataset = tf.data.TextLineDataset(kg_out_filename)
    tf_kg_dataset = tf_kg_dataset.map(_parse, num_parallel_calls=4)
    
    tf_kg_dataset = tf_kg_dataset.map(lambda hi, ti, ri:
                    tf.tuple([tf.cast(hi, tf.int32),
                    tf.cast(ti, tf.int32),
                    tf.cast(ri, tf.int32),
                    tf.cast(random.choice(list(rel_tuple.keys())), tf.int32)]))
    
