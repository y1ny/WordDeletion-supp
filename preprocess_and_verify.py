import re
import glob
import pandas as pd
import csv
import os

# remove extra whitespace from a string and normalize spaces
def clean_str(raw_str):
    raw_str = raw_str.split(' ')
    raw_str = list(filter(lambda x: x and x.strip(), raw_str))
    return ' '.join(raw_str)

# Check if string_B is a subsequence (not necessarily contiguous) of string_A.
# This means characters can match with gaps, which may produce unexpected matches                                                      
# (e.g., "ac" matches "abc"). To handle such edge cases, input is processed                                                         
# by a separate function before constituent rate analysis 
# (lines 277-283, https://github.com/y1ny/WordDeletion/blob/main/scripts/process_ptb.py). 
def is_substring(string_B, string_A):
    pointer_A = 0
    pointer_B = 0

    while pointer_A < len(string_A) and pointer_B < len(string_B):
        if string_A[pointer_A] == string_B[pointer_B]:
            pointer_B += 1
        pointer_A += 1

    return pointer_B == len(string_B)

# process raw data to get the response of LLMs
# the code is almost the same to the code released in the github repo:
# line 187, https://github.com/y1ny/WordDeletion/blob/main/utils.py
def extract_response(test_sentence, resp):
    sent_raw = test_sentence
    test_sentence = re.sub(r'\W', ' ', test_sentence)
    test_sentence = clean_str(test_sentence).lower()
    
    # normalize punctuation in response
    resp = resp.replace('.', ' . ')
    resp = resp.replace(',', ' ')

    # process 's, 'd
    tmp = []
    for idx, ch in enumerate(resp):
        if ch in ['\'', '\"']:
            if idx == 0 or idx == len(resp) - 1:
                tmp.append(ch)
            elif ch == "\'" and resp[idx+1] in ['s', 'r', 'l','m','t','v'] and resp[idx-1] != ' ':
                continue
            else:
                tmp.append(ch)
        else:
            tmp.append(ch)
    resp = ''.join(tmp)
    
    # use re to extract quoted text as candidate predictions
    p1 = re.compile(r'["](.*?)["]', re.S)
    p2 = re.compile(r"['](.*?)[']", re.S)

    candidate_pred = re.findall(p1, resp)
    candidate_pred.extend(re.findall(p2, resp))

    # also use the entire response as a candidate
    candidate_pred.append(resp)
    # keep only alphabetic chars and spaces, then deduplicate and filter:
    # - must be a subsequence of test_sentence determined by is_substring(),
    #   which checks for non-contiguous matches (see note above)
    # - must have more than 1 word, since single words may coincidentally match                                                        
    #   function words in the original test sentence.
    # - must not be identical to the original sentence
    tmp = []
    for pred in candidate_pred:
        t = ''.join(list(filter(lambda x: x.isalpha() or x.isspace(), pred)))
        tmp.append(clean_str(t))

    candidate_pred = tmp
    candidate_pred = list(set(candidate_pred))
    candidate_pred = list(map(lambda x: x.lower(), candidate_pred))
    candidate_pred = list(filter(lambda x: is_substring(x, test_sentence), candidate_pred))
    candidate_pred = list(map(lambda x: x.split(' '), candidate_pred))

    candidate_pred = list(filter(lambda x: len(x)!=1, candidate_pred))
    candidate_pred = list(filter(lambda x: ' '.join(x) != test_sentence.lower(), candidate_pred))
    
    if candidate_pred == []:        
        return [sent_raw, 'fail to follow']
    
    # sort candidates by length, then rank by positional alignment with original sentence
    pred = ['']
    candidate_pred = sorted(candidate_pred, key = lambda x: len(''.join(x)))
    candidate_pred = sorted(candidate_pred, key = lambda x: len(x))
    target_lst = test_sentence.split(' ')
    for i in range(len(candidate_pred[-1])-1, -1, -1):
        for j in range(len(test_sentence.split(' '))-1, i-1, -1):
            candidate_pred = sorted(candidate_pred, key = lambda x: int(x[min(i, len(x)-1)].lower() == target_lst[j].lower()))

    # pick the last (best-ranked) valid candidate
    for candidate in candidate_pred:
        if ' '.join(candidate) == test_sentence.lower():
            continue
        if len(candidate) == 1:
            continue
        pred = candidate
    if pred == [] :
        return [sent_raw, 'fail to follow']
    else:
        return [sent_raw, " ".join(pred)]

# Chinese version of extract_response, same logic but handles Chinese characters
# the code is almost the same to the code released in the github repo:
# line 250, https://github.com/y1ny/WordDeletion/blob/main/utils.py
def extract_response_zh(test_sentence, resp):
    sent_raw = test_sentence
    test_sentence = re.sub(r'\W', ' ', test_sentence)
    test_sentence = clean_str(test_sentence).lower()
    
    if pd.isna(resp):
        return [sent_raw, 'fail to follow']
    resp = resp.replace('.', ' . ')
    resp = resp.replace(',', ' . ')

    # common patterns of response (quotes, Chinese punctuation, etc.)
    p1 = re.compile(r'["](.*?)["]', re.S) 
    p2 = re.compile(r"['](.*?)[']", re.S) 
    p3 = re.compile(r'[‘](.*?)[’]', re.S) 
    p4 = re.compile(r"[“](.*?)[”]", re.S) 
    p5 = re.compile(r"^(.*?)[。]", re.S) 
    p6 = re.compile(r"^(.*?)[，]", re.S) 
    p7 = re.compile(r"^.*?[，](.*?)[。]", re.S) 

    candidate_pred = re.findall(p1, resp)
    candidate_pred.extend(re.findall(p2, resp))
    candidate_pred.extend(re.findall(p3, resp))
    candidate_pred.extend(re.findall(p4, resp))
    candidate_pred.extend(re.findall(p5, resp))
    candidate_pred.extend(re.findall(p6, resp))
    candidate_pred.extend(re.findall(p7, resp))
    candidate_pred.append(resp)
    
    # keep only Chinese characters, deduplicate, and filter:
    # - must be a subsequence of test_sentence
    # - must have more than 1 char
    # - must not be identical to the original sentence
    tmp = []
    for pred in candidate_pred:
        t = ''.join(list(filter(lambda x: u'\u4e00' <= x <= u'\u9fff', pred)))
        tmp.append(t)
    candidate_pred = tmp
    candidate_pred = list(set(candidate_pred))
    candidate_pred = list(map(lambda x: list(x), candidate_pred))
    candidate_pred = list(filter(lambda x: x, candidate_pred))
    candidate_pred = list(filter(lambda x: is_substring(''.join(x), test_sentence), candidate_pred))
    candidate_pred = list(filter(lambda x: len(x)!=1, candidate_pred))
    candidate_pred = list(filter(lambda x: ''.join(x).lower() != test_sentence, candidate_pred))
            
    if candidate_pred == []:
        return [sent_raw, 'fail to follow']
    # sort candidates by length, then rank by positional alignment with original sentence
    pred = ['']
    candidate_pred = sorted(candidate_pred, key = lambda x: len(''.join(x)))
    candidate_pred = sorted(candidate_pred, key = lambda x: len(x))
    target_lst = list(test_sentence)
    for i in range(len(candidate_pred[-1])-1, -1, -1):
        for j in range(len(test_sentence)-1, i-1, -1):
            candidate_pred = sorted(candidate_pred, key = lambda x: int(x[min(i, len(x)-1)] == target_lst[j]))

    # pick the last (best-ranked) valid candidate
    for candidate in candidate_pred:
        if ''.join(candidate) == test_sentence:
            continue
        if len(candidate) == 1:
            continue
        pred = candidate
        pred = candidate
    if pred == [] :
        return [sent_raw, 'fail to follow']
    else:
        return [sent_raw, "".join(pred)]
    
# process raw LLM data and verify against github data
if __name__ == "__main__":
    for exp_type in ['exp1','exp2','exp3','exp4','exp5', 'exp6']:
        for lang in ['english', 'chinese']:

            # find all raw csv files for this experiment and language
            dir_path = f"./{exp_type}/{lang}/raw/*.csv"
            raw_data_path = glob.glob(dir_path)
            if not raw_data_path:
                print('skipp, no data:', dir_path)
            for p in raw_data_path:
                file_name = os.path.split(p)[-1]
                # process the raw data
                lst = []
                data_file = pd.read_csv(p,
                                    delimiter='\t',
                                    quoting=csv.QUOTE_NONE,
                                    quotechar=None,)
                for idx, row in data_file.iterrows():
                    test_sentence = row['sentence']
                    resp = row['response']
                    demon = row['demonstration']
                    # preprocess
                    if lang == 'english':
                        processed = extract_response(test_sentence, resp)
                    elif lang == 'chinese':
                        processed = extract_response_zh(test_sentence, resp)
                    lst.append([demon]+processed+[resp, p])
                
                # load the reference data from github repo and compare row by row
                github_path = f"./{exp_type}/{lang}/github/{file_name}"
                github_file = pd.read_csv(github_path,
                                    delimiter='\t',
                                    quoting=csv.QUOTE_NONE,
                                    quotechar=None,)
                github_lst = []
                for idx, row in github_file.iterrows():
                    github_lst.append(row.tolist())
                
                # compare processed results with github data (column layout differs by exp)
                for l, g in zip(lst, github_lst):
                    if exp_type in ['exp2', 'exp4']:
                        # the exp2 exp4 data in github did not contain the demonstration
                        if l[1:3] != [g[0],g[2]]:
                            print("error for: ", l, g)
                    elif exp_type in ['exp5']:
                        # the exp5 data in github have different colums setting
                        # '##' is for the detailed 'fail to follow' reason
                        if '##' in g[2]:
                            g[2] = 'fail to follow'
                        if l[:3] != [g[-1], g[0], g[2]]:
                            print("error for: ", l, g)
                    elif exp_type in ['exp6']:
                        # the exp6 data in github have different colums setting
                        if l[:3] != [g[0],g[2],g[3]]:
                            print("error for: ", l, g)
                    else:
                        if l[:3] != g:
                            print("error for: ", l, g)
            print(f"{exp_type}-{lang} check finish")
                
            
                