import os
import csv
import json
import pickle
import pandas as pd
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer, util
from util import prepro_sentence, prepro_sentence_pair, \
    prepro_sentence_pair_single



def load_data_cross_tool(data_dir, task, k, seed, split):

    project = task.split('_')[-1].strip()
    
    # if split == 'test':
    #     corpora_file = 'data/patch_cor/' + project + '_' + split  + '_v1.csv'
    # else:
    #     corpora_file = 'data/patch_cor/' + project + '_' + split  + '_cross_v1.csv'
    if split == 'test':
        corpora_file = data_dir + '/patch_cor/' + project + '_' + split  + '_v1.csv'
    else:
        corpora_file = data_dir +  '/patch_cor/' + project + '_' + split  + '_cross_v1.csv'
    # corpora_file = 'data/patch_cor/' + project + '_' + split + '_v1.csv'
    data = []
    print('\ncorpora_file:'+corpora_file+'\n')
    with open(corpora_file, "r") as f:
        for label, text in csv.reader(f):
            data.append((text, label))
    assert np.all([len(dp) == 2 for dp in data])
    return data



def prepare_data(args, k, tokenizer, train_data, test_data, max_length, max_length_per_example,
                 n_classes=2, templates=None, method_type="generative",
                 is_training=False, use_demonstrations=False,
                 ensemble=False, is_null=False):

    if type(templates)==list:
        transform = None
        assert len(templates)==n_classes
    else:
        transform = templates
    assert method_type in ["direct", "channel"]

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if bos_token_id is None and eos_token_id is not None:
        bos_token_id = eos_token_id
    elif bos_token_id is not None and eos_token_id is None:
        eos_token_id = bos_token_id
    elif pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    elif pad_token_id is None and bos_token_id is not None:
        pad_token_id = bos_token_id



    # For calibration method, following Zhao et al. 2021
    if is_null:
        assert test_data is None
        assert method_type=="direct"
        test_data = [("N/A", "0")]

    prefixes_with_space = None
    if transform is None:
        templates = [template.strip() for template in templates]
        if method_type=="direct":
            templates = [" "+template for template in templates]
            if use_demonstrations:
                test_data = [(" "+sent, label) for sent, label in test_data]
        elif method_type=="channel":
            test_data = [(" "+sent, label) for sent, label in test_data]
            if train_data is not None:
                train_data = [(" "+sent, label) for sent, label in train_data]
            prefixes_with_space = [tokenizer(" "+template)["input_ids"] for template in templates]
        else:
            raise NotImplementedError()


    sent_embedder = SentenceTransformer('./pretrained_model/400/')

    corpus = [d[0]    for d in  train_data]
    corpus = [re.sub(r"\s+", " ", l)  for l in corpus]
    corpus_labels =  [d[1] for d in train_data]


    corpus_embeddings = sent_embedder.encode(corpus, convert_to_tensor=True)
    top_k = k
    from collections import Counter
    print('test label', Counter([d[1] for d in test_data]))
    print('train label', Counter(corpus_labels))

    if 'bear' in args.task or 'bear' in args.data_dir:
        enhanced_data_test = pickle.load(open(args.data_dir+'/patch_cor/' + args.task.split('_')[1] + '_test_v1_enhanced.pkl', 'rb'))
        target_label = {1: 'wrong', 0: 'correct'}

        enhanced_data_labels, enhanced_data_patches, enhanced_data_bugids, enhanced_data_all_failing_test_case_names, enhanced_data_all_test_coverages, enhanced_data_all_buggy_informations, enhanced_data_all_execution_traces = enhanced_data_test
        enhanced_data_test_data = []
        for i in range(len(enhanced_data_labels)):
            test_sample = {'target': enhanced_data_labels[i], 'input': enhanced_data_patches[i],
                           'bugid': enhanced_data_bugids[i],
                           'test_case_name': enhanced_data_all_failing_test_case_names[i],
                           'test_coverage': enhanced_data_all_test_coverages[i],
                           'bug_info': [enhanced_data_all_buggy_informations[i]],
                           'trace': [enhanced_data_all_execution_traces[i]]
                           }
            enhanced_data_test_data.append(test_sample)
        assert (len(test_data) == len(enhanced_data_test_data))

    elif 'patchsim' in args.data_dir:
        enhanced_data_test = pickle.load(
            open(args.data_dir+'/patch_cor/' + args.task.split('_')[1] + '_test_v1_enhanced.pkl', 'rb'))
        target_label = {1: 'wrong', 0: 'correct'}
        enhanced_data_labels, enhanced_data_patches, enhanced_data_bugids, enhanced_data_all_failing_test_case_names, enhanced_data_all_failing_test_case_methods, enhanced_data_all_test_coverages, enhanced_data_all_buggy_informations, enhanced_data_all_execution_traces = enhanced_data_test
        enhanced_data_test_data = []
        for i in range(len(enhanced_data_labels)):
            test_sample = {'target': enhanced_data_labels[i], 'input': enhanced_data_patches[i],
                           'bugid': enhanced_data_bugids[i],
                           'test_case_name': enhanced_data_all_failing_test_case_names[i],
                           'test_case_method': enhanced_data_all_failing_test_case_methods[i],
                           'test_coverage': enhanced_data_all_test_coverages[i],
                           'bug_info': enhanced_data_all_buggy_informations[i],
                           'trace': enhanced_data_all_execution_traces[i]
                           }
            enhanced_data_test_data.append(test_sample)
        print(len(test_data), len(enhanced_data_test_data))
        assert (len(test_data) == len(enhanced_data_test_data))
    else:
        enhanced_data_test =  pickle.load(open('data_checked/patch_cor/' + args.task.split('_')[1] + '_test_v1_enhanced.pkl', 'rb'))
        target_label = {1: 'wrong', 0: 'correct'}
        enhanced_data_labels, enhanced_data_patches, enhanced_data_bugids, enhanced_data_all_failing_test_case_names, enhanced_data_all_failing_test_case_methods, enhanced_data_all_test_coverages, enhanced_data_all_buggy_informations, enhanced_data_all_execution_traces = enhanced_data_test
        enhanced_data_test_data = []
        for i in range(len(enhanced_data_labels)):
            test_sample = {'target': enhanced_data_labels[i], 'input': enhanced_data_patches[i], 'bugid': enhanced_data_bugids[i],
                           'test_case_name': enhanced_data_all_failing_test_case_names[i],
                           'test_case_method': enhanced_data_all_failing_test_case_methods[i], 'test_coverage': enhanced_data_all_test_coverages[i],
                           'bug_info': enhanced_data_all_buggy_informations[i], 'trace': enhanced_data_all_execution_traces[i]
                           }
            enhanced_data_test_data.append(test_sample)
        # print(len(test_data), len(enhanced_data_test_data))
        assert (len(test_data) == len(enhanced_data_test_data))



    if transform is None:
        test_inputs = [tokenizer(sent)["input_ids"] for sent, _ in test_data]

        test_inputs = []
        ijk_index = 0

        sub_enhance_options = args.enhancement_option.split('-')

        for sent, label in test_data:

            encode_sent = ''

            for sub_option in sub_enhance_options:
                print(sub_option)
                if sub_option == 'similar':

                    ## -------- similar samples from the training data -------- ##

                    encode_sent += "\nFor your reference, there are labeled examples similar to the input patch: "

                    sent = re.sub(r"\s+", " ", sent)

                    query_embedding = sent_embedder.encode(sent, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    # print(cos_scores, cos_scores.size())
                    # top_results = torch.topk(cos_scores, k=top_k)
                    top_results = torch.topk(cos_scores, k=args.top_k_example)

                    knn_label = []
                    knn_similarty = []
                    knn_cont = []
                    # label_dict = {'1':'wrong.\n', '0':'correct.\n', '-1': ' unclear. '} ## the used one

                    print(top_results[0])
                    for score, idx__ in zip(top_results[0], top_results[1]):
                        knn_label.append(corpus_labels[idx__])
                        knn_similarty.append(score)
                        knn_cont.append(corpus[idx__])


                    for i in range(len(knn_similarty)):
                        # if knn_similarty[i] > 0.9:
                        # if 'bear' not in args.task:
                        if knn_similarty[i] > args.sim_threshold:
                            encode_sent += ' '.join(knn_cont[i].split()[:int(1*knn_similarty[i]*len(knn_cont[i].split()))]) + templates[int(knn_label[i])] + ' '
                            # encode_sent += ' '.join(knn_cont[i].split()[:int(1 * knn_similarty[i] * len(knn_cont[i].split()))]) + '\n Q: It was wrong or correct? A: It was ' + label_dict[knn_label[i]]


                elif sub_option == 'bug':
                    try:
                        bug_info = ' and '.join(enhanced_data_test_data[ijk_index]['bug_info'])
                        bug_info = "\nThe bug refers to " + bug_info + " \n"
                    except:
                        bug_info = "\nThe bug info is not available now \n"
                    encode_sent += bug_info

                elif sub_option == 'coverage':
                    try:
                        coverage = enhanced_data_test_data[ijk_index]['test_coverage']
                        coverage = "\nAlthough this patch can pass available test cases, the available test cases only cover limited coverages: \n" + coverage + "\n"
                    except:
                        coverage = "\nThe coverage info is not available now \n"
                    encode_sent += coverage

                elif sub_option == 'testcase':
                    if not ('bear' in args.task or 'bear' in args.data_dir):
                        if 'test_case_name' in enhanced_data_test_data[ijk_index] and 'test_case_method' in enhanced_data_test_data[ijk_index]:
                            test_names_ = enhanced_data_test_data[ijk_index]['test_case_name']
                            test_methods_ = enhanced_data_test_data[ijk_index]['test_case_method']
                            test_methods = [test_methods_[n] for n in test_names_]
                            new_test_methods = []
                            for t in test_methods:
                                if t is None:
                                    continue
                                else:
                                    new_test_methods.append(t)
                            test_methods = new_test_methods
                            test_methods = '\n'.join(test_methods)
                            test_methods = "\nOriginally the buggy code cannot pass some failing test cases and now the patched code can pass them. Those failing test cases are:\n" + test_methods + "\n"
                            encode_sent += test_methods
                    else:
                            test_names_ = enhanced_data_test_data[ijk_index]['test_case_name']
                            test_methods = "\nOriginally the buggy code cannot pass some failing test cases and now the patched code can pass them. Those failing test cases are:\n" + test_names_ + "\n"
                            encode_sent += test_methods

                elif sub_option == 'trace':
                    traces =  enhanced_data_test_data[ijk_index]['trace']
                    traces = '\n'.join(traces)
                    traces = "\nThe execution traces of the bug are: " + traces + "\n"
                    traces = '\n'.join(traces.split('\n')[0:30])
                    encode_sent += traces


            encode_sent += sent
            print(encode_sent)
            print()
            print(sent)
            print()
            test_inputs.append(tokenizer(encode_sent)["input_ids"])
            ijk_index += 1



        truncated = np.sum([len(inputs)>max_length_per_example-16 for inputs in test_inputs])

        if truncated > 0:
            test_inputs = [inputs[:max_length_per_example-16] for inputs in test_inputs]


        prefixes = [tokenizer(template)["input_ids"] for template in templates]
        idx = [idx for idx, _prefixes in enumerate(zip(*prefixes))
                if not np.all([_prefixes[0]==_prefix for _prefix in _prefixes])][0]


    else:
        test_inputs = [transform(dp, tokenizer,
                                 max_length_per_example-16,
                                 groundtruth_only=is_training)
                                   for dp in test_data]
        if not is_training:
            assert np.all([len(dp)==2 and
                           np.all([len(dpi)==n_classes for dpi in dp])
                           for dp in test_inputs])


    if is_training:
        assert not use_demonstrations
        assert not ensemble

        input_ids, attention_mask, token_type_ids = [], [], []
        for test_input, dp in zip(test_inputs, test_data):
            if transform is not None:
                test_input, test_output = test_input
                encoded = prepro_sentence_pair_single(
                    test_input, test_output, max_length, bos_token_id, eos_token_id, pad_token_id
                )
            else:
                prefix = prefixes[int(dp[1])]
                if method_type=="channel":
                    encoded = prepro_sentence_pair_single(
                        prefix, test_input, max_length, bos_token_id, eos_token_id, pad_token_id)
                elif method_type=="direct":
                    encoded = prepro_sentence_pair_single(
                        test_input + prefix[:idx], prefix[idx:], max_length, bos_token_id, eos_token_id, pad_token_id)
                else:
                    raise NotImplementedError()
            input_ids.append(encoded[0])
            attention_mask.append(encoded[1])
            token_type_ids.append(encoded[2])
        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))

    if use_demonstrations:

        if transform is not None:
            raise NotImplementedError()

        if ensemble:
            return prepare_data_for_parallel(
                tokenizer, train_data, test_data,
                max_length, max_length_per_example,
                method_type, n_classes,
                test_inputs, prefixes, idx, prefixes_with_space,
                bos_token_id, eos_token_id)


        assert train_data is not None
        demonstrations = []

        np.random.shuffle(train_data)

        for sent, label in train_data:
            if len(demonstrations)>0:
                if method_type=="direct":
                    sent = " " + sent
                elif method_type=="channel":
                    prefixes = prefixes_with_space

            if transform is None:
                tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
            else:
                tokens = transform(sent, tokenizer, max_length_per_example)
            prefix = prefixes[(int(label))]

            if method_type=="channel":
                tokens = prefix + tokens
            elif method_type=="direct":
                tokens = tokens + prefix
            else:
                raise NotImplementedError()

            demonstrations += tokens

    if transform is None:
        # check if idx is set well
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                assert prefixes[i][:idx]==prefixes[j][:idx]
                assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):
        if transform is None:
            prefix = prefixes[i].copy()

            if method_type=="channel":
                if use_demonstrations:
                    prefix = demonstrations.copy() + prefix
                tensor = prepro_sentence_pair([prefix], test_inputs, max_length,
                                            bos_token_id, eos_token_id, pad_token_id,
                                            )
            elif method_type=="direct":
                if use_demonstrations:
                    prompt = [demonstrations.copy() + test_input + prefix[:idx] for test_input in test_inputs]
                else:
                    prompt = [test_input + prefix[:idx] for test_input in test_inputs]
                tensor = prepro_sentence_pair(prompt,[prefix[idx:]], max_length,
                                            bos_token_id, eos_token_id, pad_token_id,
                                            )
            else:
                raise NotImplementedError()
        else:
            input_ids, attention_mask, token_type_ids = [], [], []
            for input_, output_ in test_inputs:
                encoded = prepro_sentence_pair_single(
                    input_[i], output_[i], max_length,
                    bos_token_id,
                    None if is_generation else eos_token_id,
                    allow_truncation=False)
                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])
            tensor = dict(input_ids=torch.LongTensor(input_ids),
                          attention_mask=torch.LongTensor(attention_mask),
                          token_type_ids=torch.LongTensor(token_type_ids))

        input_tensors.append(tensor)


    return input_tensors


def prepare_data_for_parallel(tokenizer, train_data, test_data,
                              max_length, max_length_per_example,
                              method_type, n_classes,
                              test_inputs, prefixes, idx, prefixes_with_space,
                              bos_token_id, eos_token_id):

    # get len(train_data) number of demonstrations

    assert train_data is not None
    demonstrations_list = []

    np.random.shuffle(train_data)

    for sent, label in train_data:
        tokens = tokenizer(sent)["input_ids"][:max_length_per_example]
        prefix = prefixes[(int(label))]
        if method_type=="channel":
            tokens = prefix + tokens
        elif method_type=="direct":
            tokens = tokens + prefix
        else:
            raise NotImplementedError()

        demonstrations_list.append(tokens)

    # check if idx is set well
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            assert prefixes[i][:idx]==prefixes[j][:idx]
            assert prefixes[i][idx]!=prefixes[j][idx]

    input_tensors = []

    for i in range(n_classes):

        if method_type=="channel":
            prefix = prefixes_with_space[i].copy()
            prompt = [demonstrations + prefix
                      for demonstrations in demonstrations_list]
            tensor = prepro_sentence_pair(
                prompt, test_inputs, max_length,
                bos_token_id, eos_token_id, pad_token_id,
                allow_truncation=True)

        elif method_type=="direct":
            prefix = prefixes[i].copy()
            prompt = [demonstrations.copy() + test_input + prefix[:idx]
                      for test_input in test_inputs
                      for demonstrations in demonstrations_list]

            tensor = prepro_sentence_pair(prompt,
                                          [prefix[idx:]], max_length,
                                          bos_token_id, eos_token_id, pad_token_id,
                                          allow_truncation=True)
        else:
            raise NotImplementedError()

        input_tensors.append(tensor)


    return input_tensors

