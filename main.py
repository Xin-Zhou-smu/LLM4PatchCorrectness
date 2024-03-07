import os
import argparse
import pickle as pkl
import random
import torch
import math
import logging
import numpy as np
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from tqdm import tqdm
from collections import Counter, defaultdict

from transformers import GPT2Tokenizer, GPT2LMHeadModel


from data import prepare_data, load_data_cross_tool
from run import train, inference
from model_util import load_checkpoint, set_extra_embeddings, \
    set_separate_lm_head, set_separate_embeddings, set_transformed_lm_head
from util import get_prompts, get_paths, flatten_label_losses, \
    prepend_task_tokens, reassign_output_tokens

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, \
    roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef

import os


"""
Diverse Sources to Includes
-
- test coverage
- test cases
- traces
- bug info

Lengths Limits:
Bloom 1.7b  -----> 2,048 tokens
Codegen2_3b -----> 2,048 tokens
Codeparrot  -----> 1,024 tokens
"""


N_LABELS_DICT = {'patch_HP':2,  'patch_Developer':2, 'patch_bear':2, 'patch_ACS':2,  'patch_Arja':2,  'patch_AVATAR':2,  'patch_CapGen':2,  'patch_Cardumen':2,  'patch_DynaMoth':2,  'patch_FixMiner':2,  'patch_GenProg':2,  'patch_HDRepair':2,  'patch_Jaid':2,  'patch_jGenProg':2,  'patch_jKali':2,  'patch_jMutRepair':2,  'patch_Kali':2,  'patch_kPAR':2,  'patch_Nopol':2,  'patch_RSRepair':2,  'patch_SequenceR':2,  'patch_SimFix':2 ,'patch_SketchFix':2,  'patch_SOFix':2,  'patch_TBar':2}


def main(logger, args):
    args.gpt2 = args.gpt2.replace("gpt2-small", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2,)

    model = None

    if args.train_task is None:
        # standard case where the training task and the test task are the same
        train_task = args.task
    else:
        # zero-shot transfer case where the training task is different from the test task
        train_task = args.train_task
        assert args.do_check


    max_length = args.max_length
    batch_size = args.batch_size

    logger.info("%s %s" % (args.method, args.task))

    assert args.method in ["direct", "channel"]

    if args.use_demonstrations:
        assert args.do_zeroshot and not args.do_train

    if args.ensemble:
        assert args.use_demonstrations

    if args.do_train or args.use_demonstrations:
        assert args.train_seed > 0

    # n_templates = 1
    n_templates = args.n_template

    k = int(args.k)
    seed = int(args.seed)

    train_data = load_data_cross_tool(args.data_dir, train_task, k, seed, "train")
    if args.split is None:
        assert args.do_zeroshot
        dev_data = None
    else:
        dev_data = load_data_cross_tool(args.data_dir, args.task, k, seed, args.split)

    accs = []
    # run over different templates
    # for template_idx in range(n_templates):
    template_idx = n_templates
    acc = run(args, logger, args.do_train, args.do_zeroshot,
              args.task, train_task,
              k, seed, args.train_seed,
              args.out_dir, args.split,
              tokenizer, model, train_data, dev_data,
              batch_size, max_length, args.gpt2,
              template_idx, args.method,
              args.lr, args.warmup_steps,
              use_demonstrations=args.use_demonstrations,
              use_calibration=args.use_calibration,
              ensemble=args.ensemble,
              is_null=args.split is None,
              prompt_tune=args.prompt_tune,
              head_tune=args.head_tune,
              transform_tune=args.transform_tune,
              do_check=args.do_check,
              n_prefix=args.n_prefix,
              deepspeed=args.deepspeed)

    accs.append(acc)




def run(args, logger, do_train, do_zeroshot, task, train_task, k, seed,
        train_seed,
        out_dir, split, tokenizer, model,
        train_data, dev_data,
        batch_size, max_length, gpt2, template_idx, method_type,
        learning_rate, warmup_steps,
        use_demonstrations=False,
        use_calibration=False,
        ensemble=False,
        is_null=False,
        prompt_tune=False,
        head_tune=False,
        transform_tune=False,
        do_check=False, n_prefix=20, deepspeed=False):

        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)

        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(train_seed)

        if head_tune or transform_tune:
            assert method_type == "direct"


        n_classes = N_LABELS_DICT.get(task, None)
        templates = get_prompts(task, template_idx)

        n_classes_train = N_LABELS_DICT.get(train_task, None)
        templates_train = get_prompts(train_task, template_idx)


        max_length_per_example = max_length

        if use_demonstrations and not ensemble:
            assert do_zeroshot and not do_train
            mem = batch_size * max_length
            if n_classes == 2:
                max_length = max_length * k
            elif n_classes in [4, 5]:
                max_length = int(max_length * 1.5 * k)
            elif n_classes in [6]:
                max_length = int(max_length * 2 * k)
            else:
                max_length = 2048
            max_length = min(max_length, 2048)
            print('max_length:', max_length)
            batch_size = int(mem / max_length)
            print('batch_size:', batch_size)

        if do_zeroshot:
            cache_paths = [get_paths(args, out_dir, gpt2, method_type, task, do_zeroshot,
                                     k, seed, train_seed, split, template_idx,
                                     use_demonstrations=use_demonstrations,
                                     ensemble=ensemble)]
            checkpoints = [None]

        else:
            out_dir = get_paths(args, out_dir, gpt2, method_type, train_task, do_zeroshot,
                                k, seed, train_seed, split, template_idx,
                                batch_size, learning_rate, warmup_steps,
                                use_demonstrations=use_demonstrations,
                                ensemble=ensemble,
                                prompt_tune=prompt_tune,
                                head_tune=head_tune,
                                transform_tune=transform_tune,
                                n_prefix=n_prefix)

            k = int(k)
            eval_period = 100
            num_training_steps = 400

            cache_paths = [os.path.join(out_dir, "{}cache-{}-{}.pkl".format(
                task + "-" if train_task != task else "",
                split, step))
                           for step in range(eval_period, num_training_steps + eval_period, eval_period)]
            checkpoints = [os.path.join(out_dir, "model-{}.pt".format(step))
                           for step in range(eval_period, num_training_steps + eval_period, eval_period)]

        mapping = None

        if do_train and (head_tune or not do_check):

            inputs = prepare_data(
                tokenizer, None, train_data,
                max_length=max_length,
                max_length_per_example=max_length_per_example,
                n_classes=n_classes_train,
                templates=templates_train,
                method_type=method_type,
                is_training=True,
                ensemble=ensemble)

            logger.info(out_dir)

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            if not do_check:
                # if gpt2.split('/')[0] == 'bigscience' or gpt2.split('-')[0] == 't5':
                #     print('\n  here here model1\n ')
                #     model = AutoModelForSeq2SeqLM.from_pretrained(gpt2,)
                # else:
                #     print('\n  here here model\n ')
                #     # model = GPT2LMHeadModel.from_pretrained(gpt2,)
                #     max_memory_mapping = {0: "18GB", 1: "21GB"}
                #     # model = AutoModelForCausalLM.from_pretrained(gpt2, device_map="auto", load_in_4bit=True)
                #     model = AutoModelForCausalLM.from_pretrained(gpt2, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
                #     )
                if gpt2.split('/')[0] == 'bigscience' or gpt2.split('-')[0] == 't5':
                    model = AutoModelForSeq2SeqLM.from_pretrained(gpt2,)
                else:
                    model = GPT2LMHeadModel.from_pretrained(gpt2,)


                if prompt_tune:
                    for param in model.parameters():
                        param.requires_grad = False

                    set_extra_embeddings(model, n_prefix)
                    inputs = prepend_task_tokens(tokenizer, inputs, n_prefix)

                elif head_tune:
                    mapping, inputs = reassign_output_tokens(inputs, for_labels=True)
                    logger.info("Created mapping with {} vocabs".format(len(mapping)))
                    set_separate_lm_head(model, mapping)
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.lm_head.my_lm_head.parameters():
                        param.requires_grad = True

                elif transform_tune:
                    set_transformed_lm_head(model)
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.lm_head.transform.parameters():
                        param.requires_grad = True

                model = model.cuda()

                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)

                train(logger, model, inputs, batch_size, out_dir,
                      learning_rate=learning_rate,
                      warmup_steps=warmup_steps,
                      eval_period=eval_period,
                      num_training_steps=num_training_steps,
                      prompt_tune=prompt_tune,
                      head_tune=head_tune,
                      transform_tune=transform_tune)

        input_tensors = prepare_data(args, k,
            tokenizer, train_data, dev_data,
            max_length=max_length,
            max_length_per_example=max_length_per_example,
            n_classes=n_classes,
            templates=templates,
            method_type=method_type,
            use_demonstrations=use_demonstrations,
            ensemble=ensemble,
            is_null=is_null)






        results = []
        for cache_path, checkpoint in zip(cache_paths, checkpoints):

            logger.info(cache_path)

            # if there is a cache, load it
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    losses = pkl.load(f)
            else:
                if checkpoint is not None and not os.path.exists(checkpoint):
                    logger.info("checkpoint %s not found..." % checkpoint)
                    assert False

                if checkpoint is None and model is not None and do_zeroshot:
                    logger.info("Reusing the loaded model...")
                    pass
                else:
                    logger.info("Loading the model")
                    torch.cuda.empty_cache()
                    del model
                    model = load_checkpoint(gpt2, checkpoint,
                                            prompt_tune=prompt_tune,
                                            head_tune=head_tune,
                                            transform_tune=transform_tune,
                                            n_prefix=n_prefix,
                                            mapping=mapping, is_deepspeed=deepspeed)
                    # model = model.cuda()
                    model.eval()
                    logger.info("Finished loading the model")

                losses = []
                for input_tensor in input_tensors:
                    losses.append(inference(model,
                                            input_tensor,
                                            batch_size))

                with open(cache_path, "wb") as f:
                    pkl.dump(losses, f)

            if is_null:
                continue

            if ensemble:
                losses = flatten_label_losses(losses, dev_data)

            if use_calibration:
                bias_path = cache_path.replace(split, "None")
                assert os.path.exists(bias_path), bias_path
                with open(bias_path, "rb") as f:
                    bias_losses = pkl.load(f)

                for i, (bias_loss, loss) in enumerate(zip(bias_losses, losses)):
                    loss = np.array(loss)
                    bias_loss = np.array(bias_loss)
                    if ensemble:
                        bias_loss = bias_loss.reshape(1, -1)
                    losses[i] = loss - bias_loss


            acc = evaluate(dev_data, {str(i): loss for i, loss in enumerate(losses)})
            logger.info(acc)
            return acc

def evaluate(dev_data, label_losses):
    dev_data = dev_data
    labels = list(label_losses.keys())
    acc = []
    all_golds, all_scores = [], []
    all_raw_preds = []
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        acc.append(prediction==label)
        all_golds.append(int(label))
        all_scores.append(int(prediction))
        all_raw_preds.append(float(prediction))
    print(all_raw_preds[0:10])
    real_pred = all_scores
    auc_score = round(roc_auc_score(y_true=all_golds, y_score=all_raw_preds), 3)
    accuracy_= accuracy_score(y_true=all_golds, y_pred=real_pred)
    f1 = f1_score(y_true=all_golds, y_pred=real_pred)
    print('label:', all_golds[0:10], Counter(all_golds))
    print('preds:', real_pred[0:10], Counter(real_pred))
    # print("ACC:{}  F1-Score:{} ".format(accuracy_, f1))
    print("ACC:{}  F1-Score:{} AUC:{}".format(accuracy_, f1, auc_score))

    return np.mean(acc)


"""
HP Tuning

-- prompt id                        
-- top k candidate
-- similarity threshold
-- label word
-- max_length

"""

if __name__ == '__main__':
    print('\n  here here \n ')
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--do_check", default=False, action="store_true")
    parser.add_argument("--n_template",type=int, default=0)
    parser.add_argument("--top_k_example", type=int, default=10)
    parser.add_argument("--sim_threshold", type=float, default=0.9)
    parser.add_argument("--enhancement_option", type=str, default="bug-trace-testcase-coverage-similar")

    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--prompt_tune", default=False, action="store_true")
    parser.add_argument("--head_tune", default=False, action="store_true")
    parser.add_argument("--transform_tune", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default="patch_ACS")
    parser.add_argument("--train_task", type=str, default=None)

    parser.add_argument("--k", type=str, default="4")
    parser.add_argument("--seed", type=str, default="100")
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=768)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--n_prefix", type=int, default=20)
    parser.add_argument("--gpt2", type=str, default="bigscience/bloom-1b1")

    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--deepspeed", default=False, action="store_true")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
