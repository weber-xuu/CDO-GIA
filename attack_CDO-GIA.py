import heapq
import sys, argparse
import os
import datetime
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from datasets import load_metric
from nlp_utils import load_gpt2_from_dict
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    LogitsProcessor, BeamSearchScorer
from init import get_init
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
from utilities import compute_grads, get_closest_tokens, get_reconstruction_loss, get_perplexity, fix_special_tokens, \
    remove_padding
from data_utils import TextDataset
from args_factory import get_args
import time

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from scipy.optimize import linear_sum_assignment
args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

if args.neptune:
    import neptune

    neptune.init(api_token=os.getenv('NEPTUNE_API_KEY'), project_qualified_name=args.neptune)
    neptune.create_experiment(args.neptune_label, params=vars(args))

def pt(args, data_pred, data_refer, data_metric_curr, data_metric_agg, data_loss):
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)

    # 定义一个函数来保存数据，使代码更加整洁
    def save_data(filename, data, fmt='%.18e'):
        path = f'/{filename}_{args.dataset}_{args.batch_size}_{args.swap_every}_{args.lr}_{current_time}_{args.queue_size}_{args.iteration_size}.txt'
        with open(path, 'w', encoding='utf-8') as f:  # 使用 utf-8 编码保存文件
            np.savetxt(f, np.array(data), fmt=fmt)
    # 保存文件，文件名包括参数和当前时间戳
    save_data('data_pred', np.array(data_pred).reshape(len(data_pred), -1), fmt='%s')  # 使用 %s 保存字符串数据
    save_data('data_refer', np.array(data_refer).reshape(len(data_refer), -1), fmt='%s')  # 使用 %s 保存字符串数据
    save_data('data_metric_curr', np.array(data_metric_curr).reshape(len(data_metric_curr), -1),
              fmt='%.3f')  # 保存三位小数的浮点数
    save_data('data_metric_agg', np.array(data_metric_agg).reshape(len(data_metric_agg), -1), fmt='%.3f')
    save_data('data_loss', np.array(data_loss).reshape(len(data_loss), -1), fmt='%.3f')
def get_loss(args, lm, model, ids, x_embeds, true_labels, true_grads, create_graph=False):
    perplexity = lm(input_ids=ids, labels=ids).loss
    rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=create_graph)
    return perplexity, rec_loss, rec_loss + args.coeff_perplexity * perplexity
class MaxHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def add_element(self, elem, seq):
        seq_tuple = tuple(seq)  # Convert seq to a tuple
        if len(self.data) < self.max_size:
            heapq.heappush(self.data, (-elem, seq_tuple))
        elif elem < -self.data[0][0]:  # Only replace if new element is smaller
            heapq.heapreplace(self.data, (-elem, seq_tuple))

    def get_data(self):
        # Store elements in a temporary list and clear the heap
        temp_list = [(-item[0], item[1]) for item in self.data[:]]
        self.data.clear()
        return temp_list


def swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads, counts_swap, tokenizer, tabu_tenure=100):
    tqdm.write('Attempt swap')
    temp_perp, temp_rec, temp_tot_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels, true_grads)
    best_x_embeds, best_tot_loss = x_embeds, temp_tot_loss
    best_ids = cos_ids
    len_heapq, iteration_step = args.queue_size, args.iteration_size

    for sen_id in range(x_embeds.data.shape[0]):
        tabu_list = set()
        change_count = 0
        perm_ids = np.arange(x_embeds.shape[1])

        pq = MaxHeap(len_heapq)
        pq.add_element(temp_tot_loss, perm_ids)
        tot_len = max_len[sen_id]
        while change_count < iteration_step:
            all_ids = pq.get_data()
            change_count += 1
            for _, Ids in all_ids:
                ids = np.array(Ids)
                for i in range(12):
                    # Generate a new candidate move
                    move_type = i%4
                    candidate = generate_candidate(ids, tot_len, move_type)

                    # Check if candidate move is in tabu list
                    if tuple(candidate) in tabu_list:
                        continue  # Skip if the move is tabu

                    # Compute loss for the new configuration
                    new_ids = cos_ids.clone()
                    new_ids[sen_id] = cos_ids[sen_id, candidate]
                    new_x_embeds = x_embeds.clone()
                    new_x_embeds[sen_id] = x_embeds[sen_id, candidate, :]
                    new_perp, new_rec, new_tot_loss = get_loss(args, lm, model, new_ids, new_x_embeds, true_labels,
                                                               true_grads)
                    # Accept the move if it improves the total loss or meets other criteria
                    if new_tot_loss < best_tot_loss:
                        best_tot_loss = new_tot_loss
                        best_x_embeds = new_x_embeds
                        counts_swap[move_type] += 1
                        best_ids = new_ids
                        # Update tabu list
                        tabu_list.add(tuple(candidate))
                        if len(tabu_list) > tabu_tenure:
                            tabu_list.pop(0)
                    pq.add_element(new_tot_loss, candidate)

        all_ids = pq.get_data()
        x_embeds.data = best_x_embeds
        cos_ids[:] = best_ids


def generate_candidate(ids, tot_len, move_type):
    # Implement the logic to generate a new candidate based on move_type
    if move_type == 0 and tot_len >= 4 :  # swap two tokens
        x, y = np.random.choice(np.arange(1, tot_len - 1), size=2, replace=False)
        ids[x], ids[y] = ids[y], ids[x]
    elif move_type == 1 and tot_len >= 4:  # move a token to another place
        x, y = np.random.choice(np.arange(1, tot_len - 1), size=2, replace=False)
        if x < y:
            ids = np.concatenate([ids[:x], ids[x + 1:y], ids[x:x + 1], ids[y:]])
        else:
            ids = np.concatenate([ids[:y], ids[x:x + 1], ids[y:x], ids[x + 1:]])
    elif move_type == 2 and tot_len >= 4:
        b, e = np.random.choice(np.arange(1, tot_len), size=2, replace=False)
        b, e = sorted([b, e])
        p = np.random.choice(np.setdiff1d(np.arange(1, tot_len), np.arange(b, e)), size=1)[0]
        if p < b:
            ids = np.concatenate([ids[:p], ids[b:e], ids[p:b], ids[e:]])
        else:  # p >= e
            ids = np.concatenate([ids[:b], ids[e:p], ids[b:e], ids[p:]])
    elif (move_type == 3) and (tot_len >= 5):  # take some prefix and put it at the end
        indices = np.random.choice(np.arange(1, tot_len - 1), size=3, replace=False)
        x, y, z = sorted(indices)
        ids = np.concatenate([ids[:x], ids[y + 1:z + 1], ids[y:y + 1], ids[x:y], ids[z + 1:]])
    return ids

def reconstruct(args, device, sample, metric, tokenizer, lm, model, counts_swap):
    sequences, true_labels = sample
    data_steps = []
    data_loss = []
    lm_tokenizer = tokenizer

    gpt2_embeddings = lm.get_input_embeddings()
    gpt2_embeddings_weight = gpt2_embeddings.weight.unsqueeze(0)

    bert_embeddings = model.get_input_embeddings()
    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)

    orig_batch = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt').to(device)
    true_embeds = bert_embeddings(orig_batch['input_ids'])
    true_grads = compute_grads(model, true_embeds, true_labels)

    if args.defense_pct_mask is not None:
        for grad in true_grads:
            grad.data = grad.data * (torch.rand(grad.shape).to(device) > args.defense_pct_mask).float()
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape).to(device) * args.defense_noise

    # BERT special tokens (0-999) are never part of the sentence
    unused_tokens = []
    if args.use_embedding:
        for i in range(tokenizer.vocab_size):
            if true_grads[0][i].abs().sum() < 1e-9 and i != BERT_PAD_TOKEN:
                unused_tokens += [i]
    else:
        unused_tokens += list(range(1, 100))
        unused_tokens += list(range(104, 999))
    unused_tokens = np.array(unused_tokens)

    # If length of sentences is known to attacker keep padding fixed
    pads = None
    if args.know_padding:
        pads = [orig_batch['input_ids'].shape[1]] * orig_batch['input_ids'].shape[0]
        for sen_id in range(orig_batch['input_ids'].shape[0]):
            for i in range(orig_batch['input_ids'].shape[1] - 1, 0, -1):
                if orig_batch['input_ids'][sen_id][i] == BERT_PAD_TOKEN:
                    pads[sen_id] = i
                else:
                    break
    print(f'Debug: ids_shape = {orig_batch["input_ids"].shape[1]}, pads = {pads}', flush=True)
    print(f'Debug: input ids = {orig_batch["input_ids"]}', flush=True)
    print(f'Debug: ref = {tokenizer.batch_decode(orig_batch["input_ids"])}', flush=True)

    # Get initial embeddings + set up opt
    x_embeds = get_init(args, model, unused_tokens, true_embeds.shape, true_labels, true_grads, bert_embeddings,
                        bert_embeddings_weight, tokenizer, lm, lm_tokenizer, orig_batch['input_ids'], pads)

    bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)
    if args.opt_alg == 'adam':
        opt = optim.Adam([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bfgs':
        opt = optim.LBFGS([x_embeds], lr=args.lr)
    elif args.opt_alg == 'bert-adam':
        opt = torch.optim.AdamW([x_embeds], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)

    if args.lr_decay_type == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=args.lr_decay)
    elif args.lr_decay_type == 'LambdaLR':
        def lr_lambda(current_step: int):
            return max(0.0, float(args.lr_max_it - current_step) / float(max(1, args.lr_max_it)))

        lr_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print('Nsteps:', args.n_steps, flush=True)

    if pads is None:
        max_len = [x_embeds.shape[1]] * x_embeds.shape[0]
    else:
        max_len = pads

    # Main loop
    best_final_error, best_final_x = None, x_embeds.detach().clone()
    coeff_reg = args.coeff_reg
    tot_steps = args.n_steps
    for it in range(tot_steps):
        t_start = time.time()
        reg = coeff_reg * ((tot_steps - it) / tot_steps)

        def closure():
            opt.zero_grad()
            rec_loss = get_reconstruction_loss(model, x_embeds, true_labels, true_grads, args, create_graph=True)
            reg_loss = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
            tot_loss =   (1-reg)*rec_loss + reg * reg_loss
            tot_loss.backward(retain_graph=True)
            with torch.no_grad():
                if args.grad_clip is not None:
                    grad_norm = x_embeds.grad.norm()
                    if grad_norm > args.grad_clip:
                        x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))
            return tot_loss

        opt.step(closure)

        lr_scheduler.step()

        fix_special_tokens(x_embeds, bert_embeddings.weight, pads)

        _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)

        # Trying swaps
        if args.use_swaps and it >= 200 and it % args.swap_every == 1:
            pre_x_embeds = x_embeds.clone()
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads, counts_swap,tokenizer)
            tqdm.write('prediction: %s' % (tokenizer.batch_decode(cos_ids)))

        _, _, temp_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels, true_grads)
        if best_final_error is None or temp_loss <= best_final_error:
            best_final_error = temp_loss.item()
            best_final_x.data[:] = x_embeds.data[:]
        steps_done = it + 1
        if steps_done % args.print_every == 0:
            _, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight)
            x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(
                cos_ids).norm(dim=2, p=2, keepdim=True)
            _, _, tot_loss_proj = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
            perplexity, rec_loss, tot_loss = get_loss(args, lm, model, cos_ids, x_embeds, true_labels, true_grads)

            step_time = time.time() - t_start

            tqdm.write('[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f [t=%.2fs]' % (
                steps_done, args.n_steps, tot_loss.item(), perplexity.item(), rec_loss.item(), tot_loss_proj.item(),
                step_time))
            tqdm.write('prediction: %s' % (tokenizer.batch_decode(cos_ids)))
            data_loss.append(tot_loss.item())

            tokenizer.batch_decode(cos_ids)

    x_embeds.data = best_final_x
    # Swaps in the end for ablation
    if args.use_swaps_at_end:
        swap_at_end_it = 1
        print('Trying %i swaps' % swap_at_end_it, flush=True)
        for i in range(swap_at_end_it):
            swap_tokens(args, x_embeds, max_len, cos_ids, lm, model, true_labels, true_grads, counts_swap,tokenizer)
    # Postprocess
    fix_special_tokens(x_embeds, bert_embeddings.weight, pads)
    m = 5
    d, cos_ids = get_closest_tokens(x_embeds, unused_tokens, bert_embeddings_weight, metric='cos')
    x_embeds_proj = bert_embeddings(cos_ids) * x_embeds.norm(dim=2, p=2, keepdim=True) / bert_embeddings(cos_ids).norm(
        dim=2, p=2, keepdim=True)
    _, _, best_tot_loss = get_loss(args, lm, model, cos_ids, x_embeds_proj, true_labels, true_grads)
    best_ids = cos_ids
    best_x_embeds_proj = x_embeds_proj

    prediction, reference = [], []
    for i in range(best_ids.shape[0]):
        prediction += [remove_padding(tokenizer, best_ids[i])]
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i])]

    # Matching
    cost = np.zeros((x_embeds.shape[0], x_embeds.shape[0]))
    for i in range(x_embeds.shape[0]):
        for j in range(x_embeds.shape[0]):
            fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
            cost[i, j] = 1.0 - fm
    row_ind, col_ind = linear_sum_assignment(cost)

    ids = list(range(x_embeds.shape[0]))
    ids.sort(key=lambda i: col_ind[i])
    new_prediction = []
    for i in range(x_embeds.shape[0]):
        new_prediction += [prediction[ids[i]]]
    prediction = new_prediction

    return prediction, reference, data_loss


def print_metrics(res, suffix, use_neptune):
    data_metric = []
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        curr = res[metric].mid
        print(
            f'{metric:10} | fm: {curr.fmeasure * 100:.3f} | p: {curr.precision * 100:.3f} | r: {curr.recall * 100:.3f}',
            flush=True)
        data_metric.append([(curr.fmeasure * 100), (curr.precision * 100), (curr.recall * 100)])
        if use_neptune:
            neptune.log_metric(f'{metric}-fm_{suffix}', curr.fmeasure * 100)
            neptune.log_metric(f'{metric}-p_{suffix}', curr.precision * 100)
            neptune.log_metric(f'{metric}-r_{suffix}', curr.recall * 100)
    sum_12_fm = res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure
    if use_neptune:
        neptune.log_metric(f'r1fm+r2fm_{suffix}', sum_12_fm * 100)
    print(f'r1fm+r2fm = {sum_12_fm * 100:.3f}\n', flush=True)
    return data_metric

def main():
    print('\n\n\nCommand:', ' '.join(sys.argv), '\n\n\n', flush=True)

    device = torch.device(args.device)
    metric = load_metric('rouge')
    data_loss = []
    data_metric_curr = []
    data_metric_agg = []
    data_pred = []
    data_refer = []
    counts_swap = np.zeros(5)
    dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size)

    lm = load_gpt2_from_dict("transformer_wikitext-103.pth", device, output_hidden_states=True).to(device)
    lm.eval()

    model = AutoModelForSequenceClassification.from_pretrained(args.bert_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()
    for i in range(0, args.n_inputs):
        t_input_start = time.time()
        sample = dataset[i]  # (seqs, labels)

        print(f'Running input #{i} of {args.n_inputs}.')
        if args.neptune:
            neptune.log_metric('curr_input', i)

        print('reference: ')
        for seq in sample[0]:
            print('========================')
            print(seq)

        print('========================', flush=True)

        prediction, reference, temp_loss = reconstruct(args, device, sample, metric, tokenizer, lm, model, counts_swap)
        predictions += prediction
        references += reference
        data_loss.append(temp_loss)

        print(f'Done with input #{i} of {args.n_inputs}.')
        print('reference: ')
        for seq in reference:
            print('========================')
            print(seq)
        print('========================')

        print('predicted: ')
        for seq in prediction:
            print('========================')
            print(seq)
        print('========================', flush=True)
        data_refer.append(reference)
        data_pred.append(prediction)
        print('[Curr input metrics]:')
        res = metric.compute(predictions=prediction, references=reference)
        temp_data_metric = print_metrics(res, suffix='curr', use_neptune=args.neptune is not None)
        data_metric_curr.append(temp_data_metric)

        print('[Aggregate metrics]:')
        res = metric.compute(predictions=predictions, references=references)
        temp_data_metric = print_metrics(res, suffix='agg', use_neptune=args.neptune is not None)
        data_metric_agg.append(temp_data_metric)

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}\n\n', flush=True)
        if i%10 == 9:
            pt(args, data_pred, data_refer, data_metric_curr, data_metric_agg, data_loss)
    pt(args, data_pred, data_refer, data_metric_curr, data_metric_agg, data_loss)
    print('Done with all.', flush=True)
    print(counts_swap)
    if args.neptune:
        neptune.log_metric('curr_input', args.n_inputs)


if __name__ == '__main__':
    main()
