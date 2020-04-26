import sys
import codecs
import os
import io
import math
import operator
from functools import reduce

import torch
from torch.autograd import Variable

from Dataloader import DataLoader_token, DataLoader_token_kg
from Train import beam_search_decode, beam_search_decode_kg
from HyperParameter import chunk_len,bleu_model_name, batch, nbatches, transformer_size, epoch_number, epoches_of_loss_record, \
	predict_length

import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    references = []
    candidates = []

    with open('EN-ATP-V226.txt', encoding='UTF8') as f:
        reference = f.readline()
        k = 0
        while (reference is not None) and (k < 100):
            print("k",k)
            len_r = len(reference.strip().split())
            print(len_r)
            _ = reference.strip().split()[0:12]
            reference = ""
            for i in range(len(_)):
                reference = reference + _[i] + " "
            print("reference", reference)
            references.append(reference)
            # context = reference[0:20]
            _ = reference.strip().split()[0:10]
            context = ""
            for i in range(len(_)):
                context = context + _[i] + " "

            print("context:", context)
            len_c = len(context.strip().split())
            print("c", len_c)

            filename = 'EN-ATP-V226.txt'
            trained_model_name = bleu_model_name
            words = context

            ents = []
            with open("kg_embed/entity2id.txt") as fin:
                fin.readline()
                for line in fin:
                    name, id = line.strip().split("\t")
                    ents.append(name)

            model = torch.load(trained_model_name).cuda()
            model.eval()
            dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
            word_list = words.replace('\n', ' ').replace('\t', ' ').split(' ')
            word_list = [i for i in word_list if (len(str(i))) != 0]
            src = Variable(dataloader.vocabularyLoader.token_tensor(word_list).unsqueeze(0))
            src_mask = Variable((src != 0).unsqueeze(-2))
            ent = Variable(torch.Tensor([24] * len(word_list)).long()).to(device)
            ents_list = []
            for i in range(len(dataloader.kg)):
                if words.find(" " + dataloader.kg[i] + " ") != -1:
                    ents_list.append(dataloader.kg[i])
            for i in range(len(ents_list)):
                key = ents_list[i].strip().split()
                if word_list.index(key[0]) >= 0:
                    ent[word_list.index(key[0])] = dataloader.kg.index(" ".join(key))
            ent = ent.unsqueeze(0)

            ent_mask = None

            output_embed_list = beam_search_decode_kg(model, src, src_mask, ent, ent_mask, max_len=2 if 2 < (len_r-len_c) else len_r-len_c)
            for j in range(len(output_embed_list)):
                output_embed = output_embed_list[j][1][0].cpu().numpy()
                result = []
                for i in output_embed:
                    result.append(dataloader.vocabularyLoader.index2token[i])
                result = result[1:]
                result = " ".join(result)
                print(result)
            candidate = context + " " + result
            print("candidate: ", candidate)
            candidates.append(candidate)
            reference = f.readline()
            k = k + 1

    with open('candidate.txt', 'w') as f:
        for candidate in candidates:
            print("c",candidate)
            f.write(candidate)

    with open('ref.txt', 'w') as f:
        for reference in references:
            print("r",reference)
            f.write(reference)

    sys.argv.append('candidate.txt')
    sys.argv.append('ref.txt')
    candidate, references = fetch_data(sys.argv[1], sys.argv[2])
    bleu = BLEU(candidate, references)
    print(bleu)
    out = open('bleu_out.txt', 'w')
    out.write(str(bleu))
    out.close()