from dataloader import build_loader
from model import RNN_ENCODER, CNN_ENCODER
from utils import ensure_dir
from loss import words_loss, sent_loss
import logging, coloredlogs
import argparse
import yaml
import os
import torch
import torch.nn as nn
import json
import torch.optim as optim
import time
import argparse
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true', default=False)

    opt = parser.parse_args()
    return opt

def build_dict():
    with open('captions_train2017.json', 'r') as file:
        anns = json.load(file)
    anns = anns['annotations']
    w2i = {'pad': 0}
    i2w = {0: 'pad'}
    index = 1
    for ann in anns:
        caption = ann['caption']
        if caption[-1] == '.':
            caption = caption[:-1]
        caption = caption.split(' ')
        for c in caption:
            if c not in w2i:
                w2i[c] = index
                i2w[c] = index
                index += 1

    with open('captions_val2017.json', 'r') as file:
        anns = json.load(file)
    anns = anns['annotations']
    for ann in anns:
        caption = ann['caption']
        if caption[-1] == '.':
            caption = caption[:-1]
        caption = caption.split(' ')
        for c in caption:
            if c not in w2i:
                w2i[c] = index
                i2w[c] = index
                index += 1

    return w2i, i2w

def train_epoch(cnn, rnn, dataloader, epoch, device):
    rnn.train()
    cnn.train()
    params = list(rnn.parameters()) + list(cnn.parameters())
    optimizer = optim.Adam(params, lr=0.002, betas=(0.5, 0.999))
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    start_time = time.time()
    for batch_idx, (img, caption, cap_lens) in enumerate(dataloader):
        img = img.to(device)
        caption = caption.to(device)
        cap_lens = cap_lens.to(device)
        optimizer.zero_grad()
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn(img)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
            # words_features = words_features.view(batch_size, nef, -1)
        batch_size = img.size(0)
        hidden = rnn.init_hidden(batch_size)
            # words_emb: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn(caption, cap_lens, hidden)
        batch_size = words_features.size(0)
        labels = torch.arange(batch_size).to(device)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, batch_size)
        w_total_loss0 += w_loss0.item()
        w_total_loss1 += w_loss1.item()
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.item()
        s_total_loss1 += s_loss1.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.25)
        optimizer.step()

    s_cur_loss0 = s_total_loss0 / len(dataloader)
    s_cur_loss1 = s_total_loss1 / len(dataloader)
    w_cur_loss0 = w_total_loss0 / len(dataloader)
    w_cur_loss1 = w_total_loss1 / len(dataloader)

    elapsed = time.time() - start_time
    print('| epoch {:3d} | time {:5.2f} | ''s_loss {:5.2f} {:5.2f} | ''w_loss {:5.2f} {:5.2f}'\
        .format(epoch, elapsed * 1000. / 10,s_cur_loss0, s_cur_loss1, w_cur_loss0, w_cur_loss1))

    if epoch % 10 == 0:
        checkpoint = {
            'cnn_state_dict': cnn.state_dict(),
            'rnn_state_dict': rnn.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        check_path = os.path.join('saved/', 'checkpoint_' + str(epoch) + '.pth')
        torch.save(checkpoint, check_path)

def val_epoch(cnn, rnn, dataloader, epoch, device):
    rnn.eval()
    cnn.eval()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    start_time = time.time()
    correct = 0
    total = len(dataloader.dataset)
    for batch_idx, (img, caption, cap_lens) in enumerate(dataloader):
        img = img.to(device)
        caption = caption.to(device)
        cap_lens = cap_lens.to(device)
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn(img)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
            # words_features = words_features.view(batch_size, nef, -1)
        batch_size = img.size(0)
        hidden = rnn.init_hidden(batch_size)
            # words_emb: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn(caption, cap_lens, hidden)
        batch_size = words_features.size(0)
        labels = torch.arange(batch_size).to(device)
        score = torch.mm(sent_code, sent_emb.transpose(1,0))
        score = score.argmax(1)
        score = torch.sum(score == labels)
        correct += score.item()
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, batch_size)
        w_total_loss0 += w_loss0.item()
        w_total_loss1 += w_loss1.item()

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, batch_size)
        s_total_loss0 += s_loss0.item()
        s_total_loss1 += s_loss1.item()

    s_cur_loss0 = s_total_loss0 / len(dataloader)
    s_cur_loss1 = s_total_loss1 / len(dataloader)
    w_cur_loss0 = w_total_loss0 / len(dataloader)
    w_cur_loss1 = w_total_loss1 / len(dataloader)

    elapsed = time.time() - start_time
    print('| epoch {:3d} | time {:5.2f} | ''s_loss {:5.2f} {:5.2f} | ''w_loss {:5.2f} {:5.2f}'\
        .format(epoch, elapsed * 1000. / 10,s_cur_loss0, s_cur_loss1, w_cur_loss0, w_cur_loss1))
    print('R-precision', (correct/total))

if __name__ == '__main__':
    opt = build_parser()

    w2i, i2w = build_dict()
    
    
    # model setting
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    torch.backends.cudnn.benchmark = False

    rnn = RNN_ENCODER(len(w2i)).to(device)
    cnn = CNN_ENCODER().to(device)

    train_dataloader = build_loader('data/train2017', w2i)
    val_dataloader = build_loader('data/val2017', w2i)

    if not opt.eval_only:
        for epoch in range(60):
            train_epoch(cnn, rnn, train_dataloader, epoch, device)
            val_epoch(cnn, rnn, val_dataloader, epoch, device)

    else:
        rnn.eval()
        cnn.eval()

        checkpoint = torch.load('saved/checkpoint_50.pth', map_location='cuda:0')['cnn_state_dict']
        cnn.load_state_dict(checkpoint)
        checkpoint = torch.load('saved/checkpoint_50.pth', map_location='cuda:0')['rnn_state_dict']
        rnn.load_state_dict(checkpoint)
        correct = 0
        total = len(val_dataloader.dataset)
        for batch_idx, (img, caption, cap_len) in enumerate(val_dataloader):
            img = img.to(device)
            caption = caption.to(device)
            cap_len = cap_len.to(device)
            features, img_feature = cnn(img)
            batch_size = img.size(0)
            hidden = rnn.init_hidden(batch_size)
            words_emb, text_feature = rnn(caption, cap_len, hidden)
            score = torch.mm(img_feature, text_feature.transpose(1,0))
            score = score.argmax(1)
            gt = torch.arange(score.size(0)).to(device)
            score = torch.sum(score == gt)
            correct += score.item()
            
        print(correct/total)