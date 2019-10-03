from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
import torch.utils.data.dataloader
import torch.optim as optim

import xml.etree.ElementTree as ET
import logging
import argparse
import pdb
from tqdm import tqdm
import pickle
import os
import random
import numpy
import time
import math


parser = argparse.ArgumentParser()
# hyperparameters
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_workers", type=int, default=8, help="number of cpu used for multi-procssing")

parser.add_argument('--train', action='store_true', help='whether train')
parser.add_argument('--validation', action='store_true', help='whether validation')
parser.add_argument('--roc', action='store_true', help='whether train')
parser.add_argument('--keeptrain', action='store_true', help='whether keep training')
parser.add_argument('--resave_frame2idx', action='store_true', help='whether resave frame2idx')
parser.add_argument('--traindata_dir', type=str, default='./logs/frameid/parsed_frame_train.txt', help='path of training data')
parser.add_argument('--testdata_dir', type=str, default='./logs/frameid/parsed_frame_test.txt', help='path of testing data')
parser.add_argument('--save_model_dir', type=str, default='transformer_checkpoint/', help='output directory of model')
parser.add_argument('--load_model_dir', type=str, default='transformer_checkpoint/model.pkl.19', help= 'directory of LM model to load')
parser.add_argument('--predict_answer_log', type=str, default='transformer_checkpoint/predict_answer_log.txt', help= 'directory of log that record pair of predicted/answer')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class TermDataset(Dataset):
    def __init__(self, data, frame_to_idx):
        self.data = data
        self.dict = frame_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):      
        tuple_data = self.data[index]
        return tuple_data
        
    # ([prev_frameset, next_frameset], target_frameset) 
    def collate_fn(self, datas):
        maxm = 0
        for context, target in datas:
            maxm = max(maxm, max(len(context[0]), len(context[1]), len(target)))
        self.max_length = min(50, maxm)
        
        batch = []
        for context, target in datas:
            pre = context[0] # is a list of frame
            nex = context[1] # is a list of frame
            pre_context = torch.tensor(self.pad([self.dict[p] for p in pre], self.max_length, 0), dtype=torch.long)
            next_context = torch.tensor(self.pad([self.dict[n] for n in nex], self.max_length, 0), dtype=torch.long)
            target_tensor = torch.tensor(self.pad([self.dict[t] for t in target], self.max_length, 0), dtype=torch.long)
            # return batch of tuple which contains tensor of context and target in index 
            batch.append( ([pre_context, next_context], target_tensor) )
        return batch
    
    def pad(self, target, max_length, padding):

        padded_arr = target

        if len(target) < max_length:
            for i in range(max_length - len(target)):
                padded_arr.append(padding)
        else:
            padded_arr = padded_arr[:max_length]

        return padded_arr
    
#############################################  Model   ##############################################
## using NN.TRANSFORMER
"""
    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
"""
class TransformerModel(nn.Module):

    def __init__(self, ntoken, simple_ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead) # we cat two tensor so need to *2
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken+2, ninp) # since we have +2 in frame2idx 
        self.ninp = ninp # embedding dimension
        
        decoder_layers = nn.TransformerDecoderLayer(ninp, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.proj = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, pre, nex, target, mode):
        if self.src_mask is None or self.src_mask.size(0) != (pre.shape[1]):
            device = pre.device
            mask = self._generate_square_subsequent_mask(pre.shape[1]).to(device)
            self.src_mask = mask # [seq_len, seq_len]

        """
         a square attention mask is required because the self-attention layers in nn.TransformerEncoder are only allowed to 
         attend the earlier positions in the sequence
        """
        prev_emb = self.encoder(pre) * math.sqrt(self.ninp)
        nex_emb = self.encoder(nex) * math.sqrt(self.ninp)
        target_emb = self.encoder(target) * math.sqrt(self.ninp) # [batch, seq_len, emb_dim]
        prev_pos_emb = self.pos_encoder(prev_emb) # [batch, seq_len, emb_dim]
        nex_pos_emb = self.pos_encoder(nex_emb) # [batch, seq_len, emb_dim]
        
        src = prev_pos_emb + nex_pos_emb # [batch, seq_len, emb_dim]
#         src = torch.cat((prev_pos_emb, nex_pos_emb), dim=2) # [batch, seq_len, emb_dim*2]
        src = src.permute(1, 0, 2) # [seq_len, batch, emb_dim*2]
        output = self.transformer_encoder(src, self.src_mask) # output: [seq_len, batch, emb_dim*2] 
#         output = self.decoder(output)
        output = output.permute(1, 0, 2)
#         pdb.set_trace()
        # output: [batch, seq_len, emb_dim*2], target_emb:[batch, seq_len, emb_dim] 
#         output = self.transformer_decoder(target_emb, output) 
        if mode == 'train':
            output = self.transformer_decoder(target_emb, output) 
            output = self.proj(output)
            output = F.log_softmax(output, dim=-1)
            return output
        elif mode == 'valid':
            seq_len = output.shape[1]
            print("seq_len: " + str(seq_len))
            while(seq_len != 0):
                print("target_emb.shape: " + str(target_emb.shape))
                nex_word = self.transformer_decoder(target_emb, output) 
                pdb.set_trace()
                print("nex_word.shape: " + str(nex_word.shape))
#                 target_emb = torch.cat((target_emb, nex_word), dim=1)
#                 pdb.set_trace()
                target_emb = torch.cat((target_emb, nex_word[:,-1,:].unsqueeze(1)), dim=1)
                seq_len -= 1
            target_emb = self.proj(target_emb)
            output = F.log_softmax(target_emb, dim=-1)
            return output
                
#         output = self.proj(output)
#         output = F.log_softmax(output, dim=-1)
#         return output
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
############################################  End of Model   ###########################################

def save(model, path):
    torch.save({
        'model': model.state_dict()
    }, path)
    
def load(path, frame_set, max_length):
    ntokens = 0 # the size of vocabulary
    simple_ntokens = 0 # the size of verb frame
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(path)['model'])
    return model


###### Preparing Training data #########

f = open(args.traindata_dir, "r")
train_data = f.readlines()
train_cbow = [] # save the training data, format is a tuple -> ( [frame[i-1], frame[i+1]], frame[i] )    # tuple[1] is target frameset

# Generate training data, three sentence a time
"""
    *** Simpler traing task: Use 1st,3rd Verb&Noun to predict 2nd Verb only
"""
# predicted_frame_set = set()
# for i in range(len(train_data)):
#     tmp = train_data[i].split()
#     predicted_frame_set.update(tmp)

# all_frame2idx =  {frame: (i+2) for i, frame in enumerate(predicted_frame_set)}
# all_idx2frame = {all_frame2idx[frame] : frame for frame in all_frame2idx}
# print("all_frame2idx size = " + str(len(all_frame2idx)))

# path = os.path.join(args.save_model_dir, 'all_frame2idx.pkl')
# f = open(path, "rb")
if args.roc:
    f = open('transformer_checkpoint_XL_2/ROC_all_frame2idx.pkl', "rb")
    all_frame2idx = pickle.load(f)
    # path = os.path.join(args.save_model_dir, 'all_idx2frame.pkl')
    # f = open(path, "rb")
    f = open('transformer_checkpoint_XL_2/ROC_all_idx2frame.pkl', "rb")
    all_idx2frame = pickle.load(f)
    print("all_frame2idx size = " + str(len(all_frame2idx))) ## 78089 
else:
    f = open('transformer_checkpoint_XL/all_frame2idx.pkl', "rb")
    all_frame2idx = pickle.load(f)
    # path = os.path.join(args.save_model_dir, 'all_idx2frame.pkl')
    # f = open(path, "rb")
    f = open('transformer_checkpoint_XL/all_idx2frame.pkl', "rb")
    all_idx2frame = pickle.load(f)
    print("all_frame2idx size = " + str(len(all_frame2idx))) ## 78089 

for i in range(1, len(train_data)-1):
    if(train_data[i-1] != "\n" and (train_data[i] != "\n" and train_data[i+1] != "\n")):
        prev_frameset = train_data[i-1].split()
        target_frameset_tmp = train_data[i].split() # To simplify the task, here I will use Verb only 
        next_frameset = train_data[i+1].split()

        target_frameset = []
        for frame in target_frameset_tmp:
            tmp = frame.split('_')
            if tmp[-1] == 'Frame': # be a verb, and need to be added into target 
                target_frameset.append(frame)
        if len(target_frameset) > 0: # a training data is added when there is at least one verb frame in target_frameset 
            train_cbow.append( ([prev_frameset, next_frameset], target_frameset) )

print(train_cbow[:3])
print("length of training data: "+ str(len(train_cbow))) ## 215737

# max_length = 0 # find the longest sentence's length
# for i in range(len(train_data)):
#     if len(train_data[i]) > max_length:
#         max_length = len(train_data[i])

def read_frameList(path):
    # use ElementTree to parse XML file 
    tree = ET.parse(path)
    root = tree.getroot()
    
    frame_list = []
    for element in root:
        frame_list.append(element.attrib['name'])
    
    return frame_list

# the simpler projection dimension == verb number
frame_list = read_frameList('./data/fndata-1.7/frameIndex.xml') 
verbframe_to_idx = {frame: (i+2) for i, frame in enumerate(frame_list)}

print("VerbFrame_to_idx size = " + str(len(verbframe_to_idx))) # 
path = os.path.join(args.save_model_dir, 'verbframe_to_idx.pkl')
f = open(path, "wb")
pickle.dump(verbframe_to_idx, f)



train_dataset = TermDataset(train_cbow, all_frame2idx)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, shuffle=False)

########## End of Preparing training data #########


########## Preparing Validation data #############
f = open(args.testdata_dir, "r")
val_data = f.readlines()
val_cbow = [] # save the training data, format is a tuple -> ( [frame[i-1], frame[i+1]], frame[i] )    # tuple[1] is target frameset

# Generate training data, three sentence a time
"""
    *** Simpler traing task: Use 1st,3rd Verb&Noun to predict 2nd Verb only
"""
# predicted_frame_set = set()
# for i in range(len(val_data)):
#     tmp = val_data[i].split()
#     predicted_frame_set.update(tmp)

# all_frame2idx =  {frame: (i+2) for i, frame in enumerate(predicted_frame_set)}
# all_idx2frame = {all_frame2idx[frame] : frame for frame in all_frame2idx}

for i in range(1, len(val_data)-1):
    if(val_data[i-1] != "\n" and (val_data[i] != "\n" and val_data[i+1] != "\n")):
        prev_frameset = val_data[i-1].split()
        target_frameset_tmp = val_data[i].split() # To simplify the task, here I will use Verb only 
        next_frameset = val_data[i+1].split()

        target_frameset = []
        for frame in target_frameset_tmp:
            tmp = frame.split('_')
            if tmp[-1] == 'Frame': # be a verb, and need to be added into target 
                target_frameset.append(frame)
        if len(target_frameset) > 0: # a training data is added when there is at least one verb frame in target_frameset 
            val_cbow.append( ([prev_frameset, next_frameset], target_frameset) )

print("Validation data:")
print(val_cbow[:3])

val_dataset = TermDataset(val_cbow, all_frame2idx)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate_fn, shuffle=False)

########## End of Preparing Validation data #########

ntokens = len(all_frame2idx) # the size of verb&noun frame of training dataset
simple_ntokens = len(verbframe_to_idx) # the size of verb frame
emsize = 800 # embedding dimension
nhid = 800 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens , simple_ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss(ignore_index=0, size_average=True)
lr = 1 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.9)

def train(epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(all_frame2idx)
    total_cnt = 0
    correct_cnt = 0
    
    for i, batch in enumerate(train_dataloader):
            prev_frame = []
            nex_frame = []
            target = []
            for tupple in batch:
                target.append(tupple[1])
                context = tupple[0]
                prev_frame.append(context[0])
                nex_frame.append(context[1])
                
            target_tensor = torch.stack(target).to(device) # [batch_size, max_sentence_length_of_batch]
            prev_frame_tensor = torch.stack(prev_frame).to(device)
            nex_frame_tensor = torch.stack(nex_frame).to(device)
#             seq2seq.zero_grad()
            optimizer.zero_grad()
            target_tensor_to_train = []
            for g in range(len(batch)):
                target_tensor_to_train.append(torch.tensor([1])) ## [1] be the start of the sentence
            target_tensor_to_train = torch.stack(target_tensor_to_train).to(device)
            target_tensor_to_train = torch.cat((target_tensor_to_train, target_tensor[:, :-1]), dim=1)
            output = model(prev_frame_tensor, nex_frame_tensor, target_tensor_to_train, 'train') # [batch, seq_len, 40794] 
            maxes, indices = torch.max(output, 2)
            target = target_tensor.view(-1)
            
            loss = criterion(output.view(-1, ntokens), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            log_interval = 200
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                log_path = os.path.join(args.save_model_dir, 'log.txt')
                with open(log_path, 'a') as f:
                    f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, len(train_dataloader), scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                    f.write('\n')
                print(('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, len(train_dataloader), scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss))))
                total_loss = 0
                start_time = time.time()
                
                # Accuracy of the batch
                for l in range(target_tensor.shape[0]-1):
                    # l is each sentnece
                    haveZero = False
                    sentence = target_tensor[l]
#                     predicted = maxes[1][l]
                    predicted = indices[l]
                    for e in range(len(sentence)):
                        if sentence[e] != 0:
                            if predicted[e] == sentence[e]:
                                correct_cnt += 1
                        else:
                            total_cnt += e
                            haveZero = True
                            break
                    if not haveZero:
                        total_cnt += len(sentence)
    # Accuracy
    return float(correct_cnt) / float(total_cnt)
    
    

def evaluate(eval_model, val_dataloader, write_log=False):
    eval_model.eval() # Turn on the evaluation mode
    
    total_loss = 0.
    start_time = time.time()
    ntokens = len(all_frame2idx)
    total_cnt = 0
    correct_cnt = 0
    
    log_path = os.path.join(args.save_model_dir, 'predict_answer_log.txt')
    predict_answer_log = open(log_path, "a")
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
                prev_frame = []
                nex_frame = []
                target = []
                for tupple in batch:
                    target.append(tupple[1])
                    context = tupple[0]
                    prev_frame.append(context[0])
                    nex_frame.append(context[1])

                target_tensor = torch.stack(target).to(device) # [batch_size, max_sentence_length_of_batch]
                prev_frame_tensor = torch.stack(prev_frame).to(device)
                nex_frame_tensor = torch.stack(nex_frame).to(device)
                target_tensor_to_test = []
                for g in range(len(batch)):
                    target_tensor_to_test.append(torch.tensor([1])) ## [1] be the start of the sentence
                target_tensor_to_test = torch.stack(target_tensor_to_test).to(device) # [batch, 1]
                output = eval_model(prev_frame_tensor, nex_frame_tensor, target_tensor_to_test, 'valid') # [batch, seq_len, 40794] 
                maxes, indices = torch.max(output, 2)
                
                target = target_tensor.view(-1) # batch * seq_len
#                 pdb.set_trace()
                loss = criterion(output.view(-1, ntokens), target)
                total_loss += loss.item()
                
                # Accuracy of the batch
                for l in range(target_tensor.shape[0]-1):
                    # l is each sentnece
                    haveZero = False
                    sentence = target_tensor[l]
#                     predicted = maxes[1][l]
                    predicted = indices[l]
                    for e in range(len(sentence)):
                        if sentence[e] != 0:
                            if predicted[e] == sentence[e]:
                                correct_cnt += 1
                        else:
                            total_cnt += e
                            haveZero = True
                            break
                    if not haveZero:
                        total_cnt += len(sentence)
                if write_log:
                    ex = indices[0] # [max_seq_len, 1]
                    tar = target_tensor[0]
                    ex = ex.tolist()
                    tar = tar.tolist()
                    predict_answer_log.write("=====================================================================\n")
                    predict_answer_log.write("Predicted:\n")
                    for i, j in enumerate(ex):
                        if j == 0:
                            pass
            #                     predict_answer_log.write("0 - ")
                        else:
                            if tar[i] == j:
                                for g in ex:
                                    if g == 0:
                                        pass
                                    else:
                                        predict_answer_log.write(str(all_idx2frame[g]) + " - ")
                                break
                    predict_answer_log.write("\nAnswer:\n")
                    for j in tar:
                        if j == 0:
                            pass
                        else:
                            predict_answer_log.write(str(all_idx2frame[j]) + " - ")
                    predict_answer_log.write("\n")
                        
    # Accuracy
    return float(correct_cnt) / float(total_cnt), total_loss / len(val_dataloader)
    
#     total_loss = 0.
#     ntokens = len(TEXT.vocab.stoi)
#     with torch.no_grad():
#         for i in range(0, data_source.size(0) - 1, bptt):
#             data, targets = get_batch(data_source, i)
#             output = eval_model(data)
#             output_flat = output.view(-1, ntokens)
#             total_loss += len(data) * criterion(output_flat, targets).item()
#     return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
best_model = None

for epoch in range(1, args.n_epochs + 1):
    epoch_start_time = time.time()
    accuracy = train(epoch)
    if epoch == 7:
        val_acc, val_loss = evaluate(model, val_dataloader, True)
    else:
        val_acc, val_loss = evaluate(model, val_dataloader, False)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid accuracy {:5.3f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, val_acc, math.exp(val_loss)))
    print('-' * 89)
    log_path = os.path.join(args.save_model_dir, 'log.txt')
    with open(log_path, 'a') as f:
        f.write("Validation Accuracy of epoch " + str(epoch) + " : " + str(val_acc))

    if val_loss < best_val_loss:
        best_model_acc = val_acc
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
    
print('-' * 50)
print("Best val loss's accuracy: " + str(best_model_acc))
print('-' * 50)
model_path = os.path.join(args.save_model_dir, 'model.pkl.{}'.format(epoch + cont_epoch))
save(best_model, model_path)
    
    



