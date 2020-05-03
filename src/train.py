from torchtext.data import Field, BucketIterator, TabularDataset
import torch
from torchtext import data
from model import Seq2Seq, Encoder, Decoder, Attention
import math
import time
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

text_field = Field(init_token = '<sos>',
                   eos_token = '<eos>',lower=True,
                   tokenize='basic_english', tokenizer_language='en')

fields = [('query', text_field), ('answer', text_field)]

train_data = TabularDataset(path='../data/en_train.tsv', format='tsv', fields=fields)

text_field.build_vocab(train_data, min_freq=5)
print("Vocabulary has been built")
print("Vocab len is {}".format(len(text_field.vocab)))

#Save the text field for testing
torch.save(text_field, '.model/text_field.Field')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

train_iterator  = BucketIterator(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: data.interleave_keys(len(x.query), len(x.answer)),
    device=device)

#Tamainak egokitu zuen beharretara
INPUT_DIM = len(text_field.vocab)
OUTPUT_DIM = len(text_field.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_IDX = text_field.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in tqdm(enumerate(iterator),total=len(iterator)):

        src = batch.query
        trg = batch.answer

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)



def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    # Save checkpoint
    torch.save(model, '../model/model.pt')

