from torchtext.data.utils import _basic_english_normalize
import torch
import random
from argparse import ArgumentParser
from model import *

import time
import telebot

# Testa prestatu

parser = ArgumentParser(description='Test the Chit Chat system')

parser.add_argument('-decoding_strategy', type=str, default='top1', choices=['top1', 'topk', 'multinomial']) 

args = parser.parse_args()

def decode(logits, decoding_strategy='max', k=3, temp=0.4):
  if decoding_strategy=='top1':
    target = logits.max(1)[1]
  elif decoding_strategy=='topk':
    target = logits.topk(k)[1][0][random.randint(0, k-1)].unsqueeze(-1)
  else:
    target = torch.multinomial(logits.squeeze().div(temp).exp().cpu(), 1)
  return target

def evaluate(sentence):
  with torch.no_grad():
    sentence = '<sos> ' + sentence + ' <eos>'
    sent_len = len(sentence.split())
    sentence = torch.Tensor([text_field.vocab.stoi[i] for i in sentence.lower().split()]).long().view(sent_len, 1)
    target = torch.Tensor([text_field.vocab.stoi['<sos>']]).long()
    output_sentence = ''
    encoder_outputs, hidden = model.encoder(sentence)
    for t in range(MAX_LENGTH):
    # first input to the decoder is the <sos> token
      output, hidden = model.decoder(target, hidden, encoder_outputs)
      target = decode(output, decoding_strategy)
      word = text_field.vocab.itos[target.numpy()[0]]
      if word == '<eos>':
        return output_sentence
      else:
        output_sentence = output_sentence + ' ' + word
  return output_sentence

#Load model and fields
text_field = torch.load('../modeleu/text_field.Field')
model = torch.load('../modeleu/model.pt', map_location=torch.device('cpu'))
torch.nn.Module.dump_patches = True
MAX_LENGTH = 10

#Main system loop
model.eval()
decoding_strategy = args.decoding_strategy


TOKEN = "1082764579:AAGTX1fWL9g5PwRye-geOK7hc3HiOv8tHq0"
bot = telebot.TeleBot(token=TOKEN)

@bot.message_handler(commands=['start']) # Ongi etorri mezua
def send_welcome(message):
    bot.reply_to(message, 'Ongi etorri!')

@bot.message_handler(commands=['help']) # Laguntza
def send_welcome(message):
    bot.reply_to(message, 'ALPHA = FEATURES MAY NOT WORK')

@bot.message_handler(func=lambda msg: msg.text is not None)
def echo(message):
    sentence = evaluate(' '.join(_basic_english_normalize(message.text)))
    print('-' + sentence.strip().capitalize())
    reply = sentence.strip().capitalize()
    bot.send_message(chat_id=message.chat.id, text=reply)

while True:
    try:
        bot.polling(none_stop=True)
    except Exception:
        time.sleep(15)
