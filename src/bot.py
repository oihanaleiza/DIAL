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

# Dekodetzea
def decode(logits, decoding_strategy='max', k=3, temp=0.4):
  if decoding_strategy=='top1':
    target = logits.max(1)[1]
  elif decoding_strategy=='topk':
    target = logits.topk(k)[1][0][random.randint(0, k-1)].unsqueeze(-1)
  else:
    target = torch.multinomial(logits.squeeze().div(temp).exp().cpu(), 1)
  return target

# Ebaluazioa
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


# Erantzunen garbiketa sinplea
def garbiketa(esaldia):

    esaldia = esaldia.replace(" ' ","'")
    if esaldia.endswith(" ,") or esaldia.endswith(" .") or esaldia.endswith(" '"):
      esaldia=esaldia[:-2]
      esaldia=esaldia+"."
    elif esaldia.endswith(" !"):
      esaldia=esaldia[:-2]
      esaldia=esaldia+"!"
    elif esaldia.endswith(" ?"):
      esaldia=esaldia[:-2]
      esaldia=esaldia+"?"
    
    return esaldia


# Load model and fields
# Hizkuntza bakoitzerako bat, ingeleserako eta euskararako
text_fieldeu = torch.load('../modeleu/text_field.Field')
modeleu = torch.load('../modeleu/model.pt', map_location=torch.device('cpu'))
text_fielden = torch.load('../model/text_field.Field')
modelen = torch.load('../model/model.pt', map_location=torch.device('cpu'))
torch.nn.Module.dump_patches = True
MAX_LENGTH = 10


#Main system loop
modeleu.eval()
modelen.eval()
decoding_strategy = args.decoding_strategy

# Hasierako hizkuntza euskara da
hizkuntza = "euskara"
model = modeleu
text_field = text_fieldeu


TOKEN = "Hemen idatzi norberaren tokena"
bot = telebot.TeleBot(token=TOKEN)

@bot.message_handler(commands=['start']) # Ongi etorri mezua
def send_welcome(message):
    bot.send_message(chat_id=message.chat.id, text="Ongi etorri!")
    reply = "Hizkuntza aukeratzeko:\n /1 Euskara \n /2 Ingelesa \n (Defektuzko hizkuntza euskara da)"
    bot.send_message(chat_id=message.chat.id, text=reply)

@bot.message_handler(commands=['help']) # Laguntza
def send_welcome(message):
    reply = "Hizkuntza aukeratzeko:\n /1 Euskara \n /2 Ingelesa "
    bot.send_message(chat_id=message.chat.id, text=reply)

@bot.message_handler(commands=['1']) # Euskara aukeraketa
def send_welcome(message):
    global hizkuntza, model, text_field, modeleu, text_fieldeu
    model = modeleu
    text_field = text_fieldeu
    bot.reply_to(message, 'Euskara aukeratuta')
    hizkuntza ="euskara"

@bot.message_handler(commands=['2']) # Ingeles aukeraketa
def send_welcome(message):
    global hizkuntza, model, text_field, modelen, text_fielden
    model = modelen
    text_field = text_fielden
    bot.reply_to(message, 'Ingelesa aukeratuta')
    hizkuntza="ingelesa"


@bot.message_handler(func=lambda  msg: msg.text is not None) # Solasaldia
def talk(message):
    global hizkuntza
    sentence = evaluate(' '.join(_basic_english_normalize(message.text)))
    sentence = garbiketa(sentence)
    reply = sentence.strip().capitalize()
    if "<unk>" not in reply:
        bot.send_message(chat_id=message.chat.id, text=reply)
    else:
        bot.send_message(chat_id=message.chat.id, text="Ez dago erantzun egokirik. Gogoratu hizkuntza " + hizkuntza + " dela!")
        bot.send_message(chat_id=message.chat.id, text="Hizkuntza aukeratzeko:\n /1 Euskara \n /2 Ingelesa ")


while True:
    try:
        bot.polling(none_stop=True)
    except Exception:
        time.sleep(15)
