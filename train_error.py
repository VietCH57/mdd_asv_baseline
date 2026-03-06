
import argparse
import torch, librosa, gc
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
from dataloader import MDD_Dataset, collate_fn
from dataloader import text_to_tensor
from MDD_model import Wav2Vec2_Error
from config import DATA_ROOT, LABEL_ROOT, WAV_SUFFIX, CHECKPOINT_DIR
from pyctcdecode import build_ctcdecoder
from jiwer import wer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--eval_start_epoch', type=int, default=7)
args = parser.parse_args()

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = args.epochs

gc.collect()

df_train = pd.read_csv(LABEL_ROOT + 'train_time.csv')
df_dev = pd.read_csv(LABEL_ROOT + 'dev.csv')
train_dataset = MDD_Dataset(df_train)

batch_size = args.batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Wav2Vec2_Error.from_pretrained(
    'facebook/wav2vec2-xls-r-300m', 
)

model.freeze_feature_extractor()
model = model.to(device)

#need fix here
list_vocab = ['t ', 'n* ', 'y* ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ey* ', 'd* ', 'jh* ', 'ow ', 'aw ', 'ao* ', 'aa ', 'z* ', 'dh* ', 'aa* ', 'uw* ', 'th ', 'er* ', 'ih ', 't* ', 'zh ', 'g* ', 'k ', 'y ', 'l ', 'uh ', 'eh* ', 'p* ', 'ow* ', 'ch ', 'w ', 'b ', 'l* ', 'v ', 'ao ', 'w* ', 'aw* ', 'ah* ', 'uh* ', 'zh* ', 's ', 'k* ', 'p ', 'iy ', 'r ', 'ae* ', 'eh ', 'b* ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'hh* ', 'dh ', 'ae ', 'v* ', 'r* ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
nll_loss = nn.NLLLoss(ignore_index = 2)
ctc_loss = nn.CTCLoss(blank = 68)
for epoch in range(num_epoch):
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, linguistic, labels, error_gt, target_lengths  = data
    # print(error_gt)
    output = labels
    transcript = labels
    logits, error_classifier = model(acoustic, linguistic)
    logits = logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    error_classifier    = F.log_softmax(error_classifier, dim = 2)

    loss_nll = nll_loss(error_classifier.reshape(-1, 2), error_gt.reshape(-1))
    loss_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    loss = 0.5*loss_nll + 0.5*loss_ctc
    if i%500==0:
      print(loss)
    running_loss.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # break

  print(f"Training loss: {sum(running_loss) / len(running_loss)}")
  #after 5-7 epoch, model converge
  if epoch >= args.eval_start_epoch:
    with torch.no_grad():
      model.eval().to(device)
      worderrorrate = []
      for point in tqdm(range(len(df_dev))):
        acoustic, _ = librosa.load(DATA_ROOT + df_dev['Path'][point] + WAV_SUFFIX, sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]
        canonical = df_dev['Canonical'][point]
        canonical = text_to_tensor(canonical)
        canonical = torch.tensor(canonical, dtype=torch.long, device=device)
        logits, _ = model(acoustic, canonical.unsqueeze(0))
        logits = F.log_softmax(logits.squeeze(0), dim=1)
        x = logits.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)

      epoch_wer = sum(worderrorrate)/len(worderrorrate)
      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        torch.save(model.state_dict(), CHECKPOINT_DIR + '/error_checkpoint_55.pth')
      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
      