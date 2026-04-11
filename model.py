import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

IMG_SIZE    = 224
MAX_SEQ_LEN = 35

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ConvBlock(32, 64)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.layer4 = ConvBlock(256, 512, stride=2)
        self.proj   = nn.Conv2d(512, hidden_dim, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H*W, C)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, h):
        energy = self.v(torch.tanh(self.W_enc(enc_out) + self.W_dec(h).unsqueeze(1)))
        alpha  = F.softmax(energy.squeeze(-1), dim=-1)
        return (alpha.unsqueeze(-1) * enc_out).sum(1), alpha


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_dim, attn_dim):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(enc_dim, hidden_dim, attn_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + enc_dim, hidden_dim)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, vocab_size)
        self.init_h    = nn.Linear(enc_dim, hidden_dim)
        self.init_c    = nn.Linear(enc_dim, hidden_dim)

    def init_hidden(self, enc_out):
        m = enc_out.mean(1)
        return torch.tanh(self.init_h(m)), torch.tanh(self.init_c(m))

    def generate(self, enc_out, word2idx, idx2word, max_len=MAX_SEQ_LEN):
        h, c  = self.init_hidden(enc_out)
        token = torch.tensor([word2idx['<SOS>']], device=enc_out.device)
        words, alphas = [], []
        for _ in range(max_len):
            emb = self.embed(token.unsqueeze(0)).squeeze(0)
            ctx, alpha = self.attention(enc_out, h)
            h, c = self.lstm_cell(torch.cat([emb, ctx], dim=1), (h, c))
            idx  = self.fc(h).argmax(dim=-1).item()
            alphas.append(alpha.squeeze(0).cpu().numpy())
            if idx == word2idx['<EOS>']:
                break
            word = idx2word.get(str(idx), idx2word.get(idx, '<UNK>'))
            if word not in ('<PAD>', '<SOS>', '<UNK>'):
                words.append(word)
            token = torch.tensor([idx], device=enc_out.device)
        return ' '.join(words), alphas


class EncoderDecoderAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, attn_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(hidden_dim)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_dim, hidden_dim, attn_dim)

    def caption(self, img, word2idx, idx2word):
        enc = self.encoder(img)
        return self.decoder.generate(enc, word2idx, idx2word)


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_model(pth_path, device):
    ckpt = torch.load(pth_path, map_location=device, weights_only=False) 
    hp       = ckpt['hyperparams']
    word2idx = ckpt['vocab_word2idx']
    idx2word = {str(k): v for k, v in ckpt['vocab_idx2word'].items()}
    model    = EncoderDecoderAttention(
        vocab_size = hp['vocab_size'],
        embed_dim  = hp['embed_dim'],
        hidden_dim = hp['hidden_dim'],
        attn_dim   = hp.get('attention_dim', 256),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, word2idx, idx2word


def predict(image_pil, model, word2idx, idx2word, device):
    tensor = transform(image_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        caption, alphas = model.caption(tensor, word2idx, idx2word)
    return caption, alphas