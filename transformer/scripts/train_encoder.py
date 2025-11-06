import os, math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.data_utils import load_parquet_dataset, build_vocab_from_texts, encode_texts_to_ids, MaskedLMDataset
from utils.train_utils import count_parameters, compute_ppl, plot_training_curves
from models.encoder import TransformerEncoder

# ================================
# Config
# ================================
DATA_DIR = "./data"
OUT_DIR = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 64
BATCH_SIZE = 16
EPOCHS = 10
D_MODEL = 256
NUM_LAYERS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
VOCAB_SIZE = 30000
MASK_TOKEN = "<mask>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 数据
# ================================
ds = load_parquet_dataset(
    os.path.join(DATA_DIR, "train-*.parquet"),
    os.path.join(DATA_DIR, "validation-*.parquet"),
    os.path.join(DATA_DIR, "test-*.parquet")
)
train_texts = ds["train"]["text"]
valid_texts = ds["validation"]["text"]

stoi, itos = build_vocab_from_texts(train_texts, VOCAB_SIZE)
train_ids = encode_texts_to_ids(train_texts, stoi)
valid_ids = encode_texts_to_ids(valid_texts, stoi)

train_loader = torch.utils.data.DataLoader(MaskedLMDataset(train_ids, SEQ_LEN, 0.15, stoi[MASK_TOKEN]),
                                           batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(MaskedLMDataset(valid_ids, SEQ_LEN, 0.15, stoi[MASK_TOKEN]),
                                           batch_size=BATCH_SIZE)

# ================================
# 消融实验+训练
# ================================
ablation_configs = [
    {'name': 'baseline',    'pos_enc': True,  'residual': True,  'layernorm': True,  'n_head': 4, 'd_ff': 512},
    {'name': '-posEnc',     'pos_enc': False, 'residual': True,  'layernorm': True,  'n_head': 4, 'd_ff': 512},
    {'name': '-residual',   'pos_enc': True,  'residual': False, 'layernorm': True,  'n_head': 4, 'd_ff': 512},
    {'name': '-layernorm',  'pos_enc': True,  'residual': True,  'layernorm': False, 'n_head': 4, 'd_ff': 512},
    {'name': '-singlehead', 'pos_enc': True,  'residual': True,  'layernorm': True,  'n_head': 1, 'd_ff': 512},
    {'name': '-smallFFN',   'pos_enc': True,  'residual': True,  'layernorm': True,  'n_head': 4, 'd_ff': 128},
]


results = []
for cfg in ablation_configs:
    model = TransformerEncoder(len(stoi), D_MODEL, cfg['n_head'], NUM_LAYERS, cfg['d_ff'],
                               use_pos=cfg['pos_enc'], use_res=cfg['residual'], use_ln=cfg['layernorm']).to(DEVICE)
    print(f"[INFO] Model {cfg['name']} Params: {count_parameters(model)/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_losses, val_losses, train_ppls, val_ppls = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_tokens = 0, 0
        for x, y in tqdm(train_loader, desc=f"{cfg['name']} Epoch {epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item() * (y != -100).sum().item()
            n_tokens += (y != -100).sum().item()
        train_loss = total_loss / n_tokens
        train_ppl = compute_ppl(train_loss)

        model.eval()
        total_val, n_val = 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                total_val += loss.item() * (y != -100).sum().item()
                n_val += (y != -100).sum().item()
        val_loss = total_val / n_val
        val_ppl = compute_ppl(val_loss)

        scheduler.step(val_loss)
        print(f"[{cfg['name']}] Epoch {epoch} | TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, TrainPPL={train_ppl:.2f}, ValPPL={val_ppl:.2f}")

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_ppls.append(train_ppl); val_ppls.append(val_ppl)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_{cfg['name']}.pt"))
    results.append({'name':cfg['name'], 'train':train_losses, 'val':val_losses,
                    'train_ppl':train_ppls, 'val_ppl':val_ppls})

plot_training_curves(results, OUT_DIR)
print("Training done! Results saved in", OUT_DIR)
