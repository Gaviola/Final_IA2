import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import re

from transformer import Transformer

def process_line( line: str, punt_charac: bool = False, ind_num: bool = True) -> str:
    if punt_charac:
        line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-?¿¡!()\"\',.]", "", line)
        line = re.sub(r"([?¿!¡()\"\',.])", r" \1 ",line)  # Separar puntuaciones para que los tome como tokens individuales
    else:
        line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-]", "", line)
    if ind_num:
        line = re.sub(r"([0-9\-])", r" \1 ", line)  # Separar números para que los tome como tokens individuales
    return line.lower().strip()

# Clases y funciones principales
class Vocabulary:
    def __init__(self, vocab_file):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Cargar palabras del archivo
        with open(vocab_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]

        # Añadir tokens especiales
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        # Añadir palabras del vocabulario
        for idx, word in enumerate(words, start=len(self.special_tokens)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.pad_token = self.word2idx['<pad>']
        self.sos_token = self.word2idx['<sos>']
        self.eos_token = self.word2idx['<eos>']
        self.unk_token = self.word2idx['<unk>']

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, data_file, src_vocab, tgt_vocab, max_length=100):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pairs = []


        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    src_sent = process_line(parts[0])
                    tgt_sent = process_line(parts[1])
                    self.pairs.append((src_sent, tgt_sent))

        self.max_length = max_length

    def sentence_to_indices(self, sentence, vocab) -> list:
        # Convertir a minúsculas y limpiar antes de tokenizar
        sentence = process_line(sentence)
        words = sentence.split()
        indices = [vocab.sos_token]

        # Mapeo con verificación explícita
        for word in words:
            if word in vocab.word2idx:
                indices.append(vocab.word2idx[word])
            else:
                indices.append(vocab.unk_token)

        indices.append(vocab.eos_token)
        return indices

    def __getitem__(self, idx):
        src_sent, tgt_sent = self.pairs[idx]

        # Procesar secuencia fuente
        src_indices = self.sentence_to_indices(src_sent, self.src_vocab)
        #src_indices = src_indices[:self.max_length]
        src_indices = src_indices + [self.src_vocab.pad_token] * (self.max_length - len(src_indices))

        # Procesar secuencia objetivo
        tgt_indices = self.sentence_to_indices(tgt_sent, self.tgt_vocab)
        #tgt_indices = tgt_indices[:self.max_length]
        tgt_indices = tgt_indices + [self.tgt_vocab.pad_token] * (self.max_length - len(tgt_indices))

        return {
            'src': torch.LongTensor(src_indices[:self.max_length]),
            'tgt': torch.LongTensor(tgt_indices[:self.max_length])
        }

    def __len__(self):
        return len(self.pairs)


def create_masks(src, tgt, src_pad_token, tgt_pad_token):
    # Máscara para encoder (padding)
    src_mask = (src == src_pad_token).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    #Unsqueeze: Añade una dimensión de tamaño 1 en la posición indicada para que las dimensiones sean compatibles con las heads
    # Máscara para decoder (padding + look-ahead)
    tgt_padding_mask = (tgt == tgt_pad_token).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    seq_length = tgt.size(1)
    look_ahead_mask = torch.triu(torch.ones(seq_length, seq_length, device=src.device), diagonal=1).bool()
    tgt_mask = tgt_padding_mask | look_ahead_mask.unsqueeze(0)  # [batch, 1, tgt_len, tgt_len]

    return src_mask.to(src.device), tgt_mask.to(src.device)


def translate(sentence, model, src_vocab, tgt_vocab, max_length=100, device='cpu', top_k=5):
    model.eval()
    sentence = process_line(sentence)
    src_indices = [src_vocab.sos_token] + [src_vocab.word2idx.get(word, src_vocab.unk_token) for word in
                                           sentence.split()] + [src_vocab.eos_token]
    src = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # Crear máscara fuente correctamente
    src_mask = (src != src_vocab.pad_token).unsqueeze(1).unsqueeze(2).to(device)  # [1, 1, 1, src_len]
    tgt_indices = [tgt_vocab.sos_token]

    for _ in range(max_length):
        tgt = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        #seq_length = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=device), diagonal=1).bool().unsqueeze(0)

        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)

            # Tomar el token con mayor probabilidad entre los top-k
            top_probs, top_indices = torch.topk(output[0, -1], top_k)
            pred_token = top_indices[0].item()  # Seleccionar el más probable
            tgt_indices.append(pred_token)

        if pred_token == tgt_vocab.eos_token:
            break

    translated_words = []
    for idx in tgt_indices[1:-1]:  # Excluir <sos> y <eos>
        word = tgt_vocab.idx2word.get(idx, '<unk>')
        translated_words.append(word)
        #print(f"Índice: {idx} -> Palabra: {word}")  # Debug
    return ' '.join(translated_words)


if __name__ == "__main__":

    # Configuración principal
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0002
    D_MODEL = 256
    NUM_LAYERS = 3
    NUM_HEADS = 4
    D_FF = 1024
    MAX_LENGTH = 100
    DROPOUT = 0.2

    print(f"device: {torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}\n"
          f"batch_size: {BATCH_SIZE}\n"
          f"num_epochs: {NUM_EPOCHS}\n"
          f"learning_rate: {LEARNING_RATE}\n"
          f"d_model: {D_MODEL}\n"
          f"num_layers: {NUM_LAYERS}\n"
          f"num_heads: {NUM_HEADS}\n"
          f"d_ff: {D_FF}\n"
          f"max_length: {MAX_LENGTH}\n"
          f"dropout: {DROPOUT}\n")


    # Cargar vocabularios
    src_vocab = Vocabulary('data/vocab_en_70000.txt')
    tgt_vocab = Vocabulary('data/vocab_es_70000.txt')

    # Preparar datos
    dataset = TranslationDataset('data/dataset_200000.txt', src_vocab, tgt_vocab, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inicializar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=MAX_LENGTH,
        dropout=DROPOUT
    ).to(device)

    # Configurar entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_token)

    # Bucle de entrenamiento
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        total_loss = 0
        model.train()
        i = 0
        for batch in dataloader:

            # Dentro del bucle de entrenamiento
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # Entrada del decoder debe excluir el último token
            tgt_input = tgt[:, :-1]

            # Salida esperada debe excluir el primer token
            tgt_output = tgt[:, 1:]

            # Crear máscaras con la versión recortada de tgt
            src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab.pad_token, tgt_vocab.pad_token)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            # Forward pass
            outputs = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 200 == 0:
                print(f'Batch: {i:04} |'
                      f' Loss: {loss.item():.4f} |'
                      f' Time: {time.time() - start_time:.2f}s |'
                      f' Examples: {i * BATCH_SIZE}')
            i += 1

        avg_loss = total_loss / len(dataloader)
        print("-"*100)
        print(f'Epoch: {epoch + 1:02} | Time: {time.time() - start_time:.2f}s | Loss: {avg_loss:.4f}')
        print("-" * 100)
        ex1 = "I am going to the park"
        ex2 = "Hello, how are you?"
        ex3 = "Hello world"
        ex4 = "This is a test sentence"
        ex5 = "this should be translated to spanish"

        for ex in [ex1, ex2, ex3, ex4, ex5]:
            translation = translate(ex, model, src_vocab, tgt_vocab, device=device.type)
            print(f"Original: {ex}\n")
            print(f"Traducción: {translation}\n")

        torch.save(model.state_dict(), f'models/best_model_mini_epoch{epoch + 1}.pt')

    # Ejemplo de traducción
    test_sentence = "your computer has enough resources"
    translation = translate(test_sentence, model, src_vocab, tgt_vocab, device=device.type)
    print(f'\nTest sentence: {test_sentence}')
    print(f'Translation: {translation}')