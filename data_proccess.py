import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def process_line(line: str, ind_num: bool = True) -> str:
    line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-']", "", line)
    line = re.sub(r"([\-])", " ", line) # Separar guiones
    #Separar apóstrofes, unicamente si estan en los extremos de una palabra ('food'), para que los tome como tokens individuales
    line = re.sub(r"(?<!\w)'|'(?!\w)", " ' ", line)
    if ind_num:
        line = re.sub(r"([0-9])", r" \1 ", line) # Separar números para que los tome como tokens individuales
    line = re.sub(r"\s+", " ", line)  # Eliminar espacios duplicados

    return line.lower()

# Clases y funciones principales
class Vocabulary:
    def __init__(self, vocab_file):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Cargar palabras del archivo
        with open(vocab_file, 'r', encoding='utf-8') as f:
            words = [line for line in f]

        # Añadir tokens especiales
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        # Añadir palabras del vocabulario
        for idx, word in enumerate(words, start=len(self.special_tokens)):
            self.word2idx[word.strip()] = idx
            self.idx2word[idx] = word.strip()

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
                parts = line.strip().split('|')
                if len(parts) == 2:
                    src_sent = parts[0].split()
                    tgt_sent = parts[1].split()
                    self.pairs.append((src_sent, tgt_sent))

        self.max_length = max_length

    def sentence_to_indices(self, sentence, vocab) -> list:
        indices = [vocab.sos_token] # Agregamos Inicio de secuencia

        # Mapeo con verificación explícita
        for word in sentence:
            if word in vocab.word2idx:
                indices.append(vocab.word2idx[word])
            else:
                indices.append(vocab.unk_token)


        indices.append(vocab.eos_token) # Agregamos Fin de secuencia
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


def build_vocab(text_path: str, max_vocab_size: int = 120000, path: str = "data/default_vocab.txt") -> None:
    words = Counter()

    def process_lines(lines):
        local_counter = Counter()
        for line in lines:
            local_counter.update(process_line(line).split())
        return local_counter

    with open(text_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunk_size = len(lines) // os.cpu_count()
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_lines, [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)])

    for result in results:
        words.update(result)

    with open(path, 'w', encoding='utf-8') as f:
        for word, _ in words.most_common(max_vocab_size):
            f.write(f"{word}\n")


def load_vocab(path: str) -> tuple[dict, dict]:
    vocab = {}
    reverse_vocab = {}
    with open(path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            word = line.strip()
            vocab[word] = idx
            reverse_vocab[idx] = word
    return vocab, reverse_vocab


def create_data_set(source: str,
                    max_sentences: int = 600000,
                    MAX_SEQ_LENGTH: int = 100,
                    vocab_only: bool = False,
                    ) -> None:

    vocab_en, _ = load_vocab(f"data/vocab_en_120000.txt")
    vocab_es, _ = load_vocab(f"data/vocab_es_120000.txt")

    text_es = []
    text_en = []

    with open(f"{source}.es", "r", encoding='utf-8') as es_file, \
            open(f"{source}.en", "r", encoding='utf-8') as en_file:

        added_count = 0
        for es_line, en_line in zip(es_file, en_file):
            if added_count >= max_sentences:
                break

            # Procesar ambas oraciones
            processed_es = process_line(es_line)
            processed_en = process_line(en_line)

            es_words = processed_es.split()
            en_words = processed_en.split()

            # Filtrar por longitud
            if len(es_words) > MAX_SEQ_LENGTH - 2 or len(es_words) > MAX_SEQ_LENGTH - 2:
                continue  # Saltar este par

            if vocab_only:
                # Filtrar por vocabulario
                valid_es = all(word in vocab_es for word in es_words)
                valid_en = all(word in vocab_en for word in en_words)
                if valid_es and valid_en:
                    text_es.append(processed_es)
                    text_en.append(processed_en)
                    added_count += 1
            else:
                text_es.append(processed_es)
                text_en.append(processed_en)
                added_count += 1

    # Escribir el dataset resultante
    with open(f"data/en-es.txt/dataset_{max_sentences}.txt", "w", encoding='utf-8') as file:
        # Se escribe en cada línea el par "inglés | español"
        for en, es in zip(text_en, text_es):
            file.write(f"{en} | {es}\n")

    print(f"Dataset creado con {len(text_es)} pares válidos (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}).")


if __name__ == "__main__":
    #build_vocab("data/en-es.txt/ParaCrawl.en-es.es", path="data/vocab_es_120000.txt")
    #build_vocab("data/en-es.txt/ParaCrawl.en-es.en", path="data/vocab_en_120000.txt")
    create_data_set(source="data/en-es.txt/ParaCrawl.en-es", vocab_only=True)


    #src_vocab = Vocabulary('data/vocab_en_70000.txt')
    # = Vocabulary('data/vocab_es_70000.txt')

    #print_vocab_stats(src_vocab,"en")
    #print_vocab_stats(tgt_vocab,"es")

    #dataset = TranslationDataset('data/dataset_200000.txt', src_vocab, tgt_vocab, MAX_SEQ_LENGTH)
    #dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Done")

