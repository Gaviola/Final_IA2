import re

import numpy as np
import torch

from torch.utils.data import Dataset

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TextDS(Dataset):
    def __init__(self,
                 english_text: list[str],
                 spanish_text: list[str]):
        self.english_text = english_text
        self.spanish_text = spanish_text

    def __len__(self):
        return len(self.english_text)

    def __getitem__(self, index: int):
        return self.english_text[index], self.spanish_text[index]

def process_line(line: str, punt_charac: bool = False, ind_num: bool = True) -> str:
    if punt_charac:
        line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-?¿¡!()\"\',.]", "", line)
        line = re.sub(r"([?¿!¡()\"\',.])", r" \1 ", line) # Separar puntuaciones para que los tome como tokens individuales
    else:
        line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-]", "", line)
    if ind_num:
        line = re.sub(r"([0-9])", r" \1 ", line) # Separar números para que los tome como tokens individuales
    return line.lower().strip()

MAX_SEQ_LENGTH = 300
max_sequense_length = MAX_SEQ_LENGTH
PADDING_TOKEN = "[PAD]"
NEG_INF = -1e10

def build_vocab(text_path: str, max_vocab_size: int = 5000, path: str = "data/default_vocab.txt") -> None:
    with open(text_path, "r", encoding='utf-8') as file:
        texts = file.readlines()

    words = set()
    words.add("[START]")
    words.add("[PAD]")
    words.add("[UNK]")
    words.add("[EOS]")
    i = 0

    while len(words) < max_vocab_size and i < len(texts):
        line = texts[i]
        trim_line = process_line(line).split()
        i += 1
        for j in range(len(trim_line)):
            words.add(trim_line[j])

    with open(path, 'w', encoding='utf-8') as f:
        for word in words:
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


def create_data_set(max_sentences: int = 100000) -> None:
    text_es = []
    text_en = []

    with open("data/en-es.txt/ParaCrawl.en-es.es", "r", encoding='utf-8') as es_file, \
            open("data/en-es.txt/ParaCrawl.en-es.en", "r", encoding='utf-8') as en_file:

        added_count = 0
        for es_line, en_line in zip(es_file, en_file):
            if added_count >= max_sentences:
                break

            # Procesar ambas oraciones
            processed_es = process_line(es_line).split()
            processed_en = process_line(en_line).split()

            # Filtrar por longitud
            if len(processed_es) > MAX_SEQ_LENGTH or len(processed_en) > MAX_SEQ_LENGTH:
                continue  # Saltar este par

            text_es.append(processed_es)
            text_en.append(processed_en)
            added_count += 1

    # Escribir solo las oraciones válidas
    with open("data/dataset.txt", "w", encoding='utf-8') as file:
        for en, es in zip(text_en, text_es):
            file.write(f"{en}\t{es}\n")

    print(f"Dataset creado con {len(text_es)} pares válidos (max_length={MAX_SEQ_LENGTH}).")


def load_data_set(path: str) -> TextDS:
    english_text = []
    spanish_text = []
    with open(path, "r", encoding='utf-8') as file:
        for line in file:
            en, es = line.strip().split("\t")
            english_text.append(eval(en))
            spanish_text.append(eval(es))
    return TextDS(english_text, spanish_text)

def save_preprocessed_data(dataset, path):
    torch.save({
        "en": [sentence for sentence in dataset.english_text],
        "es": [sentence for sentence in dataset.spanish_text]
    }, path)

def load_preprocessed_data(path):
    data = torch.load(path)
    return TextDS(data["en"], data["es"])

def fill_sentence_batch(sentences: list[list[str]], max_len: int) -> torch.Tensor:
    padded = []
    for sentence in sentences:
        padding_needed = max_len - len(sentence)
        for i in range(padding_needed):
            sentence.append(PADDING_TOKEN)
    return torch.tensor(padded, device=get_device())  # Vectorizado

def create_masks(en_batch, es_batch):
    num_sentences = len(en_batch)

    # Crea la mascara de look-ahead para el decoder
    look_ahead_mask = torch.full([MAX_SEQ_LENGTH, MAX_SEQ_LENGTH], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)

    # Inicializa las mascaras de padding para el encoder y decoder
    encoder_padding_mask = torch.full([num_sentences, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH], False)

    for i in range(num_sentences):
        # Determina la longitud de la sentencia excluyendo los tokens de padding
        en_sentence_length = en_batch[i].index(PADDING_TOKEN) if PADDING_TOKEN in en_batch[i] else MAX_SEQ_LENGTH
        es_sentence_length = es_batch[i].index(PADDING_TOKEN) if PADDING_TOKEN in es_batch[i] else MAX_SEQ_LENGTH

        # Crea las mascaras de padding para la sentencia actual
        en_chars_to_padding_mask = np.arange(en_sentence_length, MAX_SEQ_LENGTH)
        es_chars_to_padding_mask = np.arange(es_sentence_length, MAX_SEQ_LENGTH)

        # Se actualiza la mascara de padding para el encoder
        encoder_padding_mask[i, :, en_chars_to_padding_mask] = True
        encoder_padding_mask[i, en_chars_to_padding_mask, :] = True

        # Se actualiza la mascara de padding para el decoder
        decoder_padding_mask_self_attention[i, :, es_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[i, es_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[i, :, en_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[i, es_chars_to_padding_mask, :] = True

    # Crea las mascaras finales aplicando -infinito a las posiciones de padding
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INF, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INF, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INF, 0)

    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def collate_fn(batch):
    en_batch, es_batch = zip(*batch)
    en_batch = [sentence + [PADDING_TOKEN] * (max_sequense_length - len(sentence)) for sentence in en_batch]
    es_batch = [sentence + [PADDING_TOKEN] * (max_sequense_length - len(sentence)) for sentence in es_batch]
    return en_batch, es_batch


if __name__ == "__main__":
    #dataset = load_data_set("data/dataset.txt")
    create_data_set()
    build_vocab("data/en-es.txt/ParaCrawl.en-es.es", path="data/vocab_es.txt")
    build_vocab("data/en-es.txt/ParaCrawl.en-es.en", path="data/vocab_en.txt")
    #save_preprocessed_data(dataset, "data/preprocessed_data.pt")
    print("Done")

