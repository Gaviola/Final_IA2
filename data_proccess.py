import re
import numpy as np
import torch
from train import TranslationDataset, Vocabulary

from torch.utils.data import Dataset, DataLoader


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
        line = re.sub(r"([0-9\-])", r" \1 ", line) # Separar números para que los tome como tokens individuales
    return line.lower().strip()

BATCH_SIZE = 64
MAX_SEQ_LENGTH = 100
PADDING_TOKEN = "[PAD]"
NEG_INF = -1e10

def build_vocab(text_path: str, max_vocab_size: int = 70000, path: str = "data/default_vocab.txt", whole = False) -> None:
    with open(text_path, "r", encoding='utf-8') as file:
        texts = file.readlines()

    words = set()
    words.add("[START]")
    words.add("[PAD]")
    words.add("[UNK]")
    words.add("[EOS]")
    i = 0

    if whole:
        for line in texts:
            trim_line = process_line(line).split()
            for j in range(len(trim_line)):
                words.add(trim_line[j])
    else:
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


def create_data_set(max_sentences: int = 200000, vocab_only = False) -> None:
    vocab_en, _ = load_vocab("data/vocab_en_70000.txt")
    vocab_es, _ = load_vocab("data/vocab_es_70000.txt")

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
            if len(processed_es) > MAX_SEQ_LENGTH - 2 or len(processed_en) > MAX_SEQ_LENGTH - 2:
                continue  # Saltar este par

            if vocab_only:
                # Filtrar por vocabulario
                valid_es = all(word in vocab_es for word in processed_es)
                valid_en = all(word in vocab_en for word in processed_en)
                if valid_es and valid_en:
                    text_es.append(processed_es)
                    text_en.append(processed_en)
                    added_count += 1
            else:
                text_es.append(processed_es)
                text_en.append(processed_en)
                added_count += 1



    # Escribir solo las oraciones válidas
    with open("data/dataset_200000.txt", "w", encoding='utf-8') as file:
        for en, es in zip(text_en, text_es):
            file.write(f"{en}\t{es}\n")

    print(f"Dataset creado con {len(text_es)} pares válidos (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}).")


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
    en_batch = [sentence + [PADDING_TOKEN] * (MAX_SEQ_LENGTH - len(sentence)) for sentence in en_batch]
    es_batch = [sentence + [PADDING_TOKEN] * (MAX_SEQ_LENGTH - len(sentence)) for sentence in es_batch]
    return en_batch, es_batch

def print_vocab_stats(vocab, name):
    print(f"\nEstadísticas del vocabulario {name}:")
    print(f"Total de palabras: {len(vocab)}")
    print(f"Ejemplo de mapeo:")
    for i in range(5):
        print(f"{vocab.idx2word[i]} -> {i}")

def inspect_batch(batch, src_vocab, tgt_vocab):
    print("\nInspección de batch:")
    print("Secuencia fuente (índices):", batch['src'][0])
    print("Secuencia fuente (palabras):", [src_vocab.idx2word.get(idx.item(), '<unk>') for idx in batch['src'][0]])
    print("Secuencia objetivo (índices):", batch['tgt'][0])
    print("Secuencia objetivo (palabras):", [tgt_vocab.idx2word.get(idx.item(), '<unk>') for idx in batch['tgt'][0]])


if __name__ == "__main__":
    build_vocab("data/en-es.txt/ParaCrawl.en-es.es", path="data/vocab_es_70000.txt")
    build_vocab("data/en-es.txt/ParaCrawl.en-es.en", path="data/vocab_en_70000.txt")
    create_data_set(vocab_only=True)

    #src_vocab = Vocabulary('data/vocab_en_70000.txt')
    #tgt_vocab = Vocabulary('data/vocab_es_70000.txt')

    #print_vocab_stats(src_vocab,"en")
    #print_vocab_stats(tgt_vocab,"es")

    #dataset = TranslationDataset('data/dataset_200000.txt', src_vocab, tgt_vocab, MAX_SEQ_LENGTH)
    #dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    #first_batch = next(iter(dataloader))
    #inspect_batch(first_batch, src_vocab, tgt_vocab)

    print("Done")

