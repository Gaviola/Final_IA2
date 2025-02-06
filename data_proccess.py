import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def process_line(line: str, ind_num: bool = True) -> str:
    """
    Procesa una línea de texto aplicando varias transformaciones:
    - Elimina caracteres no alfanuméricos, excepto ñ, Ñ, acentos, espacios, guiones y apóstrofes.
    - Separa guiones en palabras individuales.
    - Separa apóstrofes solo si están en los extremos de una palabra.
    - Opcionalmente, separa números en tokens individuales.
    - Elimina espacios duplicados.

    Args:
        line (str): La línea de texto a procesar.
        ind_num (bool): Si es True, separa los números en tokens individuales.

    Returns:
        str: La línea de texto procesada.
    """
    line = re.sub(r"[^a-zA-Z0-9ñÑáéíóú\s\-']", "", line)
    line = re.sub(r"([\-])", " ", line) # Separar guiones
    line = re.sub(r"(?<!\w)'|'(?!\w)", " ' ", line) # Separar apóstrofes
    if ind_num:
        line = re.sub(r"([0-9])", r" \1 ", line) # Separar números
    line = re.sub(r"\s+", " ", line)  # Eliminar espacios duplicados

    return line.lower()


class Vocabulary:
    """
    Clase para manejar el vocabulario del modelo de traducción.
    """
    def __init__(self, vocab_file):
        """
        Inicializa la clase Vocabulary cargando las palabras del archivo y añadiendo tokens especiales.

        Args:
            vocab_file (str): Ruta al archivo que contiene las palabras del vocabulario.
        """
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
        """
        Devuelve el tamaño del vocabulario.

        Returns:
            int: Número de palabras en el vocabulario.
        """
        return len(self.word2idx)


class TranslationDataset(Dataset):
    """
    Clase para manejar el dataset de traducción.
    """
    def __init__(self, data_file: str, src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_length: int = 100):
        """
        Inicializa la clase TranslationDataset.

        Args:
            data_file (str): Ruta al archivo que contiene los pares de oraciones.
            src_vocab (Vocabulary): Vocabulario del lenguaje fuente.
            tgt_vocab (Vocabulary): Vocabulario del lenguaje objetivo.
            max_length (int): Longitud máxima de las secuencias.
        """
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

    def sentence_to_indices(self, sentence: list, vocab: Vocabulary) -> list:
        """
        Convierte una oración en una lista de índices según el vocabulario.

        Args:
            sentence (list): Lista de palabras de la oración.
            vocab (Vocabulary): Vocabulario a utilizar para la conversión.

        Returns:
            list: Lista de índices correspondientes a las palabras de la oración.
        """
        indices = [vocab.sos_token] # Agregamos Inicio de secuencia <sos>

        # Mapeo con verificación explícita
        for word in sentence:
            if word in vocab.word2idx:
                indices.append(vocab.word2idx[word])
            else:
                indices.append(vocab.unk_token)


        indices.append(vocab.eos_token) # Agregamos Fin de secuencia <eos>
        return indices

    def __getitem__(self, idx):
        """
        Obtiene el par de oraciones en el índice especificado y las convierte en índices.

        Args:
            idx (int): Índice del par de oraciones a obtener.

        Returns:
            dict: Diccionario con las secuencias fuente y objetivo convertidas en índices y los tokens especiales.
        """
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

    def __len__(self) -> int:
        """
        Devuelve el número de pares de oraciones en el dataset.

        Returns:
            int: Número de pares de oraciones.
        """
        return len(self.pairs)


def build_vocab(text_path: str, max_vocab_size: int = 120000, path: str = "data/default_vocab.txt") -> None:
    """
    Construye un vocabulario a partir de un archivo de texto.

    Args:
        text_path (str): Ruta al archivo de texto que contiene las oraciones.
        max_vocab_size (int): Tamaño máximo del vocabulario.
        path (str): Ruta donde se guardará el vocabulario generado.
    """
    words = Counter()

    def process_lines(lines):
        """
        Procesa una lista de líneas y cuenta las palabras.

        Args:
            lines (list): Lista de líneas de texto.

        Returns:
            Counter: Contador de palabras.
        """
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
    """
    Carga un vocabulario desde un archivo y devuelve dos diccionarios:
    uno para mapear palabras a índices y otro para mapear índices a palabras.

    Args:
        path (str): Ruta al archivo que contiene el vocabulario.

    Returns:
        tuple[dict, dict]: Un par de diccionarios (vocab, reverse_vocab).
    """
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
                    max_seq_length: int = 100,
                    vocab_only: bool = False,
                    ) -> None:
    """
    Crea un dataset de traducción a partir de archivos de texto fuente y objetivo.

    Args:
        source (str): Ruta base a los archivos de texto fuente y objetivo (sin extensión).
        max_sentences (int): Número máximo de pares de oraciones a procesar.
        max_seq_length (int): Longitud máxima de las secuencias.
        vocab_only (bool): Si es True, filtra las oraciones que contengan palabras no presentes en el vocabulario.

    Returns:
        None
    """
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
            if len(es_words) > max_seq_length - 2 or len(es_words) > max_seq_length - 2:
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

    print(f"Dataset creado con {len(text_es)} pares válidos (MAX_SEQ_LENGTH={max_seq_length}).")


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

