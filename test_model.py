import torch
from data_proccess import process_line
from train import Vocabulary, translate
from transformer import Transformer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm


def load_model(model_path: str, config: dict, device: torch.device) -> Transformer:
    """
    Carga un modelo Transformer desde un archivo.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        config (dict): Diccionario de configuración con los parámetros del modelo.
        device (torch.device): Dispositivo en el que se cargará el modelo (CPU o GPU).

    Returns:
        Transformer: El modelo Transformer cargado y preparado para evaluación.
    """
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model: torch.nn.Module,
                   src_vocab: Vocabulary,
                   tgt_vocab: Vocabulary,
                   test_file: str,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   max_samples: int = None,
                   max_length: int = 100) -> dict:
    """
    Evalúa un modelo de traducción usando las métricas BLEU y ROUGE.

    Args:
        model: Modelo de Transformer entrenado
        src_vocab: Vocabulario del lenguaje fuente
        tgt_vocab: Vocabulario del lenguaje objetivo
        test_file: Ruta al archivo de prueba (formato 'source | target')
        device: Dispositivo para ejecución
        max_samples: Número máximo de muestras a evaluar (None para todas)
        max_length: Longitud máxima de las secuencias

    Returns:
        Diccionario con puntuaciones BLEU y ROUGE
    """
    # Cargar datos de prueba
    references = []
    hypotheses = []

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples] if max_samples else f.readlines()

        for line in tqdm(lines, desc="Procesando muestras"):
            parts = line.strip().split('|')
            if len(parts) == 2:
                src = process_line(parts[0].strip())
                ref = process_line(parts[1].strip()).split()

                # Generar traducción
                translation = translate(src, model, src_vocab, tgt_vocab,
                                        max_length=max_length, device=device)

                references.append([ref])
                hypotheses.append(translation.split())

    # Calcular BLEU
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25)
    )

    # Calcular ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [] }

    for ref, hyp in tqdm(zip(references, hypotheses), desc="Calculando ROUGE"):
        scores = scorer.score(' '.join(ref[0]), ' '.join(hyp))
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # Promediar puntuaciones
    return {
        'bleu': bleu_score,
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }


if __name__ == "__main__":
    # Configuración (debe coincidir con el entrenamiento)

    mini = True

    if mini:
        CONFIG = {
            'src_vocab_size': 120000,
            'tgt_vocab_size': 120000,
            'd_model': 256,
            'num_layers': 3,
            'num_heads': 4,
            'd_ff': 1024,
            'max_len': 100,
            'dropout': 0.1
        }
    else:
        CONFIG = {
            'src_vocab_size': 120000,
            'tgt_vocab_size': 120000,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'd_ff': 2048,
            'max_len': 100,
            'dropout': 0.1
        }

    # Parámetros de prueba
    MODEL_PATH = 'models/model_mini_epoch5_120000_400000.pt'
    test_file = 'data/en-es.txt/dataset_600000.txt'
    SRC_VOCAB_FILE = 'data/vocab_en_120000.txt'
    TGT_VOCAB_FILE = 'data/vocab_es_120000.txt'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar componentes
    src_vocab = Vocabulary(SRC_VOCAB_FILE)
    tgt_vocab = Vocabulary(TGT_VOCAB_FILE)
    CONFIG['src_vocab_size'] = len(src_vocab)
    CONFIG['tgt_vocab_size'] = len(tgt_vocab)

    model = load_model(MODEL_PATH, CONFIG, DEVICE)

    ex1 = "I love machine learning"
    ex2 = "The book that you lent me is fascinating"
    ex3 = "Hello world"
    ex4 = "Although artificial intelligence has many benefits, some people worry about its ethical implications."
    ex5 = "This should be translated to spanish"

    for ex in [ex1, ex2, ex3, ex4, ex5]:
        translation = translate(ex, model, src_vocab, tgt_vocab, device=DEVICE.type)
        print("-"*100)
        print(f"Original: {ex}\n")
        print(f"Traducción: {translation}\n")
        print("-" * 100)

    # Evaluar modelo con las primeras 1000 muestras con las metricas BLEU y ROUGE
    scores = evaluate_model(
        model,
        src_vocab,
        tgt_vocab,
        test_file,
        device=DEVICE.type,
        max_samples=1000,
    )

    print("\nResultados de evaluación:")
    print(f"BLEU: {scores['bleu']:.4f}")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")

    # Bucle interactivo
    print("Traductor Inglés-Español (escribe 'exit' para salir)")
    while True:
        sentence = input("\nIngresa frase en inglés: ")
        if sentence.lower() == 'exit':
            break

        translation = translate(sentence, model, src_vocab, tgt_vocab, device=DEVICE.type)
        print("-" * 100)
        print(f"Traducción: {translation}")

