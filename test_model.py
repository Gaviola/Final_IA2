import torch
from train import Vocabulary, process_line, translate
from transformer import Transformer


def load_model(model_path, config, device):
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


if __name__ == "__main__":
    # Configuración (debe coincidir con el entrenamiento)

    mini = True

    if mini:
        CONFIG = {
            'src_vocab_size': 70000,
            'tgt_vocab_size': 70000,
            'd_model': 256,
            'num_layers': 3,
            'num_heads': 4,
            'd_ff': 1024,
            'max_len': 100,
            'dropout': 0.2
        }
    else:
        CONFIG = {
            'src_vocab_size': 70000,
            'tgt_vocab_size': 70000,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'd_ff': 2048,
            'max_len': 100,
            'dropout': 0.2
        }

    # Parámetros de prueba
    MODEL_PATH = 'models/best_model_mini_epoch7.pt'
    SRC_VOCAB_FILE = 'data/vocab_en_70000.txt'
    TGT_VOCAB_FILE = 'data/vocab_es_70000.txt'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar componentes
    src_vocab = Vocabulary(SRC_VOCAB_FILE)
    tgt_vocab = Vocabulary(TGT_VOCAB_FILE)
    CONFIG['src_vocab_size'] = len(src_vocab)
    CONFIG['tgt_vocab_size'] = len(tgt_vocab)

    model = load_model(MODEL_PATH, CONFIG, DEVICE)

    ex1 = "I am going to the park"
    ex2 = "Hello, how are you?"
    ex3 = "Hello world"
    ex4 = "This is a test sentence"
    ex5 = "this should be translated to spanish"

    for ex in [ex1, ex2, ex3, ex4, ex5]:
        translation = translate(ex, model, src_vocab, tgt_vocab, device=DEVICE.type)
        print("-"*100)
        print(f"Original: {ex}\n")
        print(f"Traducción: {translation}\n")
        print("-" * 100)

    # Bucle interactivo
    print("Traductor Inglés-Español (escribe 'exit' para salir)")
    while True:
        sentence = input("\nIngresa frase en inglés: ")
        if sentence.lower() == 'exit':
            break

        translation = translate(sentence, model, src_vocab, tgt_vocab, device=DEVICE.type)
        print("-" * 100)
        print(f"Traducción: {translation}")

