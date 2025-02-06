import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from data_proccess import process_line, Vocabulary, TranslationDataset
from transformer import Transformer


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_token: int) -> (torch.Tensor, torch.Tensor):
    """
       Crea máscaras para el encoder y el decoder.

       Args:
       src (torch.Tensor): Tensor de entrada fuente.
       tgt (torch.Tensor): Tensor de entrada objetivo.
       pad_token (int): El token de Padding <pad> --> 0.

       Return:
       Tuple[torch.Tensor, torch.Tensor]: Máscara para el encoder y máscara para el decoder.
       """
    # Máscara para encoder (padding)
    src_mask = (src == pad_token).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    #Unsqueeze: Añade una dimensión de tamaño 1 en la posición indicada para que las dimensiones sean compatibles con las heads
    # Máscara para decoder (padding + look-ahead)
    tgt_padding_mask = (tgt == pad_token).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    seq_length = tgt.size(1)
    look_ahead_mask = torch.triu(torch.ones(seq_length, seq_length, device=src.device), diagonal=1).bool()
    tgt_mask = tgt_padding_mask | look_ahead_mask.unsqueeze(0)  # [batch, 1, tgt_len, tgt_len]

    return src_mask.to(src.device), tgt_mask.to(src.device)


def remove_eos(seq: torch.Tensor, eos_token: int, pad_token: int) -> torch.Tensor:
    """
    Recibe una secuencia (tensor 1D) y, si encuentra el token eos, lo elimina y
    agrega un token pad al final para mantener la longitud original.

    Args:
    seq (torch.Tensor): La secuencia de entrada como tensor 1D.
    eos_token (int): El token de fin de secuencia (eos) --> 2.
    pad_token (int): El token de padding (pad) --> 0.

    Return:
    Tensor: La secuencia modificada con el token eos eliminado y un token pad agregado al final.
    """
    # Convertir el tensor a lista (suponemos que es de tipo int)
    seq_list = seq.tolist()
    try:
        # Buscar la primera aparición del token <eos>
        idx = seq_list.index(eos_token)
        # Eliminar el token <eos>
        seq_list.pop(idx)
        # Agregar un token <pad> al final
        seq_list.append(pad_token)
    except ValueError:
        # Si no se encuentra <eos>, se deja la secuencia como está
        pass
    # Convertir la lista nuevamente a tensor
    return torch.LongTensor(seq_list)


def translate(sentence: str,
              model: nn.Module,
              src_vocab: Vocabulary,
              tgt_vocab: Vocabulary,
              max_length: int = 100,
              device: str = 'cpu',
              top_k: int = 5
              ) -> str:
    """
    Traduce una oración de un idioma fuente a un idioma objetivo usando un modelo de Transformer.

    Args:
    sentence (str): La oración en el idioma fuente que se desea traducir.
    model (nn.Module): El modelo de Transformer entrenado.
    src_vocab (Vocabulary): El vocabulario del idioma fuente.
    tgt_vocab (Vocabulary): El vocabulario del idioma objetivo.
    max_length (int): La longitud máxima de la traducción. Por defecto es 100.
    device (str): El dispositivo en el que se ejecuta el modelo ('cpu' o 'cuda'). Por defecto es 'cpu'.
    top_k (int): El número de tokens más probables a considerar en cada paso de decodificación. Por defecto es 5.

    Return:
    str: La oración traducida en el idioma objetivo.
    """

    model.eval()
    sentence = process_line(sentence)
    src_indices = [src_vocab.sos_token] + [src_vocab.word2idx.get(word, src_vocab.unk_token) for word in
                                           sentence.split()] + [src_vocab.eos_token]
    src = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # Crear máscara fuente correctamente
    src_mask = (src == src_vocab.pad_token).unsqueeze(1).unsqueeze(2).to(device)  # [1, 1, 1, src_len]
    tgt_indices = [tgt_vocab.sos_token]

    for _ in range(max_length):
        tgt = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)

        tgt_padding_mask = (tgt == tgt_vocab.pad_token).unsqueeze(1).unsqueeze(2).to(device)  # [1, 1, tgt_len, tgt_len]
        look_ahead_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=device), diagonal=1).bool().unsqueeze(0)
        tgt_mask = tgt_padding_mask | look_ahead_mask.unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

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

    mini = False
    vocab_size = 120000
    dataset_size = 200000

    if mini:
        # Configuración mini
        BATCH_SIZE = 32
        NUM_EPOCHS = 5
        LEARNING_RATE = 0.0003
        D_MODEL = 256
        NUM_LAYERS = 3
        NUM_HEADS = 4
        D_FF = 1024
        MAX_LENGTH = 100
        DROPOUT = 0.1
    else:
        # Configuración completa
        BATCH_SIZE = 64
        NUM_EPOCHS = 2
        LEARNING_RATE = 0.0003
        D_MODEL = 512
        NUM_LAYERS = 6
        NUM_HEADS = 8
        D_FF = 2048
        MAX_LENGTH = 100
        DROPOUT = 0.1

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
    src_vocab = Vocabulary(f'data/vocab_en_{vocab_size}.txt')
    tgt_vocab = Vocabulary(f'data/vocab_es_{vocab_size}.txt')

    # Preparar datos
    dataset = TranslationDataset(f'data/en-es.txt/dataset_{dataset_size}.txt', src_vocab, tgt_vocab, MAX_LENGTH)
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

            # Entrada del decoder debe excluir el token <eos> y agregar un token <pad> al final
            tgt_input = torch.stack([
                remove_eos(seq, tgt_vocab.eos_token, tgt_vocab.pad_token)
                for seq in tgt
            ]).to(device)

            # Salida esperada debe excluir el token <sos> y agregar un token <pad> al final
            tgt_output = tgt[:, 1:]
            # Se crea un tensor de tamaño (batch_size, 1) lleno del token de padding
            pad_tensor = torch.full((tgt.size(0), 1), tgt_vocab.pad_token, dtype=tgt.dtype, device=tgt.device).to(device)
            # Se concatena a lo largo de la dimensión de secuencia para mantener la longitud original L
            tgt_output = torch.cat([tgt_output, pad_tensor], dim=1)


            # Crear máscaras con la versión recortada de tgt
            src_mask, tgt_mask = create_masks(src, tgt, src_vocab.pad_token)
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
                average_loss_batch = total_loss / (i + 1)
                print(f'Batch: {i:04} |'
                      f' Loss: {average_loss_batch:.4f} |'
                      f' Time: {time.time() - start_time:.2f}s |'
                      f' Examples: {i * BATCH_SIZE}')
            i += 1

        avg_loss = total_loss / len(dataloader)
        print("-"*100)
        print(f'Epoch: {epoch + 1:02} | Time: {time.time() - start_time:.2f}s | Loss: {avg_loss:.4f}')
        print("-" * 100)
        ex1 = "I love machine learning"
        ex2 = "The book that you lent me is fascinating"
        ex3 = "Hello world"
        ex4 = "Although artificial intelligence has many benefits, some people worry about its ethical implications."
        ex5 = "This should be translated to spanish"

        for ex in [ex1, ex2, ex3, ex4, ex5]:
            translation = translate(ex, model, src_vocab, tgt_vocab, device=device.type)
            print(f"Original: {ex}\n")
            print(f"Traducción: {translation}\n")

        if mini:
            torch.save(model.state_dict(), f'models/best_model_mini_epoch{epoch + 1}_{vocab_size}_{dataset_size}.pt')
        else:
            torch.save(model.state_dict(), f'models/best_model_epoch{epoch + 1}_{vocab_size}_{dataset_size}.pt')

    # Ejemplo de traducción
    test_sentence = "The training is finished, let's test the model"
    translation = translate(test_sentence, model, src_vocab, tgt_vocab, device=device.type)
    print(f'\nTest sentence: {test_sentence}')
    print(f'Translation: {translation}')