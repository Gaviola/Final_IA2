import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from data_proccess import process_line, Vocabulary, TranslationDataset
from transformer import Transformer

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

    mini = True

    if mini:
        # Configuración mini
        BATCH_SIZE = 32
        NUM_EPOCHS = 5
        LEARNING_RATE = 0.0002
        D_MODEL = 256
        NUM_LAYERS = 3
        NUM_HEADS = 4
        D_FF = 1024
        MAX_LENGTH = 100
        DROPOUT = 0.2
    else:
        # Configuración completa
        BATCH_SIZE = 64
        NUM_EPOCHS = 2
        LEARNING_RATE = 0.0002
        D_MODEL = 512
        NUM_LAYERS = 6
        NUM_HEADS = 8
        D_FF = 2048
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
    src_vocab = Vocabulary('data/subs/vocab_sub_en_70000.txt')
    tgt_vocab = Vocabulary('data/subs/vocab_sub_es_70000.txt')

    # Preparar datos
    dataset = TranslationDataset('data/subs/dataset_200000.txt', src_vocab, tgt_vocab, MAX_LENGTH)
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

        torch.save(model.state_dict(), f'models/best_model_mini_sub_epoch{epoch + 1}.pt')

    # Ejemplo de traducción
    test_sentence = "your computer has enough resources"
    translation = translate(test_sentence, model, src_vocab, tgt_vocab, device=device.type)
    print(f'\nTest sentence: {test_sentence}')
    print(f'Translation: {translation}')