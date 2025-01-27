import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_proccess import TextDS, load_vocab, create_masks, collate_fn, MAX_SEQ_LENGTH, load_data_set, fill_sentence
from transformer import Transformer, get_device
import time
import math

# Configuración hiperparámetros
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 3
FFN_HIDDEN = 2048
DROPOUT_PROB = 0.1


# Cargar datos y vocabularios
def prepare_data():
    # Cargar dataset
    dataset = load_data_set("data/dataset.txt")

    # Dividir en train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    # Cargar vocabularios
    en_vocab, _ = load_vocab("data/vocab_en.txt")
    es_vocab, _ = load_vocab("data/vocab_es.txt")

    return train_data, val_data, en_vocab, es_vocab


# Inicializar modelo
def initialize_model(en_vocab, es_vocab):
    model = Transformer(
        d_model=D_MODEL,
        ffn_hidden=FFN_HIDDEN,
        num_heads=NUM_HEADS,
        drop_prob=DROPOUT_PROB,
        num_layers=NUM_LAYERS,
        max_sequence_length=MAX_SEQ_LENGTH,
        es_vocab_size=len(es_vocab),
        english_to_index=en_vocab,
        spanish_to_index=es_vocab
    ).to(get_device())
    return model


# Función de entrenamiento
def train_model():
    train_data, val_data, en_vocab, es_vocab = prepare_data()

    # Crear DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = initialize_model(en_vocab, es_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["[PAD]"])

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0

        for batch_idx, (en_batch, es_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # --- Cambios clave aquí ---
            # 1. Aplicar padding ANTES de convertir a tensores
            en_batch = [fill_sentence(sentence, MAX_SEQ_LENGTH) for sentence in en_batch]
            es_batch = [fill_sentence(sentence, MAX_SEQ_LENGTH) for sentence in es_batch]

            # 2. Convertir directamente usando el vocabulario
            en_tensor = torch.tensor(
                [[en_vocab.get(word, en_vocab["[UNK]"]) for word in sentence]
                 for sentence in en_batch]).long().to(get_device())

            target_tensor = torch.tensor(
                [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence]
                 for sentence in es_batch]).long().to(get_device())

            # 3. Preparar entrada del decoder (shift right)
            decoder_input = [["[START]"] + sentence[:-1] for sentence in es_batch]
            decoder_input_tensor = torch.tensor(
                [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence]
                 for sentence in decoder_input]).long().to(get_device())

            # Crear máscaras
            enc_mask, dec_self_mask, dec_cross_mask = create_masks(en_batch, es_batch)

            # Preparar entrada/salida del decoder
            decoder_input = [["[START]"] + sentence[:-1] for sentence in es_batch]

            # Convertir a tensores
            en_tensor = torch.tensor(
                [[en_vocab.get(word, en_vocab["[UNK]"]) for word in sentence] for sentence in en_batch]).to(
                get_device())
            decoder_input_tensor = torch.tensor(
                [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence] for sentence in decoder_input]).to(
                get_device())
            target_tensor = torch.tensor(
                [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence] for sentence in es_batch]).to(
                get_device())

            # Forward pass
            outputs = model(
                en_tensor,
                decoder_input_tensor,
                encoder_self_attention_mask=enc_mask.to(get_device()),
                decoder_self_attention_mask=dec_self_mask.to(get_device()),
                decoder_cross_attention_mask=dec_cross_mask.to(get_device())
            )

            # Calcular pérdida
            loss = criterion(outputs.view(-1, len(es_vocab)), target_tensor.view(-1))
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Mostrar progreso cada 50 batches
            if batch_idx % 50 == 0:
                current_loss = loss.item()
                current_batch = batch_idx * BATCH_SIZE
                print(
                    f'Epoch: {epoch + 1:02} | Batch: {batch_idx:04} | Loss: {current_loss:.3f} | Examples: {current_batch}/{len(train_data)}')

        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for en_batch, es_batch in val_loader:
                enc_mask, dec_self_mask, dec_cross_mask = create_masks(en_batch, es_batch)
                decoder_input = [["[START]"] + sentence[:-1] for sentence in es_batch]

                en_tensor = torch.tensor(
                    [[en_vocab.get(word, en_vocab["[UNK]"]) for word in sentence] for sentence in en_batch]).to(
                    get_device())
                decoder_input_tensor = torch.tensor(
                    [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence] for sentence in decoder_input]).to(
                    get_device())
                target_tensor = torch.tensor(
                    [[es_vocab.get(word, es_vocab["[UNK]"]) for word in sentence] for sentence in es_batch]).to(
                    get_device())

                outputs = model(
                    en_tensor,
                    decoder_input_tensor,
                    encoder_self_attention_mask=enc_mask.to(get_device()),
                    decoder_self_attention_mask=dec_self_mask.to(get_device()),
                    decoder_cross_attention_mask=dec_cross_mask.to(get_device())
                )

                loss = criterion(outputs.view(-1, len(es_vocab)), target_tensor.view(-1))
                val_loss += loss.item()

        # Calcular métricas
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        # Mostrar resultados del epoch
        print(f'\nEpoch: {epoch + 1:02} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}')

        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_epoch{epoch + 1}.pt')
            print(f'New best model saved! Val Loss: {avg_val_loss:.3f}\n')


if __name__ == "__main__":
    # Verificar GPU
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Device: {get_device()}\n")

    # Iniciar entrenamiento
    train_model()