import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data_proccess import TextDS, load_vocab, create_masks, collate_fn, MAX_SEQ_LENGTH, load_data_set, process_line
from transformer import Transformer, get_device
import time

# Configuración hiperparámetros
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 3e-4
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 3
FFN_HIDDEN = 1024
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


def translate_examples(model, en_vocab, es_vocab, device, examples):
    """Traduce oraciones de ejemplo usando el modelo entrenado."""
    model.eval()
    reverse_es_vocab = {idx: word for word, idx in es_vocab.items()}

    for en_sentence in examples:
        # Preprocesar y tokenizar
        en_tokens = process_line(en_sentence).split()
        en_tokens = [en_vocab.get(word, en_vocab["[UNK]"]) for word in en_tokens][:MAX_SEQ_LENGTH]

        # Convertir a tensor y añadir padding
        en_tensor = torch.tensor([en_tokens], device=device)
        en_tensor = torch.nn.functional.pad(
            en_tensor,
            (0, MAX_SEQ_LENGTH - len(en_tokens)),
            value=en_vocab["[PAD]"]
        )

        # Generar traducción autoregresiva
        decoder_input = torch.tensor([[es_vocab["[START]"]]], device=device)
        translated_tokens = []

        with torch.no_grad():
            encoder_output = model.encoder(en_tensor, None, start_token=False, end_token=False)

            for _ in range(MAX_SEQ_LENGTH):
                decoder_output = model.decoder(
                    encoder_output,
                    decoder_input,
                    None,
                    None,
                    start_token=False,
                    end_token=False
                )
                next_token = torch.argmax(decoder_output[:, -1, :], dim=-1)
                decoded_word = reverse_es_vocab[next_token.item()]

                if decoded_word == "[EOS]":
                    break

                translated_tokens.append(decoded_word)
                decoder_input = torch.cat(
                    [decoder_input, next_token.unsqueeze(0)],
                    dim=-1
                )

        print(f"\nInput (EN): {en_sentence}")
        print(f"Output (ES): {' '.join(translated_tokens)}")



# Función de entrenamiento
def train_model():
    train_data, val_data, en_vocab, es_vocab = prepare_data()
    scaler = torch.GradScaler()  # Inicializa al inicio

    # Crear DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = initialize_model(en_vocab, es_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=es_vocab["[PAD]"])

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0

        for batch_idx, (en_batch, es_batch) in enumerate(train_loader):
            optimizer.zero_grad()

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

            # Forward pass
            with torch.autocast(device_type=get_device().type):

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
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Mostrar progreso cada 50 batches
            if batch_idx % 50 == 0:
                current_loss = loss.item()
                current_batch = batch_idx * BATCH_SIZE
                print(
                    f'Epoch: {epoch + 1:02}'
                    f' | Batch: {batch_idx:04}'
                    f' | Loss: {current_loss:.3f}'
                    f' | Examples: {current_batch}/{len(train_data)}'
                    f' | Time: {time.time() - start_time:.2f}s')

            torch.cuda.empty_cache()

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
            torch.save(model.state_dict(), f'models/best_model_epoch{epoch + 1}.pt')
            print(f'New best model saved! Val Loss: {avg_val_loss:.3f}\n')
"""
        print("\nTesting translations...")
        test_examples = [
            "Hello, how are you?",
            "The weather is nice today.",
            "Where is the nearest hospital?",
            "This is a sample sentence for testing."
        ]

        translate_examples(
            model,
            en_vocab,
            es_vocab,
            get_device(),
            test_examples
        )
"""

if __name__ == "__main__":
    # Verificar GPU
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Device: {get_device()}\n")

    # Iniciar entrenamiento
    train_model()