import os
import argparse
import torch
from data_proccess import load_vocab, process_line, MAX_SEQ_LENGTH
from train import D_MODEL, FFN_HIDDEN, NUM_HEADS, DROPOUT_PROB, NUM_LAYERS
from transformer import Transformer, get_device


def test_saved_model(model_path: str, en_vocab_path: str, es_vocab_path: str, device: torch.device):
    """
    Carga un modelo guardado y permite probar traducciones interactivamente.

    Args:
        model_path (str): Ruta al archivo .pt del modelo.
        en_vocab_path (str): Ruta al vocabulario inglés.
        es_vocab_path (str): Ruta al vocabulario español.
        device (torch.device): Dispositivo (CPU/GPU).
    """
    # Cargar vocabularios
    en_vocab, _ = load_vocab(en_vocab_path)
    es_vocab, reverse_es_vocab = load_vocab(es_vocab_path)

    # Reconstruir modelo
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
    ).to(device)

    # Cargar pesos del modelo
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Bucle interactivo
    while True:
        input_sentence = input("\nIngrese una oración en inglés (o 'salir' para terminar): ").strip()
        if input_sentence.lower() == "salir":
            break

        # Tokenizar entrada
        en_tokens = process_line(input_sentence).split()[:MAX_SEQ_LENGTH]
        en_indices = [en_vocab.get(token, en_vocab["[UNK]"]) for token in en_tokens]

        # Añadir padding
        en_tensor = torch.tensor([en_indices], device=device)
        en_tensor = torch.nn.functional.pad(
            en_tensor,
            (0, MAX_SEQ_LENGTH - len(en_indices)),
            value=en_vocab["[PAD]"]
        )

        # Generar traducción
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
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=-1)

        print(f"\nInput (EN): {input_sentence}")
        print(f"Output (ES): {' '.join(translated_tokens)}")


if __name__ == "__main__":
    # Ejecutar pruebas interactivas
    """
    python test_model.py --model best_model_epoch5.pt --en_vocab data/vocab_en.txt --es_vocab data/vocab_es.txt
    """


    parser = argparse.ArgumentParser(description="Probar modelos guardados")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo .pt")
    parser.add_argument("--en_vocab", type=str, default="data/vocab_en.txt", help="Ruta al vocabulario EN")
    parser.add_argument("--es_vocab", type=str, default="data/vocab_es.txt", help="Ruta al vocabulario ES")

    args = parser.parse_args()

    device = get_device()
    print(f"\nUsando dispositivo: {device}")

    test_saved_model(
        model_path=args.model,
        en_vocab_path=args.en_vocab,
        es_vocab_path=args.es_vocab,
        device=device
    )