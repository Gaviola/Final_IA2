import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from data_proccess import TextDS, process_line


MAX_SEQ_LENGTH = 200
NEG_INF = -1e10

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PositionalEncoding(nn.Module):
    """
    Esta clase genera la codificación posicional para cada token en la secuencia.
    """

    def __init__(self, d_model: int, max_sequence_length: int):
        """
        Inicializa la clase PositionalEncoding.

        Args:
            d_model (int): La dimensión del modelo.
            max_sequence_length (int): La longitud máxima de la secuencia
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        """
        Calcula la codificación posicional para la secuencia.

        Retorna:
            torch.Tensor: El tensor de codificación posicional.
        """
        even_i = torch.arange(0, self.d_model, 2).float()  # Get even indices
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_pos_encode = torch.sin(position / denominator)
        odd_pos_encode = torch.cos(position / denominator)
        stacked = torch.stack([even_pos_encode, odd_pos_encode], dim=2)
        pos_encode = torch.flatten(stacked, start_dim=1, end_dim=2)
        return pos_encode


class SentenceEmbedding(nn.Module):
    """
    Clase para generar embeddings de oraciones con codificación posicional.
    """

    def __init__(self, max_sequence_length: int, d_model: int, language_to_index: dict):
        """
        Inicializa la clase SentenceEmbedding.

        Args:
            max_sequence_length (int): Longitud máxima de la secuencia.
            d_model (int): Dimensión del modelo.
            language_to_index (dict): Diccionario que mapea palabras a índices.
        """
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = "[START]"
        self.END_TOKEN = "[EOS]"
        self.PADDING_TOKEN = "[PAD]"
        self.UNKNOWN_TOKEN = "[UNK]"

    def batch_tokenize(self, batch: list, start_token=True, end_token=True) -> torch.Tensor:
        """
        Tokeniza un lote de oraciones.

        Args:
            batch (torch.Tensor): Lote de oraciones a tokenizar.
            start_token (bool): Si se debe agregar el token de inicio.
            end_token (bool): Si se debe agregar el token de fin.

        Returns:
            torch.Tensor: Tensor de oraciones tokenizadas.
        """

        def tokenize(sentence: list[str], start_token: bool=True, end_token: bool =True):
            """
            Tokeniza una oración individual.

            Args:
                sentence (list[str]): Oración a tokenizar.
                start_token (bool): Si se debe agregar el token de inicio.
                end_token (bool): Si se debe agregar el token de fin.

            Returns:
                torch.Tensor: Tensor de índices de palabras.
            """
            indices = []
            if start_token:
                indices.append(self.language_to_index[self.START_TOKEN])
            indices.extend([self.language_to_index.get(word, self.language_to_index[self.UNKNOWN_TOKEN])
                            for word in sentence])
            if end_token:
                indices.append(self.language_to_index[self.END_TOKEN])
            return torch.tensor(indices[:self.max_sequence_length])

        tokenized = [tokenize(sentence) for sentence in batch]
        return torch.stack(tokenized).to(get_device())

    def forward(self, x, start_token, end_token) -> torch.Tensor:
        """
        Calcula los embeddings de las oraciones con codificación posicional.

        Args:
            x (torch.tensor): Lote de oraciones.
            start_token (bool): Si se debe agregar el token de inicio.
            end_token (bool): Si se debe agregar el token de fin.

        Returns:
            torch.Tensor: Tensor de embeddings de oraciones.
        """
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder()[:x.size(1), :].to(get_device())
        x = self.dropout(x + pos)
        return x


def scaled_dot_product(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       mask: Optional[torch.Tensor] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula el producto escalar escalado de los tensores de query, key y value.

    Args:
        q (torch.Tensor): Tensor de query.
        k (torch.Tensor): Tensor de key.
        v (torch.Tensor): Tensor de value.
        mask (Optional[torch.Tensor]): Tensor de máscara opcional para aplicar a la atención.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Un par de tensores que representan los value y la atención.
    """
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    """
    Clase que implementa la atención de múltiples cabezas.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Inicializa la clase MultiHeadAttention.

        Args:
            d_model (int): Dimensión del modelo.
            num_heads (int): Número de cabezas de atención.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer: torch.nn.Linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Realiza el paso hacia adelante de la atención de múltiples cabezas.

        Args:
            x (torch.Tensor): Tensor de entrada.
            mask (Optional[torch.Tensor]): Tensor de máscara opcional para aplicar a la atención.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la atención de múltiples cabezas.
        """
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Clase que implementa la atención cruzada de múltiples cabezas.
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Inicializa la clase MultiHeadCrossAttention.

        Args:
            d_model (int): Dimensión del modelo.
            num_heads (int): Número de cabezas de atención.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la atención cruzada de múltiples cabezas.

        Args:
            x (torch.Tensor): Tensor de entrada para key y value.
            y (torch.Tensor): Tensor de entrada para query.
            mask (Optional[torch.Tensor]): Tensor de máscara opcional para aplicar a la atención.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la atención cruzada de múltiples cabezas.
        """
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    """
    Clase que implementa la normalización por capas.
    """

    def __init__(self, parameters_shape: list[int], eps: float = 1e-5):
        """
        Inicializa la clase LayerNormalization.

        Args:
            parameters_shape (list[int]): La forma de los parámetros gamma y beta.
            eps (float): Un pequeño valor para evitar la división por cero.
        """
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la normalización por capas.

        Args:
            inputs (torch.Tensor): El tensor de entrada a normalizar.

        Returns:
            torch.Tensor: El tensor normalizado.
        """
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    """
    Clase que implementa una red neuronal feed-forward aplicada de manera posicional.
    """

    def __init__(self, d_model: int, hidden: int, drop_prob: float=0.1):
        """
        Inicializa la clase PositionwiseFeedForward.

        Args:
            d_model (int): Dimensión del modelo.
            hidden (int): Dimensión de la capa oculta.
            drop_prob (float): Probabilidad de dropout.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la red feed-forward.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la red feed-forward.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Clase que implementa una capa del encoder en el modelo Transformer.
    """

    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """
        Inicializa la clase EncoderLayer.

        Args:
            d_model (int): Dimensión del modelo.
            ffn_hidden (int): Dimensión de la capa oculta en la red feed-forward.
            num_heads (int): Número de cabezas de atención.
            drop_prob (float): Probabilidad de dropout.
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor, self_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la capa del encoder.

        Args:
            x (torch.Tensor): Tensor de entrada.
            self_attention_mask: Máscara de atención propia.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la capa del encoder.
        """
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialEncoder(nn.Sequential):
    """
    Clase que implementa un encoder secuencial que aplica múltiples capas de encoder.
    """

    def forward(self, *inputs):
        """
        Realiza el paso hacia adelante del encoder secuencial.

        Args:
            inputs: Una tupla que contiene el tensor de entrada y la máscara de atención propia.

        Returns:
            torch.Tensor: El tensor de salida después de aplicar las capas del encoder.
        """
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    """
    Clase que implementa el encoder del modelo Transformer.
    """

    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 num_heads: int,
                 drop_prob: float,
                 num_layers: int,
                 max_sequence_length: int,
                 language_to_index: dict):
        """
        Inicializa la clase Encoder.

        Args:
            d_model (int): Dimensión del modelo.
            ffn_hidden (int): Dimensión de la capa oculta en la red feed-forward.
            num_heads (int): Número de cabezas de atención.
            drop_prob (float): Probabilidad de dropout.
            num_layers (int): Número de capas del encoder.
            max_sequence_length (int): Longitud máxima de la secuencia.
            language_to_index (dict): Diccionario que mapea palabras a índices.
        """
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self,
                x: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor],
                start_token: bool,
                end_token: bool
                ) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del encoder.

        Args:
            x (torch.Tensor): Tensor de entrada.
            self_attention_mask (torch.Tensor): Máscara de atención propia.
            start_token (bool): Token de inicio.
            end_token (bool): Token de fin.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar las capas del encoder.
        """
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class DecoderLayer(nn.Module):
    """
    Clase que implementa una capa del decoder en el modelo Transformer.
    """

    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """
        Inicializa la clase DecoderLayer.

        Args:
            d_model (int): Dimensión del modelo.
            ffn_hidden (int): Dimensión de la capa oculta en la red feed-forward.
            num_heads (int): Número de cabezas de atención.
            drop_prob (float): Probabilidad de dropout.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor]=None,
                cross_attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Realiza el paso hacia adelante de la capa del decoder.

        Args:
            x (torch.Tensor): Tensor de entrada para key y value.
            y (torch.Tensor): Tensor de entrada para query.
            self_attention_mask (Optional[torch.Tensor]): Tensor de máscara opcional para aplicar a la atención propia.
            cross_attention_mask (Optional[torch.Tensor]): Tensor de máscara opcional para aplicar a la atención cruzada.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la capa del decoder.
        """
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    """
    Clase que implementa un decoder secuencial que aplica múltiples capas de decoder.
    """

    def forward(self, *inputs):
        """
        Realiza el paso hacia adelante del decoder secuencial.

        Args:
            inputs: Una tupla que contiene el tensor de entrada x, el tensor de entrada y,
                    la máscara de atención propia y la máscara de atención cruzada.

        Returns:
            torch.Tensor: El tensor de salida después de aplicar las capas del decoder.
        """
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)  # 30 x 200 x 512
        return y


class Decoder(nn.Module):
    """
    Clase que implementa el decoder del modelo Transformer.
    """

    def __init__(self, d_model: int,
                 ffn_hidden: int,
                 num_heads: int,
                 drop_prob: float,
                 num_layers: int,
                 max_sequence_length: int,
                 language_to_index: dict):
        """
        Inicializa la clase Decoder.

        Args:
            d_model (int): Dimensión del modelo.
            ffn_hidden (int): Dimensión de la capa oculta en la red feed-forward.
            num_heads (int): Número de cabezas de atención.
            drop_prob (float): Probabilidad de dropout.
            num_layers (int): Número de capas del decoder.
            max_sequence_length (int): Longitud máxima de la secuencia.
            language_to_index (dict): Diccionario que mapea palabras a índices.
        """
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index)
        self.layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor],
                cross_attention_mask: Optional[torch.Tensor],
                start_token: bool,
                end_token: bool
                ) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del decoder.

        Args:
            x (torch.Tensor): Tensor de entrada para el encoder.
            y (torch.Tensor): Tensor de entrada para el decoder.
            self_attention_mask (Optional[torch.Tensor]): Máscara de atención propia.
            cross_attention_mask (Optional[torch.Tensor]): Máscara de atención cruzada.
            start_token (bool): Si se debe agregar el token de inicio.
            end_token (bool): Si se debe agregar el token de fin.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar las capas del decoder.
        """
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    """
    Clase que implementa el modelo Transformer.
    """

    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 num_heads: int,
                 drop_prob: float,
                 num_layers: int,
                 max_sequence_length: int,
                 es_vocab_size: int,
                 english_to_index: dict,
                 spanish_to_index: dict):
        """
        Inicializa la clase Transformer.

        Args:
            d_model (int): Dimensión del modelo.
            ffn_hidden (int): Dimensión de la capa oculta en la red feed-forward.
            num_heads (int): Número de cabezas de atención.
            drop_prob (float): Probabilidad de dropout.
            num_layers (int): Número de capas del encoder y decoder.
            max_sequence_length (int): Longitud máxima de la secuencia.
            es_vocab_size (int): Tamaño del vocabulario en español.
            english_to_index (dict): Diccionario que mapea palabras en inglés a índices.
            spanish_to_index (dict): Diccionario que mapea palabras en español a índices.
        """
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, spanish_to_index)
        self.linear = nn.Linear(d_model, es_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                encoder_self_attention_mask: Optional[torch.Tensor] = None,
                decoder_self_attention_mask: Optional[torch.Tensor] = None,
                decoder_cross_attention_mask: Optional[torch.Tensor] = None,
                enc_start_token: bool = False,
                enc_end_token: bool = False,
                dec_start_token: bool = True,
                dec_end_token: bool = True) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del modelo Transformer.

        Args:
            x (torch.Tensor): Tensor de entrada para el encoder.
            y (torch.Tensor): Tensor de entrada para el decoder.
            encoder_self_attention_mask (Optional[torch.Tensor]): Máscara de atención propia para el encoder.
            decoder_self_attention_mask (Optional[torch.Tensor]): Máscara de atención propia para el decoder.
            decoder_cross_attention_mask (Optional[torch.Tensor]): Máscara de atención cruzada para el decoder.
            enc_start_token (bool): Si se debe agregar el token de inicio en el encoder.
            enc_end_token (bool): Si se debe agregar el token de fin en el encoder.
            dec_start_token (bool): Si se debe agregar el token de inicio en el decoder.
            dec_end_token (bool): Si se debe agregar el token de fin en el decoder.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar el modelo Transformer.
        """
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out