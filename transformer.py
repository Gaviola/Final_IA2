import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNormalization(nn.Module):
    """
    Implementa la capa de normalización.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Inicializa la capa de normalización.

        Args:
            d_model (int): Dimensión del modelo.
            eps (float): Pequeño valor para evitar la división por cero. Por defecto es 1e-6.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la normalización a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor normalizado.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """
    Implementa la codificación posicional para el modelo Transformer.
    """
    def __init__(self, d_model: int, max_len: int, dropout: float):
        """
        Inicializa la codificación posicional.

        Args:
            d_model (int): Dimensión del modelo.
            max_len (int): Longitud máxima de las secuencias de entrada.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 1 / (10000 ^ (2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # Pares
        pe[:, 1::2] = torch.cos(position * div_term) # Impares
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la codificación posicional a la entrada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida con la codificación posicional aplicada.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SentenceEmbedding(nn.Module):
    """
    Clase que representa la transformacion de sentencias tokenizadas en Embeddings.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float):
        """
        Inicializa la clase SentenceEmbedding.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            d_model (int): Dimensión del modelo.
            max_len (int): Longitud máxima de las secuencias de entrada.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante de la incrustación de oraciones.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida con la incrustación y codificación posicional aplicadas.
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Implementa la atención multi-cabeza.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        """
        Inicializa la atención multi-cabeza.

        Args:
            d_model (int): Dimensión del modelo.
            num_heads (int): Número de cabezas de atención.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Propagación hacia adelante de la atención multi-cabeza.

        Args:
            q (torch.Tensor): Tensor de Querys.
            k (torch.Tensor): Tensor de Keys.
            v (torch.Tensor): Tensor de Values.
            mask (torch.Tensor, opcional): Máscara de atención.

        Returns:
            torch.Tensor: Salida de la atención multi-cabeza.
        """
        batch_size = q.size(0)

        # Dividimos d_model en num_heads y d_k
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # Formula de atención
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9) # -inf para que no se considere en el softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Inicializa una capa de red neuronal feed-forward.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Args:
            d_model (int): Dimensión del modelo.
            d_ff (int): Dimensión de la capa feed-forward.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante de la capa feed-forward.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida después de aplicar la capa feed-forward.
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class EncoderLayer(nn.Module):
    """
    Clase que representa una capa del codificador en el modelo Transformer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Inicializa una capa del codificador.

        Args:
            d_model (int): Dimensión del modelo.
            num_heads (int): Número de cabezas en la atención multi-cabeza.
            d_ff (int): Dimensión de la capa de feed-forward.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante de la capa del codificador.

        Args:
            x (torch.Tensor): Tensor de entrada.
            mask (torch.Tensor): Máscara de atención para paddings.

        Returns:
            torch.Tensor: Salida de la capa del codificador.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x


class DecoderLayer(nn.Module):
    """
    Inicializa una capa del decodificador.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Args:
            d_model (int): Dimensión del modelo.
            num_heads (int): Número de cabezas en la atención multi-cabeza.
            d_ff (int): Dimensión de la capa de feed-forward.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Propagación hacia adelante de la capa del decodificador.

        Args:
            x (torch.Tensor): Tensor de entrada del objetivo.
            encoder_output (torch.Tensor): Salida del codificador.
            src_mask (torch.Tensor): Máscara del lenguaje fuente.
            tgt_mask (torch.Tensor): Máscara del lenguaje objetivo.

        Returns:
            torch.Tensor: Salida de la capa del decodificador.
        """
        attn_output = self.self_attn(x, x, x, tgt_mask) # k, q, v, mask
        x = self.norm1(x + self.dropout1(attn_output))
        attn_output = self.encoder_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x


class Encoder(nn.Module):
    """
    Clase Encoder que implementa la parte codificadora del modelo Transformer.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int,
                 dropout: float
                 ):
        """
        Inicializa un codificador.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            d_model (int): Dimensión del modelo.
            num_layers (int): Número de capas en el codificador.
            num_heads (int): Número de cabezas en la atención multi-cabeza.
            d_ff (int): Dimensión de la capa de feed-forward.
            max_len (int): Longitud máxima de las secuencias de entrada.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.embedding = SentenceEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Propagación hacia adelante del codificador.

        Args:
            x (torch.Tensor): Tensor de entrada.
            mask (torch.Tensor): Máscara de atención para paddings.

        Returns:
            torch.Tensor: Salida del codificador.
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """
    Clase Decoder que implementa la parte decodificadora del modelo Transformer.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 max_len: int,
                 dropout: float
                 ):
        """
        Inicializa un decodificador.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            d_model (int): Dimensión del modelo.
            num_layers (int): Número de capas en el decodificador.
            num_heads (int): Número de cabezas en la atención multi-cabeza.
            d_ff (int): Dimensión de la capa de feed-forward.
            max_len (int): Longitud máxima de las secuencias de entrada.
            dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.embedding = SentenceEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Propagación hacia adelante del decodificador.

        Args:
            x (torch.Tensor): Tensor de entrada del objetivo.
            encoder_output (torch.Tensor): Salida del codificador.
            src_mask (torch.Tensor): Máscara del lenguaje fuente.
            tgt_mask (torch.Tensor): Máscara del lenguaje objetivo.

        Returns:
            torch.Tensor: Salida del decodificador.
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """
       Clase Transformer que implementa un modelo de Transformer completo con un codificador y un decodificador.
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_layers: int =6,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 max_len: int = 100,
                 dropout: float = 0.1
                 ):
        """
        Inicializa un modelo Transformer.

        Args:
        src_vocab_size (int): Tamaño del vocabulario de la fuente.
        tgt_vocab_size (int): Tamaño del vocabulario del objetivo.
        d_model (int): Dimensión del modelo.
        num_layers (int): Número de capas en el codificador y decodificador.
        num_heads (int): Número de cabezas en la atención multi-cabeza.
        d_ff (int): Dimensión de la capa de feed-forward.
        max_len (int): Longitud máxima de las secuencias de entrada.
        dropout (float): Tasa de dropout.
        """
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo Transformer.

        Args:
            src (torch.Tensor): Tensor de entrada de la fuente.
            tgt (torch.Tensor): Tensor de entrada del objetivo.
            src_mask (torch.Tensor): Máscara de la fuente.
            tgt_mask (torch.Tensor): Máscara del objetivo.

        Returns:
            torch.Tensor: Salida del modelo Transformer.
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)
