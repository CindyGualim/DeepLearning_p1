"""Definiciones de red. Celdas 55 y 93 del notebook, sin cambios.

Los artefactos de `artefactos/` guardan `state_dict`, no la clase, asi que estas
definiciones tienen que coincidir exactamente con las que se usaron para entrenar:
cualquier cambio de nombre de atributo o de dimension rompe `load_state_dict`.
"""
import numpy as np
import torch
from torch import nn

from .config import CARDINALIDADES, DIM_EMBEDDING


class RedRecurrente(nn.Module):
    """Modelo B. GRU de una capa. El estado del ultimo paso resume la historia hasta
    el evento que se esta puntuando."""

    def __init__(self, n_num, oculto=64, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(c + 1, d, padding_idx=0) for c, d in zip(CARDINALIDADES, DIM_EMBEDDING))
        self.gru = nn.GRU(n_num + sum(DIM_EMBEDDING), oculto, batch_first=True)
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(oculto, 1))

    def forward(self, num, cat):
        partes = [num] + [e(cat[:, :, i]) for i, e in enumerate(self.embeddings)]
        salida, _ = self.gru(torch.cat(partes, dim=-1))
        return self.cabeza(salida[:, -1]).squeeze(-1)


class RedConvolucional(nn.Module):
    """Control de arquitectura de B: convoluciones causales dilatadas sobre la misma
    entrada. Ve el orden local pero no arrastra estado a lo largo de la secuencia."""

    def __init__(self, n_num, canales=64, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(c + 1, d, padding_idx=0) for c, d in zip(CARDINALIDADES, DIM_EMBEDDING))
        entrada = n_num + sum(DIM_EMBEDDING)
        self.bloques = nn.Sequential(
            nn.Conv1d(entrada, canales, 3, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(canales, canales, 3, padding=4, dilation=2), nn.ReLU())
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(canales, 1))

    def forward(self, num, cat):
        partes = [num] + [e(cat[:, :, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(partes, dim=-1).transpose(1, 2)
        largo = x.shape[-1]
        h = self.bloques(x)[:, :, :largo]
        return self.cabeza(h[:, :, -1]).squeeze(-1)


class RedHibrida(nn.Module):
    """Modelo C. GRU sobre la secuencia mas el vector de agregadas, unidos en la cabeza."""

    def __init__(self, n_num, n_agg, oculto=128, dropout=0.2, oculto_cabeza=64):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(c + 1, d, padding_idx=0) for c, d in zip(CARDINALIDADES, DIM_EMBEDDING))
        self.gru = nn.GRU(n_num + sum(DIM_EMBEDDING), oculto, batch_first=True)
        self.cabeza = nn.Sequential(
            nn.Linear(oculto + n_agg, oculto_cabeza), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(oculto_cabeza, 1))

    def forward(self, num, cat, agg):
        partes = [num] + [e(cat[:, :, i]) for i, e in enumerate(self.embeddings)]
        salida, _ = self.gru(torch.cat(partes, dim=-1))
        return self.cabeza(torch.cat([salida[:, -1], agg], dim=-1)).squeeze(-1)


ARQUITECTURAS = {"gru": RedRecurrente, "cnn": RedConvolucional}


def tensores(num_z, cat, idx_mat, mascara, filas):
    """(m, L, C) numerico y (m, L, 2) de indices, con el padding en cero."""
    m = mascara[filas]
    num = num_z[idx_mat[filas]] * m[:, :, None]
    codigos = (cat[idx_mat[filas]] + 1) * m[:, :, None]
    return torch.from_numpy(num), torch.from_numpy(codigos.astype(np.int64))
