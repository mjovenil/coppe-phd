"""Módulo de fonte: osciladores harmônicos para linhas LOFAR.

Equivalente submarino do HarmonicAudio do DDSP musical.
A fonte radiante de um navio é caracterizada pela frequência de lâmina
de hélice (blade rate) e suas harmônicas — as linhas LOFAR.
"""

import torch
import torch.nn as nn
import math


class HarmonicOscillator(nn.Module):
    """Banco de osciladores harmônicos para síntese de linhas LOFAR.

    Gera um sinal periódico como soma de sinusoides, condicionado na
    frequência de lâmina de hélice (blade rate) e em amplitudes por
    harmônica variantes no tempo.

    Equivale ao ``HarmonicAudio`` do DDSP (Engel et al., ICLR 2020),
    substituindo a frequência fundamental musical pelo blade rate do navio.

    Args:
        n_harmonics: Número de harmônicas a sintetizar (incluindo a fundamental).
        sample_rate: Taxa de amostragem do sinal de saída em Hz.

    Example:
        >>> osc = HarmonicOscillator(n_harmonics=20, sample_rate=16000)
        >>> blade_rate = torch.tensor([[[5.0]]])   # 5 Hz, shape [B, T, 1]
        >>> amps = torch.ones(1, 125, 20) * 0.05   # shape [B, T, n_harmonics]
        >>> audio = osc(blade_rate, amps)           # shape [B, n_samples]
    """

    def __init__(self, n_harmonics: int = 20, sample_rate: int = 16000):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate

    def forward(
        self,
        blade_rate_hz: torch.Tensor,
        harmonic_amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Sintetiza o sinal de linhas LOFAR.

        Args:
            blade_rate_hz: Frequência de lâmina de hélice em Hz, variante
                no tempo. Shape ``[batch, n_frames, 1]``.
            harmonic_amplitudes: Amplitudes por harmônica, variantes no tempo.
                Shape ``[batch, n_frames, n_harmonics]``. Devem ser não-negativas.

        Returns:
            Sinal de áudio sintetizado. Shape ``[batch, n_samples]``.

        Raises:
            ValueError: Se ``harmonic_amplitudes`` tiver dimensão final diferente
                de ``n_harmonics``.
        """
        if harmonic_amplitudes.shape[-1] != self.n_harmonics:
            raise ValueError(
                f"harmonic_amplitudes.shape[-1] = {harmonic_amplitudes.shape[-1]}, "
                f"esperado {self.n_harmonics}"
            )

        # Índices das harmônicas: [1, 2, 3, ..., n_harmonics]
        harmonic_idx = torch.arange(1, self.n_harmonics + 1,
                                    device=blade_rate_hz.device,
                                    dtype=torch.float32)

        # Frequências de cada harmônica: [batch, n_frames, n_harmonics]
        freqs = blade_rate_hz * harmonic_idx.unsqueeze(0).unsqueeze(0)

        # Upsample de n_frames para n_samples via interpolação linear
        batch, n_frames, _ = freqs.shape
        n_samples = n_frames * (self.sample_rate // 250)  # assume frame_rate=250 Hz

        freqs_up = torch.nn.functional.interpolate(
            freqs.permute(0, 2, 1),  # [B, n_harmonics, n_frames]
            size=n_samples,
            mode='linear',
            align_corners=True,
        ).permute(0, 2, 1)  # [B, n_samples, n_harmonics]

        amps_up = torch.nn.functional.interpolate(
            harmonic_amplitudes.permute(0, 2, 1),
            size=n_samples,
            mode='linear',
            align_corners=True,
        ).permute(0, 2, 1)  # [B, n_samples, n_harmonics]

        # Fase acumulada: integração discreta de f/fs
        phases = torch.cumsum(freqs_up / self.sample_rate, dim=1)

        # Síntese: soma ponderada de sinusoides
        audio = (amps_up * torch.sin(2 * math.pi * phases)).sum(dim=-1)

        return audio  # [B, n_samples]
