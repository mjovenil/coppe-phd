Visão Geral
===========

Objetivo
--------

Desenvolver um modelo generativo diferenciável para sinais SONAR passivos,
cujos módulos correspondam explicitamente a componentes físicos do sinal:

* **Módulo de fonte** — linhas espectrais LOFAR, blade rate e harmônicas
* **Módulo de canal** — função de transferência acústica (CTD, geometria)
* **Módulo de ruído** — cavitação e ruído de maquinário de propulsão

Mapeamento DDSP → Submarino
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Bloco DDSP
     - Musical
     - Submarino
   * - F0
     - Frequência fundamental
     - Blade rate (frequência de lâmina de hélice)
   * - Loudness
     - Intensidade da nota
     - Nível de pressão sonora (SL)
   * - Z
     - Timbre residual
     - Assinatura espectral residual
   * - Harmonic Audio
     - Sinusoides harmônicas
     - Linhas LOFAR
   * - Filtered Noise
     - Ruído de banda
     - Cavitação / ruído de maquinário
   * - Reverb
     - Resposta ao impulso da sala
     - Função de transferência do canal

Estrutura do Repositório
------------------------

.. code-block:: text

    coppe-phd/
      ddsp/               # Fase 1 — replicação TensorFlow/Magenta
      ddsp_sonar/         # Fase 2 — PyTorch com módulos físicos
        models/
          source.py       # HarmonicOscillator (linhas LOFAR)
          channel.py      # AcousticChannel (CTD/geometria)
          noise.py        # CavitationNoise
          autoencoder.py  # SonarAutoencoder (modelo completo)
        losses/
          spectral.py     # Perdas espectrais (LOFAR/DEMON)
        data/
          provider.py     # TFRecordProvider para sonar
          features.py     # Extração de features acústicas
        training/
          train.py        # Loop de treinamento
          eval.py         # Avaliação e geração de áudio
      docs/               # Esta documentação (Sphinx)
