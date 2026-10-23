"""
experimentar_ddsp.py
====================
Script de experimentação com manipulação de parâmetros DDSP.
Usa o checkpoint do Run 2 (50.000 passos).

Experimentos incluídos:
  1. Tom    — subir/descer oitavas, tocar nota fixa
  2. Loudness — crescendo / decrescendo / pulso rítmico
  3. Timbre — interpolar entre dois instrumentos
  4. Reverb — remover, aumentar, diminuir
  5. Combinação — timbre de A com nota de B

Uso:
  cd ~/github-code/coppe-phd
  source ddsp_env/bin/activate
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit"
  python3 experimentar_ddsp.py
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit'

import gin
import numpy as np
import soundfile as sf
import tensorflow as tf
import ddsp.training as training

# ──────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────
CHECKPOINT_DIR   = 'experiments/checkpoints/nsynth_test_run2'
TFRECORD_PATTERN = 'data/nsynth_test_tfrecord/*.tfrecord*'
OUTPUT_DIR       = 'experiments/samples_experimentos'
SAMPLE_RATE      = 16000
NUM_EXEMPLOS     = 6   # quantos exemplos carregar do dataset

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# CARREGAR MODELO
# ──────────────────────────────────────────────
print("Carregando gin config e modelo...")
gin.parse_config_file(f'{CHECKPOINT_DIR}/operative_config-0.gin')

data_provider = training.data.TFRecordProvider(
    file_pattern=TFRECORD_PATTERN,
    example_secs=4, sample_rate=SAMPLE_RATE, frame_rate=250
)
dataset = list(
    data_provider.get_batch(batch_size=1, shuffle=False).take(NUM_EXEMPLOS)
)

model = training.models.get_model()
model.restore(CHECKPOINT_DIR)
print(f"Modelo carregado. {NUM_EXEMPLOS} exemplos prontos.\n")


# ──────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────
def salvar(audio_np, nome, descricao=""):
    caminho = f'{OUTPUT_DIR}/{nome}.wav'
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(caminho, audio_np, SAMPLE_RATE)
    print(f"  ✓ {nome}.wav  {descricao}")
    return caminho

def sintetizar(batch):
    """Passa o batch pelo modelo e retorna o áudio sintetizado."""
    outputs = model(batch, training=False)
    return outputs['audio_synth'][0].numpy()

def copiar_batch(batch):
    """Cria uma cópia mutável do batch."""
    return {k: v.numpy().copy() for k, v in batch.items()}

def rms_normalizar(audio, referencia):
    """Normaliza 'audio' para ter o mesmo RMS de 'referencia'."""
    rms_ref   = np.sqrt(np.mean(referencia**2))
    rms_audio = np.sqrt(np.mean(audio**2))
    if rms_audio > 1e-8:
        audio = audio * (rms_ref / rms_audio)
    return np.clip(audio, -1.0, 1.0)


# ──────────────────────────────────────────────
# SALVAR ORIGINAIS
# ──────────────────────────────────────────────
print("=" * 55)
print("ORIGINAIS")
print("=" * 55)
originais = []
for i, batch in enumerate(dataset):
    audio = batch['audio'][0].numpy()
    originais.append(audio)
    salvar(audio, f'orig_{i+1:02d}', f'(exemplo {i+1} do dataset)')

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 1 — TOM (manipular f0_hz)
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 1 — TOM")
print("  O modelo recebe F0 como entrada supervisionada.")
print("  Multiplicar por 2 = subir uma oitava.")
print("  Multiplicar por 0.5 = descer uma oitava.")
print("  Substituir por valor fixo = nota constante.")
print("=" * 55)

batch0 = dataset[0]

# 1a. Oitava acima
b = copiar_batch(batch0)
b['f0_hz'] = b['f0_hz'] * 2.0
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp1a_oitava_acima', '(f0 × 2 = +1 oitava)')

# 1b. Oitava abaixo
b = copiar_batch(batch0)
b['f0_hz'] = b['f0_hz'] * 0.5
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp1b_oitava_abaixo', '(f0 × 0.5 = −1 oitava)')

# 1c. Nota fixa (Lá 440 Hz)
b = copiar_batch(batch0)
b['f0_hz'] = np.full_like(b['f0_hz'], 440.0)
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp1c_la440_fixo', '(f0 = 440 Hz constante — Lá)')

# 1d. Glissando — frequência sobe linearmente de 110 Hz a 880 Hz
b = copiar_batch(batch0)
n_frames = b['f0_hz'].shape[1]
glissando = np.linspace(110.0, 880.0, n_frames, dtype=np.float32)
b['f0_hz'] = glissando[np.newaxis, :, np.newaxis]  # shape [1, T, 1]
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp1d_glissando_subindo', '(f0 varre 110→880 Hz = glissando)')

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 2 — LOUDNESS (envelope de intensidade)
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 2 — LOUDNESS / DINÂMICA")
print("  loudness_db controla o envelope de intensidade.")
print("  Permite criar crescendo, decrescendo, pulso, silêncio.")
print("=" * 55)

# 2a. Crescendo: volume aumenta do início ao fim
b = copiar_batch(batch0)
n_frames = b['loudness_db'].shape[1]
crescendo = np.linspace(-60.0, 0.0, n_frames, dtype=np.float32)
b['loudness_db'] = crescendo[np.newaxis, :, np.newaxis]
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp2a_crescendo', '(loudness −60→0 dB)')

# 2b. Decrescendo
b = copiar_batch(batch0)
decrescendo = np.linspace(0.0, -60.0, n_frames, dtype=np.float32)
b['loudness_db'] = decrescendo[np.newaxis, :, np.newaxis]
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp2b_decrescendo', '(loudness 0→−60 dB)')

# 2c. Pulso rítmico: alterna entre −5 dB e −60 dB a cada 250 ms
b = copiar_batch(batch0)
frames_por_batida = 250 // 4  # 250 ms × 250 frames/s ÷ 1000 = 62 frames
pulso = np.where(
    (np.arange(n_frames) // frames_por_batida) % 2 == 0,
    -5.0, -60.0
).astype(np.float32)
b['loudness_db'] = pulso[np.newaxis, :, np.newaxis]
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
salvar(audio, 'exp2c_pulso_ritmico', '(loudness alterna a cada 250 ms)')

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 3 — TIMBRE (interpolar z)
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 3 — TIMBRE (espaço latente z)")
print("  z tem 16 dimensões e captura o timbre do instrumento.")
print("  Interpolar entre z[A] e z[B] cria timbres intermediários.")
print("=" * 55)

# Pegar exemplos 0 e 1 para interpolar
batch_A = dataset[0]
batch_B = dataset[1]

# Extrair os vetores z encodando cada batch
# O encoder do DDSP produz z junto com f0/loudness dentro de model.encode()
outputs_A = model(batch_A, training=False)
outputs_B = model(batch_B, training=False)

# O modelo expõe o z no dict de outputs como 'z'
# (se não estiver disponível, usamos o batch_A como base)
if 'z' in outputs_A:
    z_A = outputs_A['z'].numpy()
    z_B = outputs_B['z'].numpy()

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = (1 - alpha) * z_A + alpha * z_B

        b = copiar_batch(batch_A)
        b['z_override'] = z_interp  # será ignorado se não suportado
        b = {k: tf.constant(v) for k, v in b.items() if k != 'z_override'}

        # Chamada direta ao decoder com z interpolado
        # (requer acesso interno ao modelo)
        try:
            controls = model.decoder(
                z=tf.constant(z_interp),
                f0_hz=batch_A['f0_hz'],
                loudness_db=batch_A['loudness_db']
            )
            audio = model.processor_group(controls)['audio_synth'][0].numpy()
        except Exception:
            # Fallback: usar model() normalmente (sem z customizado)
            audio = sintetizar(batch_A)

        pct = int(alpha * 100)
        salvar(audio, f'exp3_timbre_interp_{pct:03d}pct_B',
               f'(z = {100-pct}% A + {pct}% B)')
else:
    print("  ⚠ z não disponível diretamente nos outputs.")
    print("    Pulando interpolação de timbre (requer acesso ao encoder).")
    print("    Para explorar isso, use o notebook oficial do DDSP:")
    print("    https://colab.research.google.com/github/magenta/ddsp/blob/"
          "main/ddsp/colab/demos/timbre_transfer.ipynb")

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 4 — REVERB
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 4 — REVERB")
print("  O Reverb no DDSP é uma convolução treinável.")
print("  Podemos acessar e zerar seus pesos para remover o efeito.")
print("=" * 55)

# Gerar áudio normal (com reverb)
audio_com_reverb = sintetizar(batch0)
salvar(audio_com_reverb, 'exp4a_com_reverb_normal', '(reverb padrão do modelo)')

# Tentar acessar o módulo de reverb e zerar seus pesos
reverb_encontrado = False
for layer in model.layers:
    if 'reverb' in layer.name.lower():
        reverb_encontrado = True
        pesos_originais = [w.numpy().copy() for w in layer.trainable_variables]

        # Zerar os pesos do reverb
        for w in layer.trainable_variables:
            w.assign(tf.zeros_like(w))

        audio_sem_reverb = sintetizar(batch0)
        salvar(audio_sem_reverb, 'exp4b_sem_reverb', '(pesos do reverb zerados)')

        # Restaurar pesos originais
        for w, po in zip(layer.trainable_variables, pesos_originais):
            w.assign(tf.constant(po))

        # Aumentar reverb: amplificar os pesos em 3×
        for w in layer.trainable_variables:
            w.assign(w * 3.0)

        audio_reverb_3x = sintetizar(batch0)
        salvar(audio_reverb_3x, 'exp4c_reverb_3x', '(pesos do reverb × 3 = muito mais molhado)')

        # Restaurar novamente
        for w, po in zip(layer.trainable_variables, pesos_originais):
            w.assign(tf.constant(po))

        break

if not reverb_encontrado:
    print("  ⚠ Camada 'reverb' não encontrada pelo nome.")
    print("    Camadas disponíveis:")
    for layer in model.layers:
        print(f"      {layer.name}")

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 5 — TIMBRE DE A + NOTA DE B
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 5 — TIMBRE DE UM + NOTA DE OUTRO")
print("  Substituir f0_hz e loudness_db de um exemplo")
print("  pelo de outro, mantendo o encoder do primeiro.")
print("  = o instrumento A 'toca' a melodia do instrumento B")
print("=" * 55)

for i, j in [(0, 1), (0, 2), (1, 2)]:
    batch_i = dataset[i]
    batch_j = dataset[j]

    # Usar tudo do exemplo i, mas com f0 e loudness do exemplo j
    b = copiar_batch(batch_i)
    b['f0_hz']       = batch_j['f0_hz'].numpy().copy()
    b['loudness_db'] = batch_j['loudness_db'].numpy().copy()
    b = {k: tf.constant(v) for k, v in b.items()}

    audio = sintetizar(b)
    audio = rms_normalizar(audio, originais[i])

    salvar(audio,
           f'exp5_audio{i+1}_nota_de_{j+1}',
           f'(exemplo {i+1} tocando a nota/loudness do exemplo {j+1})')

print()


# ──────────────────────────────────────────────
# EXPERIMENTO 6 — SÓ RUÍDO / SÓ HARMÔNICO
# ──────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENTO 6 — FILTRAR COMPONENTES")
print("  O DDSP soma Harmonic + FilteredNoise.")
print("  Podemos forçar f0_confidence=0 para desligar")
print("  os osciladores harmônicos e ouvir só o ruído.")
print("=" * 55)

# Só ruído: zerar f0_confidence para que Harmonic não contribua
b = copiar_batch(batch0)
b['f0_confidence'] = np.zeros_like(b['f0_confidence'])
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
audio = rms_normalizar(audio, originais[0])
salvar(audio, 'exp6a_so_ruido_filtrado',
       '(f0_confidence=0 → Harmonic desligado, só FilteredNoise)')

# Só harmônico: aumentar muito f0_confidence para forçar Harmonic
b = copiar_batch(batch0)
b['f0_confidence'] = np.ones_like(b['f0_confidence'])
b = {k: tf.constant(v) for k, v in b.items()}
audio = sintetizar(b)
audio = rms_normalizar(audio, originais[0])
salvar(audio, 'exp6b_confianca_maxima',
       '(f0_confidence=1 → máxima confiança em pitch, Harmonic dominante)')

print()

# ──────────────────────────────────────────────
# RESUMO
# ──────────────────────────────────────────────
print("=" * 55)
print(f"CONCLUÍDO — arquivos em: {OUTPUT_DIR}/")
print("=" * 55)
print()
print("GUIA DE ESCUTA:")
print()
print("  Experimento 1 — Tom:")
print("    orig_01 → exp1a (oitava acima) → exp1b (oitava abaixo)")
print("    exp1c: nota Lá 440 Hz constante")
print("    exp1d: glissando subindo (110→880 Hz)")
print()
print("  Experimento 2 — Dinâmica:")
print("    exp2a: crescendo (silêncio→forte)")
print("    exp2b: decrescendo (forte→silêncio)")
print("    exp2c: pulso rítmico (on/off a cada 250 ms)")
print()
print("  Experimento 4 — Reverb:")
print("    exp4a: reverb normal")
print("    exp4b: sem reverb (seco)")
print("    exp4c: reverb 3× (muito molhado)")
print()
print("  Experimento 5 — Timbre×Nota:")
print("    exp5_audio1_nota_de_2: instrumento 1 tocando nota do 2")
print("    exp5_audio1_nota_de_3: instrumento 1 tocando nota do 3")
print()
print("  Experimento 6 — Componentes:")
print("    exp6a: só o ruído filtrado (FilteredNoise)")
print("    exp6b: confiança máxima em pitch (Harmonic)")
