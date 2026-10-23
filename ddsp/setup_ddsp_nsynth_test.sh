#!/usr/bin/env bash
# =============================================================================
# setup_ddsp_nsynth_test.sh
#
# Pipeline completo para validar o DDSP com o split "test" do NSynth (~4 GB),
# sem precisar baixar os 73 GB do dataset completo via TFDS.
#
# Etapas:
#   1. Ativa o ambiente virtual já existente (ddsp_env)
#   2. Baixa nsynth-test.jsonwav.tar.gz (~4 GB) do Magenta
#   3. Extrai os WAVs
#   4. Converte para TFRecord com F0/loudness via ddsp_prepare_tfrecord
#   5. Executa treinamento com solo_instrument.gin + tfrecord.gin
#
# Uso (rodar de qualquer lugar):
#   chmod +x setup_ddsp_nsynth_test.sh
#   ./setup_ddsp_nsynth_test.sh
#
# Estrutura esperada (já existente em sua máquina):
#   ~/github-code/coppe-phd/
#     ddsp/        ← repositório clonado
#     ddsp_env/    ← virtualenv com ddsp instalado
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuração de caminhos — raiz do projeto de doutorado
# ---------------------------------------------------------------------------
WORKDIR="${HOME}/github-code/coppe-phd"
DDSP_REPO="${WORKDIR}/ddsp"
VENV_DIR="${WORKDIR}/ddsp_env"
NSYNTH_DIR="${WORKDIR}/data/nsynth_test"
TFRECORD_DIR="${WORKDIR}/data/nsynth_test_tfrecord"
CHECKPOINT_DIR="${WORKDIR}/experiments/checkpoints/nsynth_test_run1"

NSYNTH_URL="https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
NSYNTH_ARCHIVE="${WORKDIR}/data/nsynth-test.jsonwav.tar.gz"

echo ""
echo "============================================================"
echo "  DDSP — Setup com NSynth Test Split (~4 GB)"
echo "============================================================"
echo "  WORKDIR      : ${WORKDIR}"
echo "  TFRECORDS    : ${TFRECORD_DIR}"
echo "  CHECKPOINTS  : ${CHECKPOINT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Preparar diretórios e ativar o ambiente virtual existente
# ---------------------------------------------------------------------------
mkdir -p "${NSYNTH_DIR}" "${TFRECORD_DIR}" "${CHECKPOINT_DIR}" \
         "${WORKDIR}/data" "${WORKDIR}/experiments/checkpoints"

# Verificar que o WORKDIR existe
if [ ! -d "${WORKDIR}" ]; then
    echo "[ERRO] Diretório base não encontrado: ${WORKDIR}"
    echo "       Verifique se você está na máquina certa (Hess)."
    exit 1
fi

# Ativar o ambiente virtual já existente
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[ERRO] Ambiente virtual não encontrado em: ${VENV_DIR}"
    echo "       Esperado: ${WORKDIR}/ddsp_env/"
    exit 1
fi

source "${VENV_DIR}/bin/activate"
echo "[1/5] Ambiente virtual ativado: ${VENV_DIR}"

# Verificar se ddsp_prepare_tfrecord está disponível
if ! command -v ddsp_prepare_tfrecord &>/dev/null; then
    echo "      ddsp_prepare_tfrecord não encontrado — instalando extras..."
    pip install "ddsp[data_preparation]" --quiet
fi

echo "      DDSP: $(pip show ddsp 2>/dev/null | grep Version || echo 'não encontrado')"
echo "      ddsp_prepare_tfrecord: $(which ddsp_prepare_tfrecord)"

# ---------------------------------------------------------------------------
# 3. Baixar o NSynth test split
# ---------------------------------------------------------------------------
if [ ! -f "${NSYNTH_ARCHIVE}" ]; then
    echo ""
    echo "[2/5] Baixando NSynth test split (~4 GB)..."
    echo "      URL: ${NSYNTH_URL}"
    wget --progress=bar:force --no-check-certificate \
         --continue \
         -O "${NSYNTH_ARCHIVE}" \
         "${NSYNTH_URL}"
else
    echo "[2/5] Arquivo já baixado: ${NSYNTH_ARCHIVE}"
fi

# ---------------------------------------------------------------------------
# 4. Extrair os WAVs
# ---------------------------------------------------------------------------
WAV_DIR="${NSYNTH_DIR}/audio"
if [ ! -d "${WAV_DIR}" ]; then
    echo ""
    echo "[3/5] Extraindo arquivos WAV..."
    tar -xzf "${NSYNTH_ARCHIVE}" -C "${NSYNTH_DIR}" --strip-components=1
    echo "      Extração concluída."
    echo "      Total de arquivos WAV: $(ls "${WAV_DIR}"/*.wav 2>/dev/null | wc -l)"
else
    echo "[3/5] WAVs já extraídos: ${WAV_DIR}"
    echo "      Total de arquivos WAV: $(ls "${WAV_DIR}"/*.wav 2>/dev/null | wc -l)"
fi

# Para um teste rápido do pipeline, usar apenas os primeiros 500 WAVs
# (o test set completo tem ~4096 exemplos — remova o limite para usar todos)
SAMPLE_DIR="${WORKDIR}/nsynth_test_500"
if [ ! -d "${SAMPLE_DIR}" ]; then
    echo ""
    echo "      Copiando 500 WAVs para teste rápido do pipeline..."
    mkdir -p "${SAMPLE_DIR}"
    ls "${WAV_DIR}"/*.wav | head -500 | xargs -I{} cp {} "${SAMPLE_DIR}/"
    echo "      Pronto: $(ls "${SAMPLE_DIR}"/*.wav | wc -l) arquivos em ${SAMPLE_DIR}"
fi

# ---------------------------------------------------------------------------
# 5. Converter WAVs para TFRecord com F0 (CREPE) e Loudness
# ---------------------------------------------------------------------------
TFRECORD_PATTERN="${TFRECORD_DIR}/nsynth_test.tfrecord"

if ls "${TFRECORD_DIR}"/nsynth_test.tfrecord* 1>/dev/null 2>&1; then
    echo ""
    echo "[4/5] TFRecords já existem:"
    ls -lh "${TFRECORD_DIR}"/nsynth_test.tfrecord* | awk '{print "      " $5, $9}'
else
    echo ""
    echo "[4/5] Convertendo WAVs para TFRecord..."
    echo "      Isso pode levar vários minutos (CREPE extrai F0 arquivo a arquivo)."
    echo "      Usando 500 WAVs de amostra..."

    ddsp_prepare_tfrecord \
        --input_audio_filepatterns="${SAMPLE_DIR}/*.wav" \
        --output_tfrecord_path="${TFRECORD_PATTERN}" \
        --num_shards=10 \
        --sample_rate=16000 \
        --frame_rate=250 \
        --alsologtostderr

    echo ""
    echo "      TFRecords gerados:"
    ls -lh "${TFRECORD_DIR}"/nsynth_test.tfrecord* | awk '{print "      " $5, $9}'
fi

# ---------------------------------------------------------------------------
# 6. Verificar o conteúdo de um TFRecord (opcional — primeiros 2 exemplos)
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Verificando conteúdo do TFRecord..."
python3 - <<'PYEOF'
import tensorflow as tf, sys

TFRECORD_DIR = __import__('os').environ.get('TFRECORD_DIR',
    __import__('os').path.expanduser('~/doutorado/ddsp_experimento/nsynth_test_tfrecord'))

import glob
files = sorted(glob.glob(f"{TFRECORD_DIR}/nsynth_test.tfrecord*"))
if not files:
    print("  Nenhum TFRecord encontrado.")
    sys.exit(0)

ds = tf.data.TFRecordDataset(files[:1])
for i, raw in enumerate(ds.take(2)):
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    print(f"\n  Exemplo {i+1}:")
    for key, feat in sorted(ex.features.feature.items()):
        kind = feat.WhichOneof('kind')
        if kind == 'float_list':
            vals = feat.float_list.value
            print(f"    {key:20s}  float32  shape=({len(vals)},)  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}")
        elif kind == 'int64_list':
            vals = feat.int64_list.value
            print(f"    {key:20s}  int64    shape=({len(vals)},)  vals={list(vals)[:5]}")
        elif kind == 'bytes_list':
            vals = feat.bytes_list.value
            print(f"    {key:20s}  bytes    n={len(vals)}")
PYEOF

# ---------------------------------------------------------------------------
# 7. Executar treinamento DDSP
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Iniciando treinamento DDSP..."
echo "      gin: solo_instrument.gin + tfrecord.gin + basic_f0_ld.gin"
echo "      checkpoint: ${CHECKPOINT_DIR}"
echo ""
echo "      Para parar: Ctrl+C"
echo "      Logs serão exibidos abaixo."
echo ""

# Localizar os arquivos gin instalados pelo pacote ddsp
GIN_BASE=$(python3 -c "import ddsp; import os; print(os.path.join(os.path.dirname(ddsp.__file__), 'training', 'gin'))")
echo "      Gin base: ${GIN_BASE}"

ddsp_run \
    --mode=train \
    --save_dir="${CHECKPOINT_DIR}" \
    --gin_file="${GIN_BASE}/models/solo_instrument.gin" \
    --gin_file="${GIN_BASE}/datasets/tfrecord.gin" \
    --gin_file="${GIN_BASE}/eval/basic_f0_ld.gin" \
    --gin_param="TFRecordProvider.file_pattern='${TFRECORD_DIR}/nsynth_test.tfrecord*'" \
    --gin_param="batch_size=16" \
    --gin_param="train.num_steps=5000" \
    --alsologtostderr

echo ""
echo "============================================================"
echo "  Treinamento concluído!"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================================"
