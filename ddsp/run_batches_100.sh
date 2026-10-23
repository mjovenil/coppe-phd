#!/usr/bin/env bash
# =============================================================================
# run_batches_100.sh
#
# Processa WAVs em 5 lotes de 100 para evitar OOM na RAM (Beam/CREPE).
# Depois treina o DDSP com batch_size reduzido para evitar OOM na GPU.
#
# Uso (a partir do diretório do projeto):
#   chmod +x run_batches_100.sh
#   ./run_batches_100.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Caminhos — todos relativos a $HOME, sem hardcode de usuário
# ---------------------------------------------------------------------------
WORKDIR="${HOME}/github-code/coppe-phd"
VENV_DIR="${WORKDIR}/ddsp_env"
TFRECORD_DIR="${WORKDIR}/data/nsynth_test_tfrecord"
CHECKPOINT_DIR="${WORKDIR}/experiments/checkpoints/nsynth_test_run1"

# Fonte dos WAVs: prefere nsynth_test_500, cai para o dataset completo
if [ -d "${WORKDIR}/nsynth_test_500" ]; then
    WAV_SRC="${WORKDIR}/nsynth_test_500"
else
    WAV_SRC="${WORKDIR}/data/nsynth_test/audio"
fi

echo "============================================================"
echo "  DDSP — Processamento em Lotes de 100 WAVs"
echo "============================================================"
echo "  WORKDIR    : ${WORKDIR}"
echo "  WAV fonte  : ${WAV_SRC}"
echo "  TFRecords  : ${TFRECORD_DIR}"
echo ""

# Ativar venv (só se não estiver já ativo)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

mkdir -p "${TFRECORD_DIR}"

# ---------------------------------------------------------------------------
# Distribuir WAVs em 5 lotes de 100 (usando sed em vez de array bash)
# ---------------------------------------------------------------------------
echo "Verificando lotes de WAVs..."

for i in 1 2 3 4 5; do
    BATCH_DIR="${WORKDIR}/nsynth_batch${i}"
    mkdir -p "${BATCH_DIR}"
    EXISTING=$(find "${BATCH_DIR}" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "${EXISTING}" -eq 100 ]; then
        echo "  Lote ${i}: já tem 100 WAVs — ok."
    else
        START=$(( (i-1)*100 + 1 ))
        END=$(( i*100 ))
        echo "  Lote ${i}: copiando WAVs ${START}–${END} de ${WAV_SRC}..."
        find "${BATCH_DIR}" -maxdepth 1 -name "*.wav" -delete 2>/dev/null || true
        ls "${WAV_SRC}"/*.wav | sed -n "${START},${END}p" | xargs -I{} cp {} "${BATCH_DIR}/"
        COPIED=$(find "${BATCH_DIR}" -maxdepth 1 -name "*.wav" | wc -l)
        echo "    ${COPIED} WAVs copiados."
    fi
done

# ---------------------------------------------------------------------------
# Converter cada lote para TFRecord (pula se já existir)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Conversão para TFRecord (GPU + CREPE)"
echo "============================================================"

for i in 1 2 3 4 5; do
    BATCH_DIR="${WORKDIR}/nsynth_batch${i}"
    PREFIX="${TFRECORD_DIR}/nsynth_batch${i}.tfrecord"

    if find "${TFRECORD_DIR}" -maxdepth 1 -name "nsynth_batch${i}.tfrecord*" | grep -q .; then
        echo ""
        echo "[Lote ${i}/5] Já processado — pulando."
        ls -lh "${TFRECORD_DIR}"/nsynth_batch${i}.tfrecord* | awk '{print "  " $5, $9}'
        continue
    fi

    echo ""
    echo "[Lote ${i}/5] Processando 100 WAVs..."
    echo "  Início: $(date)"

    ddsp_prepare_tfrecord \
        --input_audio_filepatterns="${BATCH_DIR}/*.wav" \
        --output_tfrecord_path="${PREFIX}" \
        --num_shards=2 \
        --sample_rate=16000 \
        --frame_rate=250 \
        --alsologtostderr

    echo "  Fim: $(date)"
    ls -lh "${TFRECORD_DIR}"/nsynth_batch${i}.tfrecord* | awk '{print "  " $5, $9}'
done

echo ""
echo "============================================================"
echo "  Todos os lotes processados!"
TOTAL_SHARDS=$(find "${TFRECORD_DIR}" -maxdepth 1 -name "*.tfrecord*" | wc -l)
echo "  Total de shards: ${TOTAL_SHARDS}"
ls -lh "${TFRECORD_DIR}"/*.tfrecord* | awk '{print "  " $5, $9}'
echo "============================================================"

# ---------------------------------------------------------------------------
# Verificar conteúdo de um exemplo (lote 1, shard 0)
# ---------------------------------------------------------------------------
echo ""
echo "Verificando conteúdo do TFRecord (lote 1)..."
python3 - <<'PYEOF'
import os, glob, sys
import tensorflow as tf

WORKDIR = os.path.expanduser("~/github-code/coppe-phd")
TFRECORD_DIR = os.path.join(WORKDIR, "data", "nsynth_test_tfrecord")
files = sorted(glob.glob(os.path.join(TFRECORD_DIR, "nsynth_batch1.tfrecord*")))
if not files:
    print("  Nenhum TFRecord do lote 1 encontrado.")
    sys.exit(0)

print(f"  Arquivo: {os.path.basename(files[0])}")
ds = tf.data.TFRecordDataset(files[:1])
for i, raw in enumerate(ds.take(1)):
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    print(f"  Campos:")
    for key, feat in sorted(ex.features.feature.items()):
        kind = feat.WhichOneof('kind')
        if kind == 'float_list':
            vals = list(feat.float_list.value)
            print(f"    {key:22s}  len={len(vals):5d}  min={min(vals):8.3f}  max={max(vals):8.3f}")
        elif kind == 'bytes_list':
            print(f"    {key:22s}  bytes  n={len(feat.bytes_list.value)}")
PYEOF

# ---------------------------------------------------------------------------
# Treinamento DDSP — batch_size=4 para caber na GPU de 6 GB
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Iniciando Treinamento DDSP (5000 passos, batch_size=4)"
echo "============================================================"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo ""

mkdir -p "${CHECKPOINT_DIR}"

GIN_BASE=$(python3 -c "import ddsp, os; print(os.path.join(os.path.dirname(ddsp.__file__), 'training', 'gin'))")
echo "  Gin base: ${GIN_BASE}"
echo ""

# batch_size=4 (em vez de 16) para não dar OOM na RTX 3050 6 GB
ddsp_run \
    --mode=train \
    --save_dir="${CHECKPOINT_DIR}" \
    --gin_file="${GIN_BASE}/models/solo_instrument.gin" \
    --gin_file="${GIN_BASE}/datasets/tfrecord.gin" \
    --gin_file="${GIN_BASE}/eval/basic_f0_ld.gin" \
    --gin_param="TFRecordProvider.file_pattern='${TFRECORD_DIR}/*.tfrecord*'" \
    --gin_param="batch_size=4" \
    --gin_param="train.num_steps=5000" \
    --alsologtostderr

echo ""
echo "============================================================"
echo "  Pipeline concluído!"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================================"