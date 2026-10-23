#!/usr/bin/env python3
"""
ddsp_verify_metrics.py
======================
Verifica as métricas F0 e loudness após o treinamento, comparando
com a Tabela 1 do paper DDSP (ICLR 2020).

Uso:
    python ddsp_verify_metrics.py --checkpoint_dir ~/github-code/coppe-phd/experiments/checkpoints/nsynth_test_run1
                                  --tfrecord_pattern "~/github-code/coppe-phd/data/nsynth_test_tfrecord/nsynth_test.tfrecord*"

Referência (Tabela 1 do paper DDSP ICLR 2020):
    F0 shift (cents)     : 16.5 (DDSP-Add) vs 26.4 (baseline Supervised)
    Loudness shift (dB)  :  2.1 (DDSP-Add) vs  3.5 (baseline Supervised)
    F0 outlier (<600c)   : 15.7% (DDSP-Add)
"""

import argparse
import glob
import os
import sys

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Métricas de referência (Tabela 1, DDSP ICLR 2020)
# Modelo: DDSP-Add treinado com NSynth (gansynth_subset.f0_and_loudness)
# ---------------------------------------------------------------------------
REFERENCE = {
    "f0_shift_cents": 16.5,
    "loudness_shift_db": 2.1,
    "f0_outlier_pct": 15.7,  # % de frames com |shift| > 600 cents
}


def parse_tfrecord(file_pattern, max_examples=200):
    """Lê exemplos do TFRecord e retorna audio, f0_hz, loudness_db."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"[ERRO] Nenhum arquivo encontrado: {file_pattern}")
        sys.exit(1)

    feature_description = {
        "audio":        tf.io.VarLenFeature(tf.float32),
        "f0_hz":        tf.io.VarLenFeature(tf.float32),
        "f0_confidence":tf.io.VarLenFeature(tf.float32),
        "loudness_db":  tf.io.VarLenFeature(tf.float32),
    }

    dataset = tf.data.TFRecordDataset(files)
    examples = {"audio": [], "f0_hz": [], "loudness_db": []}

    for i, raw in enumerate(dataset.take(max_examples)):
        parsed = tf.io.parse_single_example(raw, feature_description)
        examples["audio"].append(
            tf.sparse.to_dense(parsed["audio"]).numpy())
        examples["f0_hz"].append(
            tf.sparse.to_dense(parsed["f0_hz"]).numpy())
        examples["loudness_db"].append(
            tf.sparse.to_dense(parsed["loudness_db"]).numpy())

    print(f"  Lidos {len(examples['audio'])} exemplos de {len(files)} shards.")
    return examples


def compute_f0_shift_cents(f0_true, f0_pred, confidence_threshold=0.85,
                            f0_confidence=None):
    """
    Calcula a diferença média de F0 em cents entre true e pred,
    filtrando frames com baixa confiança.
    Cents = 1200 * log2(f0_pred / f0_true)
    """
    # Filtra zeros e baixa confiança
    mask = (f0_true > 0) & (f0_pred > 0)
    if f0_confidence is not None:
        mask = mask & (f0_confidence >= confidence_threshold)

    if mask.sum() == 0:
        return np.nan, 0.0

    f0_t = f0_true[mask]
    f0_p = f0_pred[mask]

    cents = 1200.0 * np.log2(np.clip(f0_p / f0_t, 1e-6, 1e6))
    outlier_pct = 100.0 * (np.abs(cents) > 600).mean()
    # Filtra outliers antes de calcular a média (como no paper)
    valid = np.abs(cents) <= 600
    shift = np.abs(cents[valid]).mean() if valid.sum() > 0 else np.nan

    return shift, outlier_pct


def run_model_inference(checkpoint_dir, examples, batch_size=16):
    """
    Carrega o checkpoint do DDSP e roda inferência para obter
    F0 e loudness reconstruídos.
    Retorna dicts {f0_hz_pred, loudness_db_pred}.
    """
    try:
        import ddsp
        import ddsp.training
        import gin
    except ImportError:
        print("[AVISO] ddsp não instalado. Pulando inferência real.")
        print("         Execute: pip install ddsp")
        return None

    # Verificar se checkpoint existe
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.index"))
    if not ckpt_files:
        print(f"[AVISO] Nenhum checkpoint encontrado em: {checkpoint_dir}")
        print("         Execute o treinamento antes de verificar as métricas.")
        return None

    print(f"  Checkpoint: {sorted(ckpt_files)[-1]}")

    # Carregar o modelo via gin config salva junto ao checkpoint
    gin_file = os.path.join(checkpoint_dir, "operative_config-0.gin")
    if not os.path.exists(gin_file):
        # Tentar encontrar qualquer operative_config
        gin_files = glob.glob(os.path.join(checkpoint_dir, "operative_config*.gin"))
        if gin_files:
            gin_file = sorted(gin_files)[-1]
        else:
            print(f"[AVISO] Gin config não encontrado em {checkpoint_dir}.")
            return None

    gin.parse_config_file(gin_file, skip_unknown=True)
    model = gin.get_bindings("get_model")

    # TODO: carregar pesos e rodar forward pass
    # Por hora, retorna None para mostrar apenas as estatísticas dos dados
    print("  [INFO] Inferência completa requer integração com ddsp_run.")
    print("         Use este script após o treinamento para inspecionar os logs.")
    return None


def print_data_statistics(examples):
    """Imprime estatísticas descritivas dos dados (ground truth)."""
    f0_all = np.concatenate(examples["f0_hz"])
    ld_all = np.concatenate(examples["loudness_db"])

    voiced = f0_all[f0_all > 0]

    print("\n  ── Estatísticas dos dados (ground truth) ──")
    print(f"  F0 (voiced frames)")
    print(f"    n frames     : {len(voiced):,}")
    print(f"    min          : {voiced.min():.1f} Hz")
    print(f"    max          : {voiced.max():.1f} Hz")
    print(f"    mediana      : {np.median(voiced):.1f} Hz")
    print(f"    % voiced     : {100*len(voiced)/len(f0_all):.1f}%")
    print(f"  Loudness (dB)")
    print(f"    min          : {ld_all.min():.1f} dB")
    print(f"    max          : {ld_all.max():.1f} dB")
    print(f"    média        : {ld_all.mean():.1f} dB")
    print(f"    std          : {ld_all.std():.1f} dB")

    # Verificar frame rate (esperado: 250 Hz → 1 frame a cada 4ms)
    # Para áudio de 16kHz e 4 segundos → 64000 amostras → 1000 frames
    sample_audio = examples["audio"][0]
    sample_f0 = examples["f0_hz"][0]
    if len(sample_audio) > 0 and len(sample_f0) > 0:
        inferred_sr = 16000  # padrão NSynth
        audio_dur_s = len(sample_audio) / inferred_sr
        frame_rate = len(sample_f0) / audio_dur_s
        print(f"\n  Frame rate inferido  : {frame_rate:.1f} Hz  (esperado: 250 Hz)")
        print(f"  Duração do áudio     : {audio_dur_s:.2f} s  (esperado: 4.0 s)")


def compare_with_paper(f0_shift=None, loudness_shift=None, outlier_pct=None):
    """Compara os resultados com a Tabela 1 do paper DDSP."""
    print("\n  ── Comparação com Tabela 1 (DDSP ICLR 2020) ──")
    print(f"  {'Métrica':<28} {'Paper':>8} {'Nosso':>8} {'Δ':>8}")
    print(f"  {'-'*52}")

    ref_f0 = REFERENCE["f0_shift_cents"]
    ref_ld = REFERENCE["loudness_shift_db"]
    ref_out = REFERENCE["f0_outlier_pct"]

    def fmt(val, ref):
        if val is None:
            return f"{'—':>8} {'—':>8}"
        delta = val - ref
        sign = "+" if delta >= 0 else ""
        return f"  {ref:>8.1f} {val:>8.1f} {sign}{delta:>7.1f}"

    print(f"  {'F0 shift (cents)':<28}" + fmt(f0_shift, ref_f0))
    print(f"  {'Loudness shift (dB)':<28}" + fmt(loudness_shift, ref_ld))
    print(f"  {'F0 outlier % (>600c)':<28}" + fmt(outlier_pct, ref_out))

    print(f"\n  Nota: valores negativos em Δ indicam melhora relativa ao paper.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_dir", type=str,
                        default=os.path.expanduser(
                            "~/github-code/coppe-phd/experiments/checkpoints/nsynth_test_run1"),
                        help="Diretório com os checkpoints do treinamento.")
    parser.add_argument("--tfrecord_pattern", type=str,
                        default=os.path.expanduser(
                            "~/github-code/coppe-phd/data/nsynth_test_tfrecord/nsynth_test.tfrecord*"),
                        help="Glob pattern para os arquivos TFRecord.")
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Máximo de exemplos a carregar para estatísticas.")
    args = parser.parse_args()

    print("\n============================================================")
    print("  DDSP — Verificação de Métricas (Tabela 1 ICLR 2020)")
    print("============================================================")

    # 1. Estatísticas dos dados
    print("\n[1/3] Carregando TFRecords...")
    examples = parse_tfrecord(args.tfrecord_pattern, args.max_examples)
    print_data_statistics(examples)

    # 2. Inferência com o modelo treinado
    print("\n[2/3] Carregando modelo para inferência...")
    preds = run_model_inference(args.checkpoint_dir, examples)

    # 3. Calcular métricas (só disponível se a inferência rodou)
    print("\n[3/3] Métricas:")
    if preds is not None:
        # Calcular F0 shift e loudness shift entre reconstruction e input
        f0_shifts = []
        ld_shifts = []
        outlier_pcts = []

        for true_f0, pred_f0, true_ld, pred_ld in zip(
                examples["f0_hz"], preds["f0_hz_pred"],
                examples["loudness_db"], preds["loudness_db_pred"]):
            shift, out_pct = compute_f0_shift_cents(
                np.array(true_f0), np.array(pred_f0))
            f0_shifts.append(shift)
            outlier_pcts.append(out_pct)
            # Loudness shift em dB
            mask = np.array(true_f0) > 0
            if mask.sum() > 0:
                ld_shifts.append(np.abs(
                    np.array(pred_ld)[mask] - np.array(true_ld)[mask]).mean())

        mean_f0_shift = np.nanmean(f0_shifts)
        mean_ld_shift = np.nanmean(ld_shifts) if ld_shifts else None
        mean_outlier = np.nanmean(outlier_pcts)
        compare_with_paper(mean_f0_shift, mean_ld_shift, mean_outlier)
    else:
        compare_with_paper(f0_shift=None, loudness_shift=None, outlier_pct=None)
        print("\n  Para obter as métricas reais após o treinamento, verifique os")
        print("  logs do ddsp_run — as métricas F0/loudness são logadas a cada")
        print("  avaliação e ficam disponíveis no TensorBoard:")
        print(f"    tensorboard --logdir={args.checkpoint_dir}")

    print("\n  Dica: para rodar o evaluate separado do treino:")
    print("    ddsp_run --mode=eval --save_dir=<ckpt_dir> \\")
    print("             --gin_file=<mesmos gin files do treino>")
    print("             --alsologtostderr")
    print()


if __name__ == "__main__":
    main()
