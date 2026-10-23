# salve como gerar_audios_run2.py e rode: python3 gerar_audios_run2.py
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit'
import gin, tensorflow as tf, ddsp, ddsp.training as training, soundfile as sf

CHECKPOINT_DIR = 'experiments/checkpoints/nsynth_test_run2'
TFRECORD_PATTERN = 'data/nsynth_test_tfrecord/*.tfrecord*'
OUTPUT_DIR = 'experiments/samples_run2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

gin.parse_config_file(f'{CHECKPOINT_DIR}/operative_config-0.gin')
data_provider = training.data.TFRecordProvider(
    file_pattern=TFRECORD_PATTERN, example_secs=4, sample_rate=16000, frame_rate=250)
dataset = data_provider.get_batch(batch_size=1, shuffle=False).take(4)
model = training.models.get_model()
model.restore(CHECKPOINT_DIR)

for i, batch in enumerate(dataset):
    outputs = model(batch, training=False)
    audio_orig = batch['audio'][0].numpy()
    audio_synth = outputs['audio_synth'][0].numpy()
    sf.write(f'{OUTPUT_DIR}/original_{i+1:02d}.wav', audio_orig, 16000)
    sf.write(f'{OUTPUT_DIR}/reconstructed_{i+1:02d}.wav', audio_synth, 16000)
    print(f'Exemplo {i+1} salvo.')
