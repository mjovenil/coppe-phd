# coppe-phd

Repositório do doutorado em Engenharia Elétrica (ênfase em Inteligência Computacional) — COPPE/UFRJ.  
Tema: modelo DDSP diferenciável para sinais de SONAR passivo.

## Estrutura

```
coppe-phd/
  ddsp/           ← Fase 1: replicação do DDSP original (TensorFlow 2.11.1)
  ddsp_sonar/     ← Fase 2: adaptação para acústica submarina (PyTorch)
  .devcontainer/  ← configuração VS Code Dev Container
  .gitignore
  README.md
```

## Ambiente

Imagem Docker: `ddsp-sonar:latest`  
Dockerfile em: [github.com/mjovenil/dockers/ddsp_sonar](https://github.com/mjovenil/dockers/tree/main/ddsp_sonar)

### Requisitos
- NVIDIA GPU (testado: RTX 3050 Laptop, CUDA 11.5+)
- Docker + nvidia-container-toolkit
- VS Code + extensão Dev Containers

### Iniciar

```bash
# Clonar
git clone https://github.com/mjovenil/coppe-phd.git
cd coppe-phd

# Build da imagem (uma vez)
git clone https://github.com/mjovenil/dockers.git /tmp/dockers
docker build -t ddsp-sonar:latest /tmp/dockers/ddsp_sonar/

# Abrir no VS Code dentro do container
code .
# Ctrl+Shift+P → Dev Containers: Reopen in Container
```

## Fase 1 — Replicação DDSP (`ddsp/`)

Replicação do autoencoder DDSP (Google/Magenta) com dataset NSynth.  
Runs concluídos: Run 1 (5k steps), Run 2 (50k steps, loudness_db ≈ 12.66).

## Fase 2 — DDSP-Sonar (`ddsp_sonar/`)

Reimplementação em PyTorch com módulos físicos submarinos:
- `blade_rate_oscillator.py` — oscilador de taxa de pá (→ f0)
- `ctd_reverb.py` — reverberação por canal CTD/geometria
- `cavitation_noise.py` — ruído de cavitação (→ FilteredNoise)
