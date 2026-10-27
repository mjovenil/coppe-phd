Instalação
==========

O desenvolvimento ocorre dentro do container Docker ``ddsp-sonar:latest``.
Não é necessário instalar dependências manualmente — elas já estão na imagem.

Requisitos
----------

* Docker com suporte a GPU (``nvidia-container-toolkit``)
* NVIDIA driver compatível com CUDA 11.7

Iniciar o container
-------------------

.. code-block:: bash

    docker run --gpus all -it --rm \
      --name ddsp-sonar \
      -v ~/github-code/coppe-phd:/workspace \
      ddsp-sonar:latest \
      bash

Dentro do container, o código estará em ``/workspace/ddsp_sonar/``.

Gerar a documentação
--------------------

.. code-block:: bash

    # Dentro do container
    pip install sphinx sphinx-rtd-theme  # já instalado na imagem
    cd /workspace/docs
    make html

    # Abrir no navegador (fora do container, na máquina host)
    xdg-open ~/github-code/coppe-phd/docs/_build/html/index.html

Dependências Python (já na imagem Docker)
-----------------------------------------

.. list-table::
   :header-rows: 1

   * - Pacote
     - Versão
   * - torch
     - 2.x (cu117)
   * - tensorflow
     - 2.11.1
   * - ddsp
     - 3.6.0
   * - sphinx
     - instalado via pip
   * - sphinx-rtd-theme
     - instalado via pip
