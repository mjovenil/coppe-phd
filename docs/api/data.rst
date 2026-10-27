Dados (``ddsp_sonar.data``)
===========================

Preparação e carregamento de dados para sonar passivo.

O ``ddsp_prepare_tfrecord`` do Magenta foi projetado para áudio musical
(usa CREPE para F0, loudness A-weighted). Para sonar, é necessário um
preparador próprio que extraia descritores físicos relevantes.

.. automodule:: ddsp_sonar.data.provider
   :members:
   :undoc-members: False
   :show-inheritance:

.. automodule:: ddsp_sonar.data.features
   :members:
   :undoc-members: False
   :show-inheritance:
