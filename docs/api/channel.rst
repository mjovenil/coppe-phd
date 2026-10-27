Módulo de Canal (``ddsp_sonar.models.channel``)
================================================

Módulo que modela a função de transferência do canal acústico submarino.
Equivalente físico do ``Reverb`` treinável do DDSP musical.

O canal é determinado pelo perfil CTD (velocidade do som × profundidade),
geometria (distância fonte-receptor, batimetria) e frequência.
Este módulo aprende a separar o que vem da **fonte** do que o **canal** acrescenta.

.. automodule:: ddsp_sonar.models.channel
   :members:
   :undoc-members: False
   :show-inheritance:
