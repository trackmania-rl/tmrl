TMRL documentation
==================


The ``tmrl`` library is a complete framework designed to help you implement
deep reinforcement learning pipelines in real-world applications such as robots or videogames.

As a fun example, we readily provide a training pipeline for the
`TrackMania 2020`_ videogame.

.. _`TrackMania 2020`: https://www.trackmania.com

We strongly encourage new readers to visit our GitHub_
as it contains a lot of information and tutorials to help you get on track :)

.. _GitHub: https://github.com/trackmania-rl/tmrl

The documentation describes the ``tmrl`` python API and is intended for developers who want to
implement their own training pipelines.
We also provide an `advanced tutorial`_ for this purpose.

.. _`advanced tutorial`: https://github.com/trackmania-rl/tmrl/blob/master/tmrl/tuto/tuto.py

The three most important classes are ``Server``, ``RolloutWorker`` and ``Trainer``.
All these classes are defined in the ``tmrl.networking`` module.



.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation
   cli


.. toctree::
   :maxdepth: 4
   :caption: API:

   tmrl


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
