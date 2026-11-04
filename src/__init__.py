"""Centinela — codigo de servicio del modelo del Proyecto 1.

El notebook `proyecto1_Gualim_Ocaña.ipynb` sigue siendo la fuente de verdad del
entrenamiento. Este paquete extrae de ahi, sin cambiarlas, las piezas que hacen
falta para *servir* el modelo: generador, construccion de caracteristicas,
definiciones de red y puntuador. La prueba `pruebas/test_paridad.py` verifica que
los puntajes del servicio coinciden con los del notebook.
"""
