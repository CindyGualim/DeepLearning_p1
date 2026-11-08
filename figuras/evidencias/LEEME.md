# Capturas del servicio en producción

Tomar con el servicio corriendo (`uvicorn api.app:app`) y guardarlas aquí:

| Archivo sugerido | Qué debe mostrar |
|---|---|
| `01_cola.png` | La cola de revisión priorizada por el modelo C |
| `02_caso_normal.png` | El caso **C082** abierto, con su trayectoria y la historia |
| `03_caso_barajado.png` | El mismo caso tras pulsar «Barajar historia»: B se desploma de 1.000 a 0.041 y A no se mueve |
| `04_evidencias.png` | La pestaña de evidencias con la tabla de permutación y el control F3 |
| `05_monitoreo.png` | La pestaña de monitoreo con las peticiones y latencias reales |
| `06_docs.png` | `/docs`, la API interactiva, como prueba de que un tercero puede usarla |

Después enlazarlas desde la sección 7.6 del README.
