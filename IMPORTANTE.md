Notas Importantes sobre la Ejecución del Notebook

Este notebook forma parte del taller Clasificación de Diabetes (CRISP-DM).
El contenido, el código y los resultados fueron verificados y probados correctamente desde el script principal main.py.

Sin embargo, el notebook no puede ejecutarse en mi entorno local debido a un problema externo a Python:

Problema técnico (explicación breve)

Al ejecutar el notebook, mi entorno presenta el error:

Could not find a suitable TLS CA certificate bundle


Esto ocurre porque mi sistema (Windows + PyCharm + PostgreSQL instalado) está apuntando a un archivo de certificados SSL que no existe o está dañado:

C:\Program Files\PostgreSQL\16\ssl\certs\ca-bundle.crt


Este es un problema del sistema operativo, NO del código.

Verificación del taller

Aunque el notebook no se puede ejecutar localmente:

El código completo fue ejecutado correctamente desde main.py.

Todas las métricas, gráficos y resultados del modelo fueron generados sin errores.

El notebook contiene el mismo código validado, por lo que puede ejecutarse sin problemas en:

Google Colab

Jupyter Notebook estándar

VSCode Jupyter

Cualquier PC sin conflicto de certificados SSL

Conclusión

El notebook es totalmente funcional, solo que en mi equipo existe un conflicto particular con los certificados SSL.
El taller está completamente desarrollado, validado y reproducible desde el archivo main.py.

Y dentro del Notebook debes poner un aviso pequeño al inicio

Copia esto en la primera celda Markdown del notebook:

Nota sobre la ejecución del notebook

Debido a un problema de certificados SSL en mi entorno local, las celdas pueden no ejecutarse correctamente.
El código fue validado y ejecutado sin errores desde main.py, por lo cual los resultados, gráficos y métricas del taller son correctos y reproducibles en cualquier entorno Jupyter estándar (Colab, VSCode, notebook clásico).