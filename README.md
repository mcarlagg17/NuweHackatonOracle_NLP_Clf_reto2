# Contexto

“NUWE EVA”, la cadena de restauración fast food y healthy, sigue creciendo y queremos dar un paso más para mejorar y ofrecer el mejor servicio posible a nuestros clientes.

Queremos evolucionar y ofrecer nuevas capacidades a nuestro robot y talentoso camarero, “EVA”, añadiendo la función de poder detectar, a través de frases y palabras utilizadas por los comensales, el grado de satisfacción de los platos probados y así ver qué platos y productos han gustado más o menos a los clientes.

Nuestro robot ya sabe detectar lo que pedirán los comensales según los productos elegidos. Ahora queremos saber cuál es el feedback de cada cliente para detectar si le ha gustado o no el plato servido.

# Dataset

Para este reto, dispondrás de 2 CSVs: Train y Test. Como sus nombres indican, el primero te servirá para entrenar tu modelo de clasificación y el test para saber a qué etiqueta pertenecen. Contarás con los siguientes atributos para poder realizar las clasificaciones:

train_idx/test_idx: Identificador de texto.
text: revisar los datos para clasificar si son positivos o negativos.
label: integer representation of the sentiment
label_text: text of the sentiment


# Tareas

Crea un modelo predictivo de clasificación para poder ordenar y o catalogar las reseñas. Primero entrena tu modelo con las reseñas de entrenamiento. Una vez tengas el modelo que maximice la puntuación F1 (macro.), utiliza las reseñas de prueba como entrada para tu modelo.