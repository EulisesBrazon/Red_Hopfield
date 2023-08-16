import tkinter as tk # utilizado para generar una interface grafica
import numpy as np   # para manejo de matrices y calculos numericos

# se define la clase para representar la red de Hopfield
class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size # tamaño de la matriz, en este caso se utilizo dimension de n=5
        self.weights = np.zeros((pattern_size, pattern_size))# se inicializan los pesos en 0, en la matriz nxn
    
    # entrenamiento
    def train(self, patterns): # recibe una lista de patrones como entrada
        # se inicializa con la longitud de la lista de patrones, 
        # lo cual representa el número total de patrones en el conjunto de entrenamiento.
        num_patterns = len(patterns) 
        for pattern in patterns: #se itera sobre cada patron
            # np.reshape trasforma las dimensiones de la matriz 
            # en esta caso pasa de un vector de 25, a uan matriz de 5x5
            pattern = np.reshape(pattern, (1, self.pattern_size))
            # se realiza una suma ponderada para determinar
            # la fuerza de la conexión sináptica entre las neuronas
            self.weights += np.dot(pattern.T, pattern)
        # np.fill_diagonal se utiliza para establecer los valores de la diagonal de la matriz de pesos en cero,
        # los valores en la diagonal de la matriz de pesos representan las conexiones de una neurona consigo misma
        # que pueden influir sobre su activacion
        np.fill_diagonal(self.weights, 0)
        # normalizar los pesos para que esten en una escala adecuada
        self.weights /= num_patterns
    
    # a partir de un patron de entrada, lo recalcula 
    # para encontrar la aproximacion dentro de los patrones de entrenamiento
    # iterando en reiteradas ocasiones, hasta obtener una similitud
    def recall(self, pattern, max_iterations=100): 
        # se trasforma las dimensiones de la matriz 
        pattern = np.reshape(pattern, (1, self.pattern_size))
        # se crea un recorrido para el patron a consultar
        for _ in range(max_iterations):
            prev_pattern = pattern.copy() #almacena una copia en un espacio de memoria distinto
            # np.dot(pattern, self.weights) realiza el producto punto entre el patrón y la matriz de pesos
            # el resultado se pasa por la funcion de activacion np.sign() que retorna 1 o -1
            # pattern, almacena el patron con la nueva activacion calculada
            pattern = np.sign(np.dot(pattern, self.weights)) #
            # en el momento que el patron deje de cambiar, se llego a una de las soluciones
            if np.array_equal(pattern, prev_pattern):
                break
        return pattern # se retorna el resultado

# interface grafica para visualizar los datos
class HopfieldInterface:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.test_pattern = np.zeros(pattern_size)
        
        self.root = tk.Tk()
        self.root.title("Red de Hopfield")
        
        self.canvas = tk.Canvas(self.root, width=250, height=250)
        self.canvas.pack()
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        self.btn_query = tk.Button(self.root, text="Consultar", command=self.query_network)
        self.btn_query.pack()
        
        self.draw_grid()
        
        # Entrenar la red con los patrones de las vocales
        patterns = [
            [-1,  1,  1, -1, -1,
              1, -1, -1,  1, -1,
              1,  1,  1,  1, -1,
              1, -1, -1,  1, -1,
              1, -1, -1,  1, -1],  # Patrón de la vocal 'A'

            [ 1,  1,  1,  1, -1,
              1, -1, -1, -1, -1,
              1,  1,  1,  1, -1,
              1, -1, -1, -1, -1,
              1,  1,  1,  1, -1],  # Patrón de la vocal 'E'

            [ 1,  1,  1,  1,  1,
             -1, -1,  1, -1, -1,
             -1, -1,  1, -1, -1,
             -1, -1,  1, -1, -1,
              1,  1,  1,  1,  1],  # Patrón de la vocal 'I'

            [ 1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1,  1,  1,  1,  1],  # Patrón de la vocal 'O'

            [ 1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1, -1, -1, -1,  1,
              1,  1,  1,  1,  1],  # Patrón de la vocal 'U'
        ]
        
        self.network = HopfieldNetwork(pattern_size)
        self.network.train(patterns)
        
        self.root.mainloop()
    
    def draw_grid(self): # dibujo de las cuadrifculas
        self.canvas.delete("all")
        
        cell_size = 50
        for i in range(5):
            for j in range(5):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                fill_color = "white" if self.test_pattern[i * 5 + j] == 0 else "black"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color)

    # se detecta el evento del click y su coordenada para actualizar el valor de la cuadricula
    def on_canvas_click(self, event):
        cell_size = 50
        x = event.x // cell_size
        y = event.y // cell_size
        idx = y * 5 + x
        
        self.test_pattern[idx] = 1 if self.test_pattern[idx] == 0 else 0
        self.draw_grid()
    
    # al presionar el boton de consultar, se envia la cuadricula a la red de Hopfield
    # y se muestra el resultado obtenido
    def query_network(self): 
        retrieved_pattern = self.network.recall(self.test_pattern)
        
        pattern_window = tk.Toplevel(self.root)
        pattern_window.title("Patrón Recuperado")
        
        pattern_canvas = tk.Canvas(pattern_window, width=250, height=250)
        pattern_canvas.pack()
        
        cell_size = 50
        for i in range(5):
            for j in range(5):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                index = i * 5 + j
                if index < self.pattern_size:
                    fill_color = "white" if retrieved_pattern[0][index] == -1 else "black"
                    pattern_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color)

# Ejemplo de uso
pattern_size = 25

interface = HopfieldInterface(pattern_size)
