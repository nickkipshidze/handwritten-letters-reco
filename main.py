import pyglet
from pyglet.window import key

import model
from model import AlphaModel, ConvAlphaModel

class ProgressBar():
    def __init__(self, x, y, width, height, color=(255, 255, 255, 255), bgcolor=(128, 128, 128, 255), status=0.0):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.color, self.bgcolor = color, bgcolor
        self.status = status

        self.background = pyglet.shapes.Rectangle(self.x, self.y, self.width, self.height, self.bgcolor)
        self.foreground = pyglet.shapes.Rectangle(self.x, self.y, self.width*self.status, self.height, self.color)

    def draw(self):
        self.background.width = self.width
        self.foreground.width = self.width*self.status
        self.background.draw()
        self.foreground.draw()

class MainWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixels = []
        self.grid_size = (28, 28)
        self.grid_offsets = (30, 30)
        self.grid_cell_size = 15
        self.cells = [[0 for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]

        self.labels = []
        self.notes = [
            pyglet.text.Label(x=self.width-420, y=self.height-30, text="შესრულებულია ნიკა ყიფშიძის მიერ", font_name="monospace", font_size=16, bold=True),
            pyglet.text.Label(x=10, y=self.height-30, text="\"C\" - გაასუფთავე დაფა", font_name="monospace", font_size=16)
        ]

        self.model = model.Model("model-v6-10000.pth")

        pyglet.clock.schedule_interval(self.predict_model, 1/30)
    
    def draw_grid(self, size=(28, 28), offsets=(0, 0), step=20, color=(255, 255, 255, 125)):
        gridbatch = pyglet.shapes.Batch()
        lines = []
        ox, oy = offsets
        w, h = size
        for x in range(0, w * step, step):
            lines.append(pyglet.shapes.Line(x+ox, oy, x+ox, h*step+oy, 1, color=color, batch=gridbatch))
        lines.append(pyglet.shapes.Line(x+ox+step, oy, x+ox+step, h*step+oy, 1, color=color, batch=gridbatch))
        lines.append(pyglet.shapes.Line(ox, oy+(h*step), w*step+ox, oy+(h*step), 1, color=color, batch=gridbatch))
        for y in range(0, h * step, step):
            lines.append(pyglet.shapes.Line(ox, y+oy, w*step+ox, y+oy, 1, color=color, batch=gridbatch))
        gridbatch.draw()
    
    def grid_coords(self, absolute=(0, 0)):
        relative = [absolute[0] - self.grid_offsets[0], absolute[1] - self.grid_offsets[1]]
        relative[0] //= self.grid_cell_size
        relative[1] //= self.grid_cell_size
        if self.grid_size[0] > relative[0] >= 0 and self.grid_size[1] > relative[1] >= 0:
            return relative
        else:
            return None
    
    def draw_cells(self):
        cellsbatch = pyglet.shapes.Batch()
        cells = []
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                cells.append(
                    pyglet.shapes.Rectangle((x*self.grid_cell_size)+self.grid_offsets[0], (y*self.grid_cell_size)+self.grid_offsets[1], self.grid_cell_size, self.grid_cell_size, color=(255, 255, 255, self.cells[y][x]), batch=cellsbatch)
                )
        cellsbatch.draw()
    
    def fill_cell(self, coords, step=80):
        if self.grid_size[0] > coords[0] >= 0 and self.grid_size[1] > coords[1] >= 0:
            self.cells[coords[1]][coords[0]] += step
            self.cells[coords[1]][coords[0]] = min(255, self.cells[coords[1]][coords[0]])
    
    def on_draw(self):
        self.clear()

        for pixel in self.pixels:
            pixel.draw()
        self.draw_grid(self.grid_size, self.grid_offsets, self.grid_cell_size)
        self.draw_cells()

        for note in self.notes:
            note.draw()

        for label in self.labels:
            label.draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        relative = self.grid_coords((x, y))
        if relative != None:
            x, y = relative
            self.fill_cell((x, y))
            self.fill_cell((x+1, y+1))
            self.fill_cell((x, y+1))

    def on_key_press(self, symbol, modifiers):
        if symbol == key.C:
            self.cells = [[0 for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]
        
    def draw_predictions(self, class_names, probabilities, offsets=(550, 80)):
        if self.labels == []:
            gap = 11
            x_shift = 0
            for index, (label, prob) in enumerate(zip(class_names, probabilities)):
                x_shift += 10
                x_shift %= 20
                self.labels.append(
                    pyglet.text.Label(x=offsets[0]+x_shift, y=index*gap+offsets[1], text=label, font_size=10, bold=True)
                )
                self.labels.append(
                    pyglet.text.Label(x=offsets[0]+50, y=index*gap+offsets[1], font_size=10, text=round(prob, 4).__str__())
                )
                self.labels.append(
                    ProgressBar(x=offsets[0]+100, y=index*gap+offsets[1], width=200, height=6, status=prob)
                )
            argmax = probabilities.index(max(probabilities))
            self.labels.append(
                pyglet.text.Label(x=offsets[0], y=offsets[1]-20, text=f"საბოლოო პასუხი: {class_names[argmax]}", bold=True)
            )
        elif self.labels != []:
            gap = 11
            x_shift = 0
            for index, (label, prob) in enumerate(zip(class_names, probabilities)):
                x_shift += 10
                x_shift %= 20
                li = index*3
                self.labels[li].text = label
                self.labels[li+1].text = round(prob, 4).__str__()
                self.labels[li+2].status = prob
            argmax = probabilities.index(max(probabilities))
            self.labels[-1].text = f"საბოლოო პასუხი: {class_names[argmax]}"

    def predict_model(self, dt):
        class_names, probabilities = self.model.predict(self.cells)
        self.draw_predictions(class_names, probabilities)

window = MainWindow(
    width=1024,
    height=512
)

pyglet.app.run()