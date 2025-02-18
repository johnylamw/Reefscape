import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock

from kivy.core.window import Window

import math

class HexagonLayout(RelativeLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Background image dimensions (original size)
        self.bg_width = 1886
        self.bg_height = 1528

        # Background image
        self.background = Image(
            source="blue_field.jpg",
            size_hint=(None, None),
            size=(self.bg_width, self.bg_height),
            allow_stretch=True,
            keep_ratio=True
        )
        self.add_widget(self.background)

        # Define button positions relative to background size
        self.buttons_data = [
            {"text": "A", "pos": (420, 320)},
            {"text": "B", "pos": (700, 600)},
            {"text": "C", "pos": (1000, 900)}
        ]

        # Create buttons list
        self.buttons = []
        for data in self.buttons_data:
            button = Button(text=data["text"], size_hint=(None, None))
            self.add_widget(button)
            self.buttons.append(button)

        # Bind resizing event
        Window.bind(on_resize=self.update_layout)

        # Initial layout update
        self.update_layout(Window, Window.width, Window.height)
        
        # Getting the positions 
        self.mouse = Label(text="Mouse Position: (0, 0)", size_hint=(None, None), pos=(10, 10))
        Clock.schedule_interval(self.update_mouse_position, 0.05)
        self.add_widget(self.mouse)

    
    def update_layout(self, instance, width, height):
        # Update background size
        self.background.size = (width, height)

        # Scale button size proportionally
        button_size_ratio = 0.005  # Adjust ratio as needed
        button_width = width * button_size_ratio
        button_height = height * button_size_ratio

        for button, data in zip(self.buttons, self.buttons_data):
            original_x, original_y = data["pos"]
            button.size = (button_width, button_height)
            
            # Scale button position proportionally
            button_x_ratio = original_x / self.bg_width
            button_y_ratio = original_y / self.bg_height
            button.pos = (width * button_x_ratio, height * button_y_ratio)

    
    def update_mouse_position(self, dt):
        x, y = Window.mouse_pos
        self.mouse.text = f"Mouse Position: ({int(x)}, {int(y)})"
        
class ReefVisualizer(App):

    def build(self):
        return HexagonLayout()

if __name__ == '__main__':
    ReefVisualizer().run()