from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget


class MainWidge(Widget):
    pass


class MainApp(App):
    def build(self):
        return Label(text="hello algeo")


if __name__ == '__main__':
    MainApp().run()
