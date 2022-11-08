from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.layout import Layout


class MainWidget(Widget):
    pass


class MainApp(App):
    def build(self):
        return MainWidget()


if __name__ == '__main__':
    MainApp().run()
