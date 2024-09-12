# Self Driving Car

# Importing the libraries
import math
import sys
import time
from random import randint, random

import matplotlib.pyplot as plt
import numpy as np

# Importing the Kivy packages
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import NumericProperty, ObjectProperty, ReferenceListProperty
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# from ai2_defalt import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set("input", "mouse", "mouse,multitouch_on_demand")

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
# brain = Dqn(8, 3, 0.9)
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

# Initializing the map
first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


# Initializing the last distance
last_distance = 0

# Creating the car class


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # print("move")
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle = (self.angle + self.rotation) % 360
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        self.signal1 = (
            math.ceil(
                int(
                    np.sum(
                        sand[
                            int(self.sensor1_x) - 10 : int(self.sensor1_x) + 10,
                            int(self.sensor1_y) - 10 : int(self.sensor1_y) + 10,
                        ]
                    )
                )
                / 400.0
                * 10
            )
            / 10
        )
        self.signal2 = (
            math.ceil(
                int(
                    np.sum(
                        sand[
                            int(self.sensor2_x) - 10 : int(self.sensor2_x) + 10,
                            int(self.sensor2_y) - 10 : int(self.sensor2_y) + 10,
                        ]
                    )
                )
                / 400.0
                * 10
            )
            / 10
        )
        self.signal3 = (
            math.ceil(
                int(
                    np.sum(
                        sand[
                            int(self.sensor3_x) - 10 : int(self.sensor3_x) + 10,
                            int(self.sensor3_y) - 10 : int(self.sensor3_y) + 10,
                        ]
                    )
                )
                / 400.0
                * 10
            )
            / 10
        )
        if (
            self.sensor1_x > longueur - 10
            or self.sensor1_x < 10
            or self.sensor1_y > largeur - 10
            or self.sensor1_y < 10
        ):
            self.signal1 = 1.0
        if (
            self.sensor2_x > longueur - 10
            or self.sensor2_x < 10
            or self.sensor2_y > largeur - 10
            or self.sensor2_y < 10
        ):
            self.signal2 = 1.0
        if (
            self.sensor3_x > longueur - 10
            or self.sensor3_x < 10
            or self.sensor3_y > largeur - 10
            or self.sensor3_y < 10
        ):
            self.signal3 = 1.0


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Creating the game class


class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        # print("update")
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        # NN의 성능을 위해서 x, y 좌표값 [0,1] 범위로 정규화
        normalized_x = self.car.x / self.width
        normalized_y = self.car.y / self.height
        if goal_x > goal_y:
            goal_boolean = 1
        else:
            goal_boolean = 0
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0
        # orientation = Vector(*self.car.velocity).signed_angle((xx, yy)) / 180.0
        last_signal = [
            self.car.signal1,
            self.car.signal2,
            self.car.signal3,
            orientation,
            -orientation,
            # normalized_x,
            # normalized_y,
            # goal_boolean,
        ]
        # print(Vector(*self.car.velocity).angle((xx, yy)), orientation)

        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:  # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.3
            if distance < last_distance:
                last_reward = 0.2 * (-(distance - last_distance)) / 6.0
                print(last_reward)

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        last_distance = distance


# Adding the painting tools


class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.0
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == "left":
            touch.ud["line"].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.0
            density = n_points / (length)
            touch.ud["line"].width = int(20 * density + 1)
            sand[
                int(touch.x) - 10 : int(touch.x) + 10,
                int(touch.y) - 10 : int(touch.y) + 10,
            ] = 1
            last_x = x
            last_y = y

    def refresh_canvas(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.0
            for x in range(0, sand.shape[0], 5):  # 5단위로 건너뛰기 -> 성능 최적화
                for y in range(0, sand.shape[1], 5):
                    if sand[x, y] > 0:
                        # Rectangle으로 더 빠르게 그리기
                        Ellipse(pos=(x, y), size=(10, 10))


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()

        clearbtn = Button(text="clear")
        savebtn = Button(text="save", pos=(parent.width, 0))
        loadbtn = Button(text="load", pos=(2 * parent.width, 0))
        savesandbtn = Button(text="save sand", pos=(3 * parent.width, 0))
        loadsandbtn = Button(text="load sand", pos=(4 * parent.width, 0))

        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        savesandbtn.bind(on_release=self.save_sand)
        loadsandbtn.bind(on_release=self.load_sand)

        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(savesandbtn)
        parent.add_widget(loadsandbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

    def save_sand(self, obj):
        np.save("sand.npy", sand)
        print("Sand configuration saved.")

    def load_sand(self, obj):
        global sand
        sand = np.load("sand.npy")
        self.painter.refresh_canvas()  # 로드 후 화면 갱신
        print("Sand configuration loaded.")


# Running the whole thing
if __name__ == "__main__":
    CarApp().run()
