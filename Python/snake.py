
from turtle import Turtle


start_position = [(0,0), (-20,0),(-40,0)]
move_forward = 20
UP = 90
DOWN = 270
LEFT = 180
RIGHT = 0


class Snake:
    def __init__(self):
        self.blocks = []
        self.create_snake()
        self.head = self.blocks[0]
    def create_snake(self):
        # making initial snake body
        for seg in start_position:
            self.add_body(seg)

    def add_body(self, seg):
        new_segment = Turtle("square")
        new_segment.color("white")
        new_segment.penup()
        new_segment.goto(seg)
        self.blocks.append(new_segment)
    def extend_body(self):
        self.add_body(self.blocks[-1].position())

    def move(self):
        for block in range(len(self.blocks) - 1, 0, -1):
            new_x = self.blocks[block - 1].xcor()
            new_y = self.blocks[block - 1].ycor()
            self.blocks[block].goto(new_x, new_y)
        self.head.forward(move_forward)
    def up(self):
        if self.head.heading() != DOWN:
            self.head.setheading(UP)
    def down(self):
        if self.head.heading() != UP:
            self.head.setheading(DOWN)
    def left(self):
        if self.head.heading() != RIGHT:
            self.head.setheading(LEFT)
    def right(self):
        if self.head.heading() != LEFT:
            self.head.setheading(RIGHT)