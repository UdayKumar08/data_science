from turtle import Turtle
class Score(Turtle):
    def __init__(self):
        super().__init__()
        self.score = 0
        self.color("white")
        self.penup()
        self.goto(0,280)
        self.hideturtle()
        self.update_score()

    def update_score(self):

        self.write(f"score: {self.score}", align='center', font=('Arial', 24, 'normal'))
    def game_over(self):
        self.goto(0,0)
        self.write("GAME OVER", align='center', font=('Arial', 24, 'normal'))

    def increase_score(self):
        self.score += 1
        self.clear()
        self.update_score()

