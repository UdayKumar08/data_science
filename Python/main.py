from turtle import Screen
import time
from snake import Snake
from project1.food import Food
from score import Score

screen = Screen()

#screen set of with screen size
screen.setup(600,600)
screen.bgcolor("black")
screen.title("snake Game")

screen.tracer(0)
snake = Snake()
food = Food()
score = Score()
screen.listen()

screen.onkey(snake.up,"Up")
screen.onkey(snake.down, "Down")
screen.onkey(snake.left, "Left")
screen.onkey(snake.right, "Right")

Start_game = True
while Start_game:
    screen.update()
    time.sleep(0.1)
    snake.move()

    if snake.head.distance(food) < 15:
        food.refresh()
        snake.extend_body()
        score.increase_score()

    if snake.head.xcor() > 295 or snake.head.xcor() < -295 or snake.head.ycor() > 295 or snake.head.ycor() < -295:
        Start_game = False
        score.game_over()

    for seg  in snake.blocks[1:]:
        if snake.head.distance(seg) < 10:
            Start_game = False
            score.game_over()

screen.exitonclick()


