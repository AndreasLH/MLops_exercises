import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel, test, training


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    epochs = 50

    model_out = training(model, train_set, criterion, optimizer, epochs)

    torch.save(model_out.state_dict(), 's1_development_environment/exercise_files/final_exercise/checkpoint.pth')
    print('Model saved to: s1_development_environment/exercise_files/final_exercise/checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    class Color:
        red = '\033[31m'
        green = '\033[32m'
    color = Color.green
    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    _, test_set = mnist()
    model = MyAwesomeModel()

    accuracy = test(model, test_set, state_dict)
    if accuracy.item()*100 < 85:
        color = Color.red

    print(color,f'Accuracy: {accuracy.item()*100:.3f}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
    