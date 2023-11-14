import click
from Explorer.input_recorder.recorder import record_input


@click.group()
def main() -> None:
    pass


@click.command()
def hello_world() -> None:
    print("Testing - Hello world!")


@click.command()
def record() -> None:
    record_input()


main.add_command(hello_world)
main.add_command(record)

if __name__ == "__main__":
    main()
