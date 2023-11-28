import click
from Explorer.io.recorder import Recorder
import time


@click.group()
def main() -> None:
    pass


@click.command()
def hello_world() -> None:
    print("Testing - Hello world!")


@click.command()
def record() -> None:
    recorder = Recorder()
    recorder.start()

    while recorder.is_running():
        time.sleep(1)
    recorder.finish()


main.add_command(hello_world)
main.add_command(record)

if __name__ == "__main__":
    main()
