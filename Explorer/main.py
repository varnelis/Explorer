import click
from Explorer.io.recorder import Recorder
from Explorer.io.scanner import Scanner
import time


@click.group()
def main() -> None:
    pass


@click.command()
def hello_world() -> None:
    print("Testing - Hello world!")


@click.command()
def record() -> None:
    Recorder.record_data = True
    recorder = Recorder()
    recorder.start()

    while recorder.is_running():
        time.sleep(1)
    recorder.finish()

@click.command()
@click.argument('url')
def scan(url) -> None:
    scanner = Scanner(url)
    scanner.scan()


main.add_command(hello_world)
main.add_command(record)
main.add_command(scan)

if __name__ == "__main__":
    main()
