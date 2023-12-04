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
@click.argument("url")
@click.option("--d", required=False, type=int, help="Scan depth limit")
@click.option("--a", required=False, type=int, help="Active scan limit")
@click.option("--npass", default = 0, type = int, help="Do n non-active passes")
@click.option("--prefix", default = "", type = str, help="File prefix")
def scan(url, d, a, npass, prefix) -> None:
    scanner = Scanner(url, prefix = prefix)
    scanner.scan(d, a, npass)


main.add_command(hello_world)
main.add_command(record)
main.add_command(scan)

if __name__ == "__main__":
    main()
