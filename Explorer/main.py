import click
from Explorer.io.recorder import Recorder
from Explorer.io.scanner import Scanner
from Explorer.io.scrapper import Scrapper, count_profiles, first_level_uris, most_referenced, most_referenced_first_level_uris, show_graph
import time
import cProfile

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
@click.option("--npass", default=0, type=int, help="Do n non-active passes")
@click.option("--prefix", default="", type=str, help="File prefix")
@click.option(
    "--response", default=0.01, type=float, help="Interactables response time (s)"
)
def scan(url, d, a, npass, prefix, response) -> None:
    scanner = Scanner(url, prefix=prefix, response=response)
    scanner.scan(d, a, npass)


@click.command()
def scrape() -> None:
    crawler = Scrapper("https://www.khanacademy.org")
    crawler.start("/")

@click.command()
def scrape_info() -> None:
    count_profiles()
    first_level_uris()
    most_referenced()
    most_referenced_first_level_uris()

@click.command()
def visualise() -> None:
    show_graph()


main.add_command(hello_world)
main.add_command(record)
main.add_command(scan)
main.add_command(scrape)
main.add_command(scrape_info)
main.add_command(visualise)

if __name__ == "__main__":
    main()
