import random
import click
from Explorer.io.recorder import Recorder
from Explorer.io.scanner import Scanner
from Explorer.io.scrapper import Scrapper, count_profiles, first_level_uris, generate_scanning_links, most_referenced, most_referenced_first_level_uris, show_graph
import time
import cProfile
import json

from Explorer.io.selenium_scanner import SeleniumScanner

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
@click.argument("source")
@click.argument("url")
@click.option("--d", required=False, type=int, help="Scan depth limit")
@click.option("--a", required=False, type=int, help="Active scan limit")
@click.option(
    "--response", default=0.01, type=float, help="Interactables response time (s)"
)
def scan_bulk(source: str, url: str, d, a, response) -> None:
    uris = []
    with open(source, "r") as f:
        uris = [line[:-1] for line in f]

    scanner = Scanner(
        url = url + uris[0],
        prefix = uris[0].replace("/", "-"),
        response = response,
        mode = "bulk",
        screen_origin = (1119, 300),
        screen_width = 100,
        screen_height = 301,
        url_popup_origin = (1337, 587),
        url_popup_width = 10,
        url_popup_height = 10
    )
    scanner.scan(d, a)
    for u in uris[1:]:
        scanner.load_next_url(url + u, prefix=u.replace("/", "-"))
        scanner.scan(d, a)
    
@click.command()
@click.argument("url")
def selenium_scan(url) -> None:
    SeleniumScanner.setup_driver()
    SeleniumScanner.load_url(url)
    SeleniumScanner.load_screenshot()
    SeleniumScanner.load_bbox()
    SeleniumScanner.draw_bbox()


@click.command()
def bulk_selenium_scan() -> None:
    with open("./scanning_links.json", "r") as f:
        links = json.load(f)

    SeleniumScanner.setup_driver()
    for l in random.choices(links["/math"]["links"], k = 20):
        print(f"Loading: {'https://www.khanacademy.org' + l}")
        SeleniumScanner.load_url("https://www.khanacademy.org" + l)
        SeleniumScanner.load_screenshot()
        SeleniumScanner.load_bbox()
        SeleniumScanner.draw_bbox()

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
def generate_scanning_json() -> None:
    links = generate_scanning_links()
    with open("./scanning_links.json", "w") as f:
        json.dump(links, f)

@click.command()
def visualise() -> None:
    show_graph()


main.add_command(hello_world)
main.add_command(record)
main.add_command(scan)
main.add_command(selenium_scan)
main.add_command(bulk_selenium_scan)
main.add_command(scan_bulk)
main.add_command(scrape)
main.add_command(scrape_info)
main.add_command(generate_scanning_json)
main.add_command(visualise)

if __name__ == "__main__":
    main()
