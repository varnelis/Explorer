from collections import defaultdict
import csv
import random
import click
from tqdm import tqdm
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
@click.option("--scale", required=True, type=float, help="Scale factor")
def selenium_scan(url, scale) -> None:
    SeleniumScanner.prepare_directories()
    SeleniumScanner.setup_driver(scale)
    SeleniumScanner.load_url(url)

    eop = SeleniumScanner.end_of_page()
    while next(eop) is False:
        SeleniumScanner.load_screenshot()
        SeleniumScanner.load_bbox()
        SeleniumScanner.draw_bbox()
        SeleniumScanner.save_scan()
        SeleniumScanner.scroll_page()

@click.command()
@click.option("--g", required=True, type=int, help="Group number")
@click.option("--scale", required=True, type=float, help="Scale factor")
@click.option("--scroll", required=True, type=bool, help="Scroll page")
@click.option(
    "--ep", required=False, default = 0, type=float, help="Button expand probability"
)
def bulk_selenium_scan(g, scale, scroll, ep) -> None:
    SeleniumScanner.prepare_directories()

    with open("./scanning_links_allocations.json", "r") as f:
        links_to_scan = json.load(f)
    with open("./selenium_scans/metadata/visited_list.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        scanned_links = {line["url"]: line["uuid"] for line in csv_reader}

    pb = tqdm(total = len(links_to_scan[str(g)]), position = 0)
    SeleniumScanner.setup_driver(scale)
    for link in links_to_scan[str(g)]:
        if link in scanned_links:
            pb.update(1)
            continue
        
        SeleniumScanner.load_url(link)
        eop = SeleniumScanner.end_of_page()
        while next(eop) is False:
            SeleniumScanner.load_screenshot()
            SeleniumScanner.load_bbox()
            SeleniumScanner.draw_bbox()
            SeleniumScanner.save_scan()
            SeleniumScanner.scroll_page()

        scanned_links["link"] = SeleniumScanner.current_uuid.hex
        pb.update(1)

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
@click.option("--s", required=True, type=int, help="Random number generator seed")
@click.option("--n", required=True, type=int, help="Number of links to scan")
@click.option("--g", required=True, type=int, help="Groups to allocate")
def generate_scanning_json(s, n, g) -> None:
    random.seed(s)
    
    links = generate_scanning_links()
    total_links = links["total"]
    allocations = defaultdict(list)

    for k in links.keys():
        if k == "total":
            continue

        n_links_to_allocate = max(1, int(links[k]["count"] / total_links * n))
        if k == "/login":
            n_links_to_allocate = 1

        random_links = random.choices(links[k]["links"], k = n_links_to_allocate)

        for link in random_links:
            group = int(random.random() * g)
            allocations[group].append(link)

    with open("./scanning_links_allocations.json", "w") as f:
        json.dump(allocations, f)

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
