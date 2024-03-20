from collections import defaultdict
import csv
import random
import sys
from bson import ObjectId
import click
import numpy as np
from tqdm import tqdm
from Explorer.db.mongo_db import MongoDBInterface
from Explorer.db.mongo_db_types import ImageTextContent, OCRModel, Platform
from Explorer.io.recorder import Recorder
from Explorer.io.scanner import Scanner
from Explorer.io.scrapper import Scrapper, count_profiles, first_level_uris, generate_scanning_links, most_referenced, most_referenced_first_level_uris, show_graph
import time
import json
import io
from dacite import from_dict
from PIL import Image
from imagehash import average_hash
from datetime import datetime
import os

from Explorer.io.selenium_scanner import SeleniumScanner
from Explorer.objectives.objective_1 import Objective
from Explorer.ocr.ocr import KhanOCR
from Explorer.io.snapshot_grabber import SnapshotGrabber
from Explorer.tf_idf.tf_idf import Index as TFIDF_Index
from Explorer.tf_idf.tokenizer import Tokenizer as TFIDF_Tokenizer
from Explorer.tf_idf.filters import LowerCaseFilter as TFIDF_LowerCaseFilter
from Explorer.overlay.shortlister import Shortlister
from Explorer.speech.speech2text import CommandPhrase, Speech2Text
from Explorer.trace_similarity.action_matching import ActionMatching
from Explorer.trace_processing.trace_processor import TraceProcessor, TraceVisualiser
from Explorer.trace_similarity.screen_similarity import ScreenSimilarity

import seaborn as sns
import matplotlib.pylab as plt

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import QEvent, Qt

def from_dict_list(_type, data: list) -> list:
    return [from_dict(_type, d) for d in data]

@click.group()
def main() -> None:
    pass

@click.command()
def hello_world() -> None:
    print("Testing - Hello world!")


@click.command()
def record() -> None:
    Recorder.record_data = True
    grabber = SnapshotGrabber()
    recorder = Recorder()

    start_time = time.time()
    recorder.start(start_time)
    grabber.start(start_time)

    print("started!")

    while recorder.is_running():
        time.sleep(1)

    recorder.finish()
    grabber.join()



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
def generate_splits():
    with open("./selenium_scans/metadata/visited_list.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        links = {line["url"]: line["uuid"] for line in csv_reader}
    split = [0.8, 0.1, 0.1]
    total_links = len(links)
    training = random.sample([(url, uuid) for url, uuid in links.items()], k = int(split[0] * total_links))
    for t in training:
        links.pop(t[0])
    testing = random.sample([(url, uuid) for url, uuid in links.items()], k = int(split[1] * total_links))
    for t in testing:
        links.pop(t[0])
    validation = [(url, uuid) for url, uuid in links.items()]
    
    data = {}
    with open("./selenium_scans/metadata/training.json", "w") as f:
        data["items"] = [{"url": url, "uuid": uuid} for url, uuid in training]
        json.dump(data, f)
    with open("./selenium_scans/metadata/testing.json", "w") as f:
        data["items"] = [{"url": url, "uuid": uuid} for url, uuid in testing]
        json.dump(data, f)
    with open("./selenium_scans/metadata/validation.json", "w") as f:
        data["items"] = [{"url": url, "uuid": uuid} for url, uuid in validation]
        json.dump(data, f)


@click.command()
def visualise() -> None:
    show_graph()


@click.command()
def print_database():
    MongoDBInterface.connect()

    ocr_model = list(MongoDBInterface.get_items({"version":"0.1.0"}, "ocr-models"))[0]
    ocr_model = from_dict(OCRModel, ocr_model)
    platform = list(MongoDBInterface.get_items({"metadata":{"owner":"Iason"}}, "platforms"))[0]
    platform = from_dict(Platform, platform)
    print(ocr_model, platform)

    screenshot = list(MongoDBInterface.get_items({}, "screenshots"))
    ocr = list(MongoDBInterface.get_items({}, "image-text-content"))

    print(len(screenshot))
    print(len(ocr))
    '''
    for i in range(len(screenshot)):
        #print(f"_id {screenshot[i]['_id']}, uuid {screenshot[i]['uuid']}, platform {screenshot[i]['platform_id']}, hash {screenshot[i]['hash']}")
        screenshot_id.append(screenshot[i]['_id'])
    for i in range(len(ocr)):
        print('\n')
        print(f"_id {ocr[i]['_id']}, screehshot {ocr[i]['screenshot_id']}")
        print(f"text {ocr[i]['text']}")
        print(f"confidence {ocr[i]['confidence']}")
    '''


@click.command()
def add_model_weights():
    MongoDBInterface.connect()
    model2version = {"web7kbal":"v0.1.0", "web350k":"v0.2.0", "vins":"v0.3.0", "interactable-detector":"v0.4.0", "screensimilarity":"v1.0.0"}
    gdrive_urls = {"web7kbal":"https://drive.google.com/file/d/1QQVmG6u4jgmptT-iMJdS_ESdEWwuC9U2/view?usp=sharing", 
                   "web350k":"https://drive.google.com/file/d/1WwgONDUkrQSc8NwokL1ePJ_OA3NQh17t/view?usp=sharing", 
                   "vins":"https://drive.google.com/file/d/16a-_TKxAaVYTuWeAdTJVWNW5LXLBeNuY/view?usp=sharing", 
                   "interactable-detector":"https://drive.google.com/file/d/1C_GKcmW8WdDNpVN4eNQdqWCKJ--hct19/view?usp=sharing",
                   "screensimilarity":"https://drive.google.com/file/d/17eOpWioh9lvYwN4XGhNvFyoiBOIDQEPW/view?usp=sharing",}
    detector_items = []
    for model in model2version:
        item = {"version": model2version[model], 
                "datetime": datetime.now(), 
                "url": gdrive_urls[model],
                "metadata": {'model-name': model},
                }
        detector_items.append(item)

    MongoDBInterface.add_items(detector_items, "screensimilarity")


@click.command()
def add_ocr_data():
    MongoDBInterface.connect()
    platform = list(MongoDBInterface.get_items({"metadata":{"owner":"Iason"}}, "platforms"))[0]
    platform_id = platform['_id']
    ocr_model = list(MongoDBInterface.get_items({"version":"0.1.0"}, "ocr-models"))[0]
    ocr_version = ocr_model['version']

    khan_ocr = KhanOCR(img_paths_file="./selenium_scans/metadata/domain_map.json")
    all_image_paths = khan_ocr.get_all_img()

    # add Screenshot collection
    db_screenshots = []

    for img_uuid in tqdm(all_image_paths, desc='Screenshots collection'):
        path = all_image_paths[img_uuid]

        img = Image.open(path)
        img_hash = str(average_hash(img))

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        db_image = {
            "uuid": img_uuid,
            "platform_id": platform_id,
            "hash": img_hash,
            "image": img_bytes.getvalue(),
        }
        db_screenshots.append(db_image)
    insert_ids_screenshots = MongoDBInterface.add_items(db_screenshots, "screenshots")

    # add OCR data collection
    with open("./selenium_scans/metadata/uuid2ocr_base.json", "r") as f:
        ocr_base = json.load(f)
    
    db_ocr = []
    for image_id in tqdm(insert_ids_screenshots, desc='Image Text Content collection'):
        img_uuid = list(MongoDBInterface.get_items({"_id":image_id}, "screenshots"))[0]['uuid']
        text_location = list(map(lambda doc: doc[0], ocr_base[img_uuid]))
        text = list(map(lambda doc: doc[1], ocr_base[img_uuid]))
        condifence = list(map(lambda doc: doc[2], ocr_base[img_uuid]))

        db_image_ocr = {
            "screenshot_id": image_id,
            "ocr_version": ocr_version,
            "text_location": text_location,
            "text": text,
            "confidence": condifence,
        }
        db_ocr.append(db_image_ocr)
    insert_ids_ocr = MongoDBInterface.add_items(db_ocr, "image-text-content")


@click.command()
def analyse_ocr():
    MongoDBInterface.connect()
    ocr_model = list(MongoDBInterface.get_items({"version":"0.1.0"}, "ocr-models"))[0]
    ocr_version = ocr_model['version']

    ocr_data = MongoDBInterface.get_items({"ocr_version":ocr_version}, "image-text-content")
    ocr_data: list[ImageTextContent] = from_dict_list(ImageTextContent, ocr_data)

    confidence_threshold = 0
    ocr_text = {}
    for d in ocr_data:
        filtered_text = [t for t, c in zip(d.text, d.confidence) if c > confidence_threshold]
        text = " ".join(filtered_text)
        ocr_text[d.screenshot_id] = text
    
    tokenizer = TFIDF_Tokenizer(
        [" ","+","[","]",":","_","|","$",",","-",".","?","!",],
        [TFIDF_LowerCaseFilter],
        1
    )
    index = TFIDF_Index.index_data(ocr_text, tokenizer)
    depth = -1
    matches = index.find_all_to_all_match(depth = depth)
    match_array = np.array([[score for _, score in sorted(match.items())] for _, match in sorted(matches.items())])
    labels = [id for id, _ in sorted(matches.items())]
    print(f" t = {confidence_threshold}, depth = {depth}, rms = {np.std(match_array)}")

    ax = sns.heatmap(match_array, xticklabels = labels, yticklabels = labels)
    plt.title(f"depth = {depth}, threshold = {confidence_threshold}")
    plt.show()


@click.command()
@click.argument("id")
def show_image_by_id(id):
    _id = ObjectId(id)

    MongoDBInterface.connect()
    screenshot = MongoDBInterface.get_items({"_id":_id}, "screenshots").limit(1)
    screenshot = list(screenshot)[0]
    image = screenshot["image"]
    image = Image.open(io.BytesIO(image))
    image.show()


@click.command()
@click.option("--img", required=False, default="", type=str, help="Image to get bboxes")
@click.option("--shortlist_threshold", required=False, default=0.5, type=float, help="Lower threshold for interactables")
@click.option("--nms_iou_threshold", required=False, default=0.2, type=float, help="Upper threshold for IoU NMS")
def shortlist_image_bbox(
    img: str = "",
    shortlist_threshold: float = 0.5,
    nms_iou_threshold: float = 0.2,
):
    if img == "":
        img = Image.open("./shortlist_images/image_raw.png")
        #img = Image.open("./Explorer/trace_similarity/test_action_matching/Ex2_ArnasToIasonScreen/iason1.png")
    else:
        img = Image.open(img)
    shortlister = Shortlister()

    shortlister.set_img(img)

    #shortlister.set_model("ocr").set_bboxes().save()
    shortlister.set_model("interactable-detector").set_bboxes().show()#.save()#
    #shortlister.set_model("vins").set_bboxes().save()
    #shortlister.set_model("web350k").set_bboxes().save()
    #shortlister.set_model("web7kbal").set_bboxes().save()

@click.command()
def objective_1():
    qt_app = QApplication(sys.argv)
    app = Objective("interactable-detector")
    app.show()

    speech2text = Speech2Text()
    speech2text.attach_exec(
        exec_func = lambda : QApplication.postEvent(
            app,
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        ),
        target = CommandPhrase.SHOW
    )
    speech2text.attach_exec(
        exec_func = lambda idx: QApplication.postEvent(
            app,
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, text = str(idx))
        ),
        target = CommandPhrase.CLICK
    )
    speech2text.attach_exec(
        exec_func = lambda : QApplication.postEvent(
            app,
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        ),
        target = CommandPhrase.STOP
    )
    speech2text.listen()

    qt_app.exec_()
    speech2text.stop_listen()

@click.command()
def speech_execution():
    speech2text = Speech2Text(verbose=False)
    
    #speech2text.attach_exec(exec_func=None, target='show')
    #speech2text.attach_exec(exec_func=None, target='click')
    #speech2text.attach_exec(exec_func=None, target='stop')

    speech2text.listen()
    while 1: #for _ in range(10):
        time.sleep(1)
        speech2text.disp()
    speech2text.stop_listen()

@click.command()
@click.option("--include_ocr", required=False, default=False, type=bool, help="Use Screensim model + OCR embedding distance")
def trace_sim(include_ocr):
    def uuid2image(uuid: str) -> Image.Image:
        path = os.path.join('./selenium_scans/screenshots', uuid + '.png')
        image = Image.open(path)
        return image

    screensim = ScreenSimilarity()
    trace_frames = [
        'cd263a3ee2de4b239f913cec26e01b2f', '60108111cf3a4104b52b57d907c85ff0', 'fa0f7067dac845abb3fa50e387d40e46',
        'a050de9a20bd4b18838ffadf8e7700a6', '15d6c0f784864b3a8ba294a80d4c9a00', '405d8166eb864a828fe9d37f13931087',
        '2681a9d5e5be47f28d9f1c861ccdd64c', '8bab9972d17c41e280fc10bdef997226', '817456f70e744ec1a235368d1e88a94c',
    ]
    for i in range(len(trace_frames)):
        trace_frames[i] = uuid2image(trace_frames[i])
    screensim.trace_self_similarity(trace_frames, include_ocr=include_ocr)


@click.command()
@click.option("--include_ocr_top_k", required=False, default=0, type=int, help="Use Screensim model + OCR embedding distance")
def action_matching(include_ocr_top_k):
    test_path = './Explorer/trace_similarity/test_action_matching/Ex3_ArnasToIasonScreen'

    def name2image(name: str) -> Image.Image:
        path = os.path.join(test_path, name + '.png')
        image = Image.open(path)
        return image
    
    action_matcher = ActionMatching()
    # Example 1 - Iason to Iason
    '''image_user1 = name2image('user1_khanexample1')
    image_user2 = name2image('user2_khanexample1')
    image_user3 = name2image('user3_khanexample1')
    click_user1 = (150, 690)
    bbox_user1 = (33,627,336,719)'''

    # Example 2 - Nino (user1) to Iason (user2)
    '''state_info = { # state: user1name, user2name, user1click
        'state1': {'user1': 'nino1', 'user2': 'iason1', 'user1_click': (467,1218)},
        'state2': {'user1': 'nino2', 'user2': 'iason2', 'user1_click': (1548,875)},
        #'state3': {'user1': 'nino3', 'user2': 'iason3', 'user1_click': (1202,1380)},
        'state4': {'user1': 'nino4', 'user2': 'iason4', 'user1_click': (2307,1447)},
    }'''

    # Example 3 - Arnas to Iason
    state_info = { # state: user1name, user2name, user1click
        'state1': {'user1': 'arnas1', 'user2': 'iason1', 'user1_click': (1237,672)},
        'state2': {'user1': 'arnas1', 'user2': 'iason1', 'user1_click': (69,671)},
        'state3': {'user1': 'arnas1', 'user2': 'iason1', 'user1_click': (1080,178)},
        'state4': {'user1': 'iason1', 'user2': 'arnas1', 'user1_click': (1769,642)},
        'state5': {'user1': 'iason1', 'user2': 'arnas1', 'user1_click': (78,651)},
        'state6': {'user1': 'iason1', 'user2': 'arnas1', 'user1_click': (1316,147)},
    }
    mode = 'resized_full'

    for state in state_info:
        print('state compared: ', state)
        image_user1 = name2image(state_info[state]['user1'])
        image_user2 = name2image(state_info[state]['user2'])
        click_user1 = state_info[state]['user1_click']
        
        bbox_clicked_user1 = action_matcher.get_interactable_at_click(image_user1, click_user1)
        best_bbox_user2 = action_matcher.interactable_matching(
            image_user1, 
            image_user2, 
            click_user1, 
            mode,
            include_ocr_top_k=include_ocr_top_k,
            show_user1=True,
        )
        action_matcher.show(image_user1, bbox_clicked_user1, "green", savedir=None) #savedir=os.path.join(test_path, 'user1_click_bbox.png'))
        action_matcher.show(image_user2, best_bbox_user2, "red", savedir=None) #savedir=os.path.join(test_path, f'{next_user}_best_bbox_{mode}.png'))

        input()

@click.command()
def process_trace():
    processor = TraceVisualiser()
    processor.make_gif()
    processor.calculate_embeddings().load_screenshot_similarities()

    processor.start_plot().plot_similarities().end_plot()
    processor.start_plot().plot_similarities().plot_similarities_moving_average(10).plot_left_click().end_plot()

    #processor.start_plot() \
    #.plot_similarities() \
    #.plot_similarities_moving_average(10) \
    #.plot_state_change_detector(10, 0.5) \
    #.end_plot()

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
main.add_command(generate_splits)
main.add_command(print_database)
main.add_command(add_model_weights)
main.add_command(add_ocr_data)
main.add_command(analyse_ocr)
main.add_command(show_image_by_id)
main.add_command(shortlist_image_bbox)
main.add_command(speech_execution)
main.add_command(objective_1)
main.add_command(trace_sim)
main.add_command(action_matching)
main.add_command(process_trace)


if __name__ == "__main__":
    main()
