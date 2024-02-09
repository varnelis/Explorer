import csv
from itertools import compress
import json
import os
import time
import random
from typing import Any
from uuid import UUID, uuid4
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.action_chains import ActionChains

from selenium.common.exceptions import ElementClickInterceptedException

from PIL import Image, ImageDraw
from tqdm import tqdm


class SeleniumScanner:
    driver: WebDriver | None = None
    viewport: dict | None = None

    current_url: str | None = None
    current_uuid: UUID | None = None
    current_screen: Image.Image | None = None
    dir = "./selenium_scans/"
    bbox: list[tuple[int, int, int, int, WebElement]] = []
    scale_factor: int | None = None

    @classmethod
    def prepare_directories(cls):
        dir = cls.dir
        if not os.path.exists(dir):
            os.mkdir(dir)

        if not os.path.exists(dir + "annotations"):
            os.mkdir(dir + "annotations")

        if not os.path.exists(dir + "bbox_screenshots"):
            os.mkdir(dir + "bbox_screenshots")

        if not os.path.exists(dir + "metadata"):
            os.mkdir(dir + "metadata")

        if not os.path.isfile(dir + "metadata/visited_list.csv"):
            with open(dir + "metadata/visited_list.csv", "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["url", "uuid"])

        if not os.path.exists(dir + "screenshots"):
            os.mkdir(dir + "screenshots")

    @classmethod
    def setup_driver(cls, scale_factor):
        cls.driver = webdriver.Chrome()
        cls.scale_factor = scale_factor

    @classmethod
    def load_url(cls, url: str):
        cls.current_screen = None
        cls.bbox = []

        cls.current_url = url
        cls.current_uuid = uuid4()
        cls.driver.get(url)
        cls.viewport = cls.driver.get_window_size()

    @classmethod
    def scroll_page(cls):
        cls.current_uuid = uuid4()
        ActionChains(cls.driver).scroll_by_amount(0, cls.viewport["height"]).perform()
        time.sleep(2)
    
    @classmethod
    def end_of_page(cls):
        old_y_pos = cls.driver.execute_script("return window.pageYOffset;")
        yield False

        new_y_pos = cls.driver.execute_script("return window.pageYOffset;")
        print(f"{old_y_pos} -> {new_y_pos}")
        while new_y_pos != old_y_pos:
            yield False
            old_y_pos = new_y_pos
            new_y_pos = cls.driver.execute_script("return window.pageYOffset;")
            print(f"{old_y_pos} -> {new_y_pos}")
        yield True

    @classmethod
    def expand_button(cls, click_rand_button_p):
        click_probability = random.random()
        if click_probability > click_rand_button_p:
            return
        
        expandable_buttons: list[WebElement] = []
        expandable_buttons += cls.driver.find_elements(
            By.CSS_SELECTOR, '[aria-expanded="false"]'
        )
        rand_button = expandable_buttons[
            random.randint(0, len(expandable_buttons) - 1)
        ]
        try:
            rand_button.click()
        except:# ElementClickInterceptedException:
            # Element not interactable
            return 0

    @classmethod
    def load_screenshot(cls):
        if cls.current_url is None:
            return

        time.sleep(5)
        file_path = cls.dir + "screenshots/" + cls.current_uuid.hex + ".png"
        cls.driver.save_screenshot(file_path)
        cls.current_screen = Image.open(file_path)

    @classmethod
    def load_bbox(cls):
        elements: list[WebElement] = []
        elements += cls.get_buttons()
        elements += cls.get_links()
        elements += cls.get_inputs()

        # repeat for all other interactable types

        elements = cls.filter_elements_by_area(elements)
        elements = cls.deduplicate_elements(elements)
        elements = cls.filter_elements_by_visibility(elements)

        cls.bbox = []
        for e in elements:
            pos = e.location
            dim = e.size
            cls.bbox.append(
                (
                    pos["x"],
                    pos["y"],
                    (pos["x"] + dim["width"]),
                    (pos["y"] + dim["height"]),
                    e,
                )
            )

        cls.bound_bbox_by_view_port()
        cls.bound_bbox_by_parents()
        cls.overlay_bbox_by_header()
        cls.bbox.sort(key=lambda x: x[1])

    @classmethod
    def draw_bbox(cls):
        if len(cls.bbox) == 0 or cls.current_screen is None:
            return
        
        viewport_dy = cls.driver.execute_script("return window.pageYOffset;")
        draw_bbox = ImageDraw.Draw(cls.current_screen)
        for i, bbox in enumerate(cls.bbox):
            shifted_bbox = [bbox[0], bbox[1] - viewport_dy, bbox[2], bbox[3] - viewport_dy]
            scaled_bbox = [e * cls.scale_factor for e in shifted_bbox]
            draw_bbox.rectangle(scaled_bbox, outline="black", width=3)

    @classmethod
    def save_scan(cls):
        bbox_screenshot_path = (
            cls.dir + "bbox_screenshots/bbox-" + cls.current_uuid.hex + ".png"
        )
        annotation_path = cls.dir + "annotations/" + cls.current_uuid.hex + ".json"
        visited_path = cls.dir + "metadata/visited_list.csv"

        with open(annotation_path, "w") as f:
            json.dump(
                {
                    "id": cls.current_uuid.hex,
                    "clickable": [{"bbox": list(b[:4])} for b in cls.bbox],
                },
                f,
            )
        cls.current_screen.save(bbox_screenshot_path)

        with open(visited_path, "a+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow([cls.current_url, cls.current_uuid.hex])

    @classmethod
    def get_buttons(cls) -> list[WebElement]:
        buttons = cls.driver.find_elements(By.XPATH, "//button")
        buttons += cls.driver.find_elements(By.XPATH, '//*[@role="button"]')
        return buttons

    @classmethod
    def get_links(cls) -> list[WebElement]:
        # I should not be doing this (excluding # elements)
        links = cls.driver.find_elements(By.XPATH, "//a[@href]")
        return links

    @classmethod
    def get_inputs(cls) -> list[WebElement]:
        inputs = cls.driver.find_elements(By.XPATH, "//input")
        return inputs

    @classmethod
    def filter_elements_by_area(cls, elements: list[WebElement], area_limit: int = 10):
        filtered_elements = []
        for e in elements:
            size = e.size
            area = size["width"] * size["height"]
            if area < 10:
                continue
            filtered_elements.append(e)
        return filtered_elements

    @classmethod
    def deduplicate_elements(cls, elements: list[WebElement]):
        existing = []
        filtered_elements = []
        for e in elements:
            pos = e.location
            size = e.size
            key = (pos["x"], pos["y"], size["width"], size["height"])
            if key in existing:
                continue
            existing.append(key)
            filtered_elements.append(e)
        return filtered_elements

    @classmethod
    def filter_elements_by_visibility(cls, elements: list[WebElement]):
        filtered_elements = []

        for e in elements:
            is_visible = cls.driver.execute_script(
                "return arguments[0].offsetParent !== null;", e
            )

            if not e.is_displayed():
                continue
            if not is_visible:
                continue

            filtered_elements.append(e)

        return filtered_elements

    @classmethod
    def bound_bbox_by_view_port(cls):
        active_mask = []

        vp_left, vp_top = cls.driver.execute_script("return [window.pageXOffset, window.pageYOffset];")
        vp_right = vp_left + cls.viewport["width"]
        vp_bottom = vp_top + cls.viewport["height"]

        for i, e in enumerate(cls.bbox):
            element = e[-1]
            left, top, right, bottom = e[:4]

            left = max(left, vp_left)
            top = max(top, vp_top)
            right = min(right, vp_right)
            bottom = min(bottom, vp_bottom)

            cls.bbox[i] = (left, top, right, bottom, element)
            active_mask.append(left < right and top < bottom)
        cls.bbox = list(compress(cls.bbox, active_mask))

    @classmethod
    def print_parents_bbox(cls, elements: list[WebElement]):
        sorted_elements = sorted(elements, key=lambda x: x.location["y"])
        target = sorted_elements[19]
        print(f"pos: {target.location}, size {target.size}")
        ancestors = target.find_elements(By.XPATH, "ancestor::*")
        for a in ancestors:
            print(f"\t pos: {a.location}, size {a.size}")

    @classmethod
    def bound_bbox_by_parents(cls):
        active_mask = []
        pb = tqdm(total=len(cls.bbox), position=1)
        for i, e in enumerate(cls.bbox):
            element = e[-1]
            left, top, right, bottom = e[:4]

            ancestors = element.find_elements(By.XPATH, "ancestor::*")
            for a in ancestors:
                pos = a.location
                a_left, a_top = pos["x"], pos["y"]
                size = a.size
                a_right, a_bottom = pos["x"] + size["width"], pos["y"] + size["height"]
                if a_left == 0 and a_top == 0:
                    continue
                left = max(left, a_left)
                top = max(top, a_top)
                right = min(right, a_right)
                bottom = min(bottom, a_bottom)
            cls.bbox[i] = (left, top, right, bottom, element)
            active_mask.append(left < right and top < bottom)
            pb.update()
        cls.bbox = list(compress(cls.bbox, active_mask))

    @classmethod
    def overlay_bbox_by_header(cls):
        header = cls.driver.find_element(By.XPATH, "//div[@id='top-header-container']")
        pos = header.location
        _, h_top = pos["x"], pos["y"]
        size = header.size
        _, h_bottom = pos["x"] + size["width"], pos["y"] + size["height"]
        print(f"pos {pos}, size {size}")

        new_bbox = []
        
        for e in cls.bbox:
            element = e[-1]

            is_header_ancestor = cls.driver.execute_script(
                """
                var ancestor = arguments[0];
                var child = arguments[1];
                return ancestor.contains(child);
                """,
                header,
                element
            )
            if is_header_ancestor is True:
                new_bbox.append(e)
                continue

            left, top, right, bottom = e[:4]

            if top > h_top and bottom < h_bottom:
                continue
            if h_bottom > top > h_top:
                top = h_bottom
            if h_bottom > bottom > h_top:
                bottom = h_top
            new_bbox.append([left, top, right, bottom, element])
        cls.bbox = new_bbox
