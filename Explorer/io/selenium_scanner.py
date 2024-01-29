from itertools import compress
import time
from typing import Any
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from PIL import Image, ImageDraw
from tqdm import tqdm

class SeleniumScanner:

    driver: WebDriver | None = None

    current_url: str | None = None
    current_hash: int | None = None
    current_screen: Image.Image | None = None
    dir = "./selenium_scans/"
    bbox: list[tuple[int, int, int, int, WebElement]] = []
    scale_factor = 2

    @classmethod
    def setup_driver(cls):
        cls.driver = webdriver.Chrome()

    @classmethod
    def load_url(cls, url: str):
        cls.current_screen = None
        cls.bbox = []

        cls.current_url = url
        cls.current_hash = hash(url)
        cls.driver.get(url)

    @classmethod
    def load_screenshot(cls):
        if cls.current_url is None:
            return
        
        time.sleep(5)
        file_path = cls.dir + str(cls.current_hash) + ".png"
        cls.driver.save_screenshot(file_path)
        cls.current_screen = Image.open(file_path)
    
    @classmethod
    def load_bbox(cls):
        elements: list[WebElement] = []
        elements += cls.get_buttons()
        elements += cls.get_links()
        
        # repeat for all other interactables

        elements = cls.filter_elements_by_area(elements)
        elements = cls.deduplicate_elements(elements)
        elements = cls.filter_elements_by_visibility(elements)

        cls.print_parents_bbox(elements)

        cls.bbox = []
        for e in elements:
            pos = e.location
            dim = e.size
            cls.bbox.append((
                pos["x"], pos["y"],
                (pos["x"] + dim["width"]),
                (pos["y"] + dim["height"]),
                e
            ))

        cls.bound_bbox_by_parents()
        cls.bbox.sort(key = lambda x: x[1])
    
    @classmethod
    def draw_bbox(cls):
        if len(cls.bbox) == 0 or cls.current_screen is None:
            return
        
        draw_bbox = ImageDraw.Draw(cls.current_screen)
        for i, bbox in enumerate(cls.bbox):
            scaled_bbox = [e * cls.scale_factor for e in bbox[:4]]
            draw_bbox.rectangle(scaled_bbox, outline = "black", width = 3)
        
        file_path = cls.dir + "bbox-" + str(cls.current_hash) + ".png"
        cls.current_screen.save(file_path)

    @classmethod
    def get_buttons(cls) -> list[WebElement]:
        buttons = cls.driver.find_elements(By.XPATH, '//button')
        buttons += cls.driver.find_elements(By.XPATH, '//*[@role="button"]')
        return buttons
    
    @classmethod
    def get_links(cls) -> list[WebElement]:
        # I should not be doing this (excluding # elements)
        links = cls.driver.find_elements(By.XPATH, "//a[@href and not(starts-with(@href, '#'))]")
        return links

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
            is_visible = cls.driver.execute_script("return arguments[0].offsetParent !== null;", e)
        
            if not e.is_displayed():
                continue
            if not is_visible:
                continue

            filtered_elements.append(e)
            
        return filtered_elements
    
    @classmethod
    def print_parents_bbox(cls, elements: list[WebElement]):
        sorted_elements = sorted(elements, key= lambda x: x.location["y"])
        target = sorted_elements[19]
        print(f"pos: {target.location}, size {target.size}")
        ancestors = target.find_elements(By.XPATH, "ancestor::*")
        for a in ancestors:
            print(f"\t pos: {a.location}, size {a.size}")

    @classmethod
    def bound_bbox_by_parents(cls):
        active_mask = []
        pb = tqdm(total = len(cls.bbox))
        for i, e in enumerate(cls.bbox):
            element = e[-1]
            left, top, right, bottom = e[:4]

            ancestors = element.find_elements(By.XPATH, "ancestor::*")
            for a in ancestors:
                pos = a.location
                a_left, a_top = pos["x"], pos["y"]
                size = a.size
                a_right, a_bottom = pos["x"] + size["width"], pos["y"] + size["height"]

                left = max(left, a_left)
                top = max(top, a_top)
                right = min(right, a_right)
                bottom = min(bottom, a_bottom)
            cls.bbox[i] = (left, top, right, bottom, e)
            active_mask.append(left < right and top < bottom)
            pb.update()
        print(cls.bbox[19])
        cls.bbox = list(compress(cls.bbox, active_mask))