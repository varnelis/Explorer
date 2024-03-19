from collections import OrderedDict
import os
from uuid import UUID
import matplotlib.pyplot as plt
import csv
import glob
from PIL import Image
import numpy as np
from torch import Tensor
from tqdm import tqdm
import contextlib

from Explorer.trace_similarity.screen_similarity import ScreenSimilarity

class TraceProcessor:
    def __init__(self) -> None:
        self._fig = None
        self._ax = None

        self.raw_states = []
        with open("./temp/capture_log.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            self.raw_states = list(reader)
        print("raw traces loaded")
        
        self.screen_similarity_model = ScreenSimilarity()
        self.screenshot_embeddings: dict[UUID, Tensor] = {}
        self.screenshot_timestamps: dict[UUID, any] = {}
        self.screen_similarities: tuple[list[float], list] | None = None

        with open("./temp/screenshot_timestamps.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for entry in reader:
                self.screenshot_timestamps[entry["uuid"]] = int(entry["timestamp"])
        print("screenshot timestamps loaded")

    def calculate_embeddings(self):
        for img_name in tqdm(glob.glob("./temp/screenshots/*.png")):
            img = Image.open(img_name)
            embedding = self.screen_similarity_model.image2embedding(img)
            uuid = self.extract_uuid(img_name)
            self.screenshot_embeddings[uuid] = embedding
        
        print("embeddings calculated")
        return self

    def extract_uuid(self, filepath: str) -> UUID:
        _, tail = os.path.split(filepath)
        return tail.split(".")[0]
            
    def get_left_clicks(self) -> list:
        currently_pressed = False
        left_clicks = []
        for s in self.raw_states:
            state = int(s["mouse_state"])

            if currently_pressed is True and state == 0:
                left_clicks.append(int(s["time(ms)"]))

            currently_pressed = True if state == 2 else False

        return left_clicks
    
    def get_embeddings_sorted_by_time(self) -> list[tuple[Tensor, any]]:
        uuid2embedding = [(k, v) for k, v in self.screenshot_embeddings.items()]
        uuid2embedding.sort(key = lambda x: self.screenshot_timestamps[x[0]])
        return [(v, self.screenshot_timestamps[k]) for k, v in uuid2embedding]
    
    def load_screenshot_similarities(self) -> "TraceProcessor":
        if len(self.screenshot_embeddings) == 0:
            print("Screenshot embeddings have not been calculated yet, please do that first by calling calculate_embeddings().")
            return

        embeddings = self.get_embeddings_sorted_by_time()
        similarities = [0.0]
        timestamps = [embeddings[0][1]]

        pb = tqdm(total = len(embeddings) - 1)
        for e1, e2 in zip(embeddings, embeddings[1:]):
            similarity, _ = self.screen_similarity_model.embeddings2similarity(e1[0], e2[0])
            similarities.append(similarity)
            timestamps.append(e2[1])
            pb.update()
        
        self.screen_similarities = (similarities, timestamps)

        print("screen similarities calculated")
        return self

class TraceVisualiser(TraceProcessor):
    def __init__(self) -> None:
        super().__init__()
    
    def make_gif(self):
        fp_img = glob.glob("./temp/screenshots/*.png")
        fp_img.sort(
            key = lambda x: self.screenshot_timestamps[self.extract_uuid(x)]
        )
        fp_out = "./temp/trace.gif"

        with contextlib.ExitStack() as stack:
            imgs = (stack.enter_context(Image.open(f)) for f in fp_img)
            img = next(imgs)
            img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
        
        print("gif created")
        return self
    
    def start_plot(self) -> "TraceVisualiser":
        self._fig, self._ax = plt.subplots()
        return self
    
    def end_plot(self):
        plt.show()

    def plot_left_click(self) -> "TraceVisualiser":
        timestamps = self.get_left_clicks()
        y = [1] * len(timestamps)

        self._ax.stem(timestamps, y)
        return self
    
    def plot_similarities(self) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        self._ax.plot(timestamps, similarities)
        return self

    def plot_similarities_moving_average(self, n: int) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        avg_similarities = _DataProcessor.moving_average(similarities, n)

        self._ax.plot(timestamps, avg_similarities)
        return self

    def plot_state_change_detector(self, n: int, height: float = 1) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        avg_similarities = _DataProcessor.moving_average(similarities, n)
        edges, values = _DataProcessor.comparator(avg_similarities, timestamps, 0.2)
        values = values * height

        self._ax.stairs(values, edges)
        return self

class _DataProcessor:
    @classmethod
    def moving_average(cls, data: list[float], window: int) -> np.ndarray:
        avg_data = np.zeros(len(data))
        data = [0] * (window - 1) + data
        for i in range(window, len(data) + 1):
            avg = np.sum(data[i-window:i]) / window
            avg_data[i - window] = avg

        return avg_data
    
    @classmethod
    def comparator(cls, data_y: list[float], data_x: list[float], threshold: float) -> tuple[np.ndarray, list[float]]:
        edges = [data_x[0]]
        values = np.array([])

        state = 0

        for y, x in zip(data_y, data_x):
            if y > threshold and state == 0:
                state = 1
                edges.append(x)
                values = np.append(values, 1)
            if y < threshold and state == 1:
                state = 0
                edges.append(x)
                values = np.append(values, 0)
        return edges, values
