from collections import deque
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
import json

from Explorer.trace_similarity.screen_similarity import ScreenSimilarity

class Buffer(deque):
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__([0] * maxsize)

    def add(self, item):
        self.popleft()
        self.append(item)

class RollingStateProcessor:
    def __init__(self, roll_window, u_threshold, l_threshold, static_trigger: callable, changing_trigger: callable) -> None:
        self.is_static = False
        self.u_threshold = u_threshold
        self.l_threshold = l_threshold
        self.state_is_changing_trigger = changing_trigger
        self.state_is_static_trigger = static_trigger

        self.screen_similarity_model = ScreenSimilarity()

        self.last_embedding: Tensor | None = None
        self.screen_similarities = Buffer(roll_window)
        self.timestamps = Buffer(roll_window)
    
    def add_img(self, img: Image.Image, timestamp) -> Tensor:
        embedding = self.screen_similarity_model.image2embedding(img)
        if self.last_embedding is not None:
            d_embedding, is_same = self.screen_similarity_model.embeddings2similarity(self.last_embedding, embedding)
            self.screen_similarities.add(d_embedding)
            self.timestamps.add(timestamp)
        self.last_embedding = embedding
        self.update_state()
    
    def update_state(self):
        similarity = sum(self.screen_similarities)
        if self.is_static is True and similarity > self.u_threshold:
            self.is_static = False
            self.state_is_changing_trigger()
        elif self.is_static is False and similarity < self.l_threshold:
            self.is_static = True
            self.state_is_static_trigger()

class TraceProcessor:
    def __init__(self, folder: str = "./temp") -> None:
        self.folder = folder
        self._fig = None
        self._ax = None

        self.raw_states = []
        with open(f"{folder}/capture_log.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            self.raw_states = list(reader)
        print("raw traces loaded")
        
        self.screen_similarity_model = ScreenSimilarity()
        self.screenshot_embeddings: dict[UUID, Tensor] = {}
        self.screenshot_timestamps: dict[UUID, any] = {}
        self.screen_similarities: tuple[list[float], list] | None = None

        with open(f"{folder}/screenshot_timestamps.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for entry in reader:
                self.screenshot_timestamps[entry["uuid"]] = int(entry["timestamp"])
        print("screenshot timestamps loaded")

    def calculate_embeddings(self):
        for img_name in tqdm(glob.glob(f"{self.folder}/screenshots/*.png")):
            img = Image.open(img_name)
            embedding = self.screen_similarity_model.image2embedding(img)
            uuid = self.extract_uuid(img_name)
            self.screenshot_embeddings[uuid] = embedding
        
        print("embeddings calculated")
        return self

    def extract_uuid(self, filepath: str) -> UUID:
        _, tail = os.path.split(filepath)
        return tail.split(".")[0]
            
    def get_left_clicks(self) -> tuple[list, list]:
        currently_pressed = False
        timestamps = []
        positions = []
        for s in self.raw_states:
            state = int(s["mouse_state"])

            if currently_pressed is True and state == 0:
                timestamps.append(int(s["time(ms)"]))
                positions.append((int(s["x"]), int(s["y"])))

            currently_pressed = True if state == 2 else False

        return timestamps, positions
    
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

    def generate_trace_recording(
            self,
            window: int = 10,
            u_threshold: float = 0.05,
            l_thresholds: float = 0.05
        ):

        timestamps_left_clicks, positions_left_clicks = self.get_left_clicks()

        similarities, timestamps_similarities = self.screen_similarities
        avg_similarities = _DataProcessor.moving_average(similarities, window)
        edges, values = _DataProcessor.comparator(
            avg_similarities,
            timestamps_similarities,
            u_threshold,
            l_thresholds
        )

        timestamps_state_change = [e for e, v in zip(edges, values) if v == 1]
        current_state_change = timestamps_state_change.pop(0)
        timestamps_target_clicks = []
        positions_target_clicks = []
        for prev_t, curr_t, prev_p in zip(timestamps_left_clicks, timestamps_left_clicks[1:], positions_left_clicks):
            if curr_t < current_state_change:
                continue

            timestamps_target_clicks.append(prev_t)
            positions_target_clicks.append(prev_p)
            if len(timestamps_state_change) == 0:
                break
            current_state_change = timestamps_state_change.pop(0)
        if len(timestamps_state_change) != len(timestamps_target_clicks) and current_state_change > timestamps_left_clicks[-1]:
            timestamps_target_clicks.append(timestamps_left_clicks[-1])
            positions_target_clicks.append(positions_left_clicks[-1])


        current_click = timestamps_target_clicks.pop(0)
        target_states = []
        for prev, curr in zip(timestamps_similarities, timestamps_similarities[1:]):
            if curr < current_click:
                continue

            target_states.append(prev)
            if len(timestamps_target_clicks) == 0:
                break
            current_click = timestamps_target_clicks.pop(0)
        
        target_states.append(timestamps_similarities[-1])

        print(f"target_states: {target_states}")
        print(f"clicks: {timestamps_left_clicks}")

        def get_action(uuid, pos):
            if pos is None:
                return {
                    "img": uuid,
                    "action": "None"
                }

            return {
                "img": uuid,
                "action": {
                    "type": "left_click",
                    "position": pos
                }
            }

        trace_dict = {
            "trace" : {
                "state_action_pairs" : []
            }
        }

        uuid2timestamp = [(key, val) for key, val in self.screenshot_timestamps.items()]
        uuid2timestamp.sort(key = lambda x: x[1])
        current_target_state = target_states.pop(0)
        for uuid, t in uuid2timestamp:
            if t != current_target_state:
                continue
            
            if len(target_states) == 0:
                trace_dict["trace"]["state_action_pairs"].append(get_action(uuid, None))
                break
            trace_dict["trace"]["state_action_pairs"].append(get_action(uuid, positions_target_clicks.pop(0)))
            current_target_state = target_states.pop(0)

        with open(f"{self.folder}/processed_trace.json", "w") as f:
            json.dump(trace_dict, f)


class TraceVisualiser(TraceProcessor):
    def __init__(self, path: str) -> None:
        super().__init__(path)
    
    def make_gif(self):
        fp_img = glob.glob(f"{self.folder}/screenshots/*.png")
        fp_img.sort(
            key = lambda x: self.screenshot_timestamps[self.extract_uuid(x)]
        )
        fp_out = f"{self.folder}/trace.gif"

        with contextlib.ExitStack() as stack:
            imgs = (stack.enter_context(Image.open(f)) for f in fp_img)
            img = next(imgs)
            img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=10, loop=0)
        
        print("gif created")
        return self
    
    def start_plot(self) -> "TraceVisualiser":
        self._fig, self._ax = plt.subplots()
        return self
    
    def end_plot(self):
        plt.legend()
        plt.show()

    def plot_left_click(self, height: float = 1) -> "TraceVisualiser":
        timestamps, _ = self.get_left_clicks()
        y = [height] * len(timestamps)

        self._ax.stem(timestamps, y, label = "action")
        return self
    
    def plot_similarities(self) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        self._ax.plot(timestamps, similarities, label = "similarities")
        return self

    def plot_similarities_moving_average(self, n: int) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        avg_similarities = _DataProcessor.moving_average(similarities, n)

        self._ax.plot(timestamps, avg_similarities, label = "moving avg. sim.")
        return self

    def plot_state_change_detector(self, window: int, height: float = 1, l_threshold = 0.05, u_threshold = 0.05) -> "TraceVisualiser":
        similarities, timestamps = self.screen_similarities
        avg_similarities = _DataProcessor.moving_average(similarities, window)
        edges, values = _DataProcessor.comparator(avg_similarities, timestamps, u_threshold, l_threshold)
        values = values * height

        self._ax.stairs(values, edges, label = "state change")
        return self

    def add_title(self, name: str) -> "TraceVisualiser":
        plt.title(name)
        return self
    
class TraceProcessorUser1:
    def __init__(self, folder: str = "./temp") -> None:
        with open(f"{folder}/processed_trace.json") as f:
            self.processed_trace_user_1 = json.load(f)
        
    def get_trace_length(self) -> int:
        """ Number of states in User 1 processed trace """
        return len(self.processed_trace_user_1["trace"]["state_action_pairs"])

    def get_state_action_n(self, n: int) -> tuple[Image.Image, tuple[float,float]]:
        """ Return Image & Action Position for nth state in User 1 processed Trace """
        state_action_pair_n = self.processed_trace_user_1["trace"]["state_action_pairs"][n]
        state_uuid = state_action_pair_n["img"]
        state = Image.open(f"{self.folder}/screenshots/{state_uuid}.png")
        if state_action_pair_n["action"] != "None":
            action = state_action_pair_n["action"]["position"]
        else:
            action = None

        return state, action

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
    def comparator(cls, data_y: list[float], data_x: list[float], u_threshold: float, l_threshold: float) -> tuple[np.ndarray, list[float]]:
        edges = [data_x[0]]
        state = 0
        values = np.array([state])

        for y, x in zip(data_y, data_x):
            if y > u_threshold and state == 0:
                state = 1
                edges.append(x)
                values = np.append(values, 1)
            if y < l_threshold and state == 1:
                state = 0
                edges.append(x)
                values = np.append(values, 0)
        edges.append(data_x[-1])
        return edges, values
