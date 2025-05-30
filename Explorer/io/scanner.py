import webbrowser
from Explorer.io.controller import Controller
from Explorer.io.io_state import IOState
from Explorer.io.key_map import AnyButton, AnyKey
from Explorer.io.recorder import Recorder
from pynput.keyboard import Key
import time

from PIL import Image, ImageChops, ImageGrab, ImageOps, UnidentifiedImageError, ImageDraw
import numpy as np
import hashlib
from collections import defaultdict


class Scanner:
    # Scans a given page for left click interactables
    # 1. Opens the page?
    # 2. Moves mouse in the grid style manner
    # 3. @ each position
    #   3.1. checks if there are changes on the screen
    #   3.2. if there are changes, tries to identify the bounding box of the
    #        interactable

    def __init__(
        self,
        url: str,
        prefix: str,
        response: float,
        mode: str = "manual",
        screen_origin: tuple[int, int] | None = None,
        screen_width: int | None = None,
        screen_height: int | None = None,
        url_popup_origin: tuple[int, int] | None = None,
        url_popup_width: int | None = None,
        url_popup_height: int | None = None,
    ):
        self.mode = mode
        self._url = url
        self.prefix = prefix

        self.response_time = response
        self.width_step = 30
        self.height_step = 30
        self.width_space = 10
        self.height_space = 10

        Recorder.record_data = False
        self._recorder = Recorder()
        self._controller: Controller | None = None

        self._updater = IOState()
        self._updater.attach_updater(self.on_click, "mouse_state")
        self._updater.attach_updater(self.on_press, "keyboard_state")

        self.screen_origin = screen_origin
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.url_popup_origin = url_popup_origin
        self.url_popup_width = url_popup_width
        self.url_popup_height = url_popup_height
        self.screen: Image.Image | None = None
        self.interactables_overlay: np.NDArray | None = None
        self.bbox = []
        self.__init_screen()

    def __init_screen(self):
        webbrowser.open(self._url)

        if self.mode == "manual":
            self.get_screen_dimensions()
        else:
            time.sleep(10)

        self.screen = self.take_screen_shot(
            (0, 0), self.screen_width, self.screen_height
        )
        if self.screen is None:
            raise RuntimeError("Unable to capture screen")
        
        self.url_popup = self.take_screen_shot(
            (0, 0),
            self.url_popup_width,
            self.url_popup_height,
            self.url_popup_origin
        )
        if self.url_popup is None:
            raise RuntimeError("Unable to capture screen")
        self.url_popup_active = False
        
        self.interactables_overlay = np.zeros((self.screen_width, self.screen_height))
        self.interactables_bb_overlay = Image.new("L", (self.screen_width, self.screen_height), "white")
        self.interactable_groups = defaultdict(list)
        self.bbox = []

        if self._controller is None:
            self._controller = Controller()
    
    def load_next_url(self, url: str, prefix: str):
        self._url = url
        self.prefix = prefix
        self.__init_screen()
    
    def get_screen_dimensions(self):
        self._screen_bounding_box: list[tuple[int, int]] = []
        self._activate = False

        self._recorder.start()
        while len(self._screen_bounding_box) < 4:
            time.sleep(1)
        self._recorder.finish()

        self.screen_origin = self._screen_bounding_box[0]
        self.screen_width = (
            self._screen_bounding_box[1][0] - self.screen_origin[0]
        ) // self.width_step * self.width_step + 1
        self.screen_height = (
            self._screen_bounding_box[1][1] - self.screen_origin[1]
        ) // self.height_step * self.height_step + 1

        self.url_popup_origin = self._screen_bounding_box[2]
        self.url_popup_width = self._screen_bounding_box[3][0] - self.url_popup_origin[0]
        self.url_popup_height = self._screen_bounding_box[3][1] - self.url_popup_origin[1]

    def on_click(
        self, pos: tuple[float, float], button: AnyButton, press: bool
    ) -> None:
        if press is True and self._activate is True:
            print((int(pos[0]), int(pos[1])))
            self._screen_bounding_box.append((int(pos[0]), int(pos[1])))

    def on_press(self, key: AnyKey) -> None:
        if key == Key.enter:
            self._activate = True

    def scan(self, depth: int = None, active_depth: int = None, npass=0):
        start = time.time()

        for i in range(
            0,
            (self.screen_width // self.width_step) * self.width_step - 1,
            self.width_step,
        ):
            for j in range(
                0,
                (self.screen_height // self.height_step) * self.height_step - 1,
                self.height_step,
            ):
                self.fill_area(
                    (i, j),
                    self.width_step + 1,
                    self.height_step + 1,
                    cut_off_d=depth,
                    active_d=active_depth,
                )
        
        print(f"Time to scan: {time.time() - start}s")

        self.bound_groups()
        self.show_with_overlay()
    
    def bound_groups(self):
        draw_on_image = ImageDraw.Draw(self.interactables_bb_overlay)
        for g, p in self.interactable_groups.items():
            top, bottom, left, right = p[0][1], p[0][1], p[0][0], p[0][0]
            for p_i in p:
                x_i, y_i = p_i
                if x_i > right:
                    right = x_i
                elif x_i < left:
                    left = x_i
                
                if y_i > bottom:
                    bottom = y_i
                elif y_i < top:
                    top = y_i
            self.bbox.append([top, bottom, left, right])
            draw_on_image.rectangle(
                (
                    left,
                    top,
                    right,
                    bottom
                ),
                outline = "black",
                width = 1
            )

    def show_with_overlay(self):
        overlay = Image.fromarray(
            (self.interactables_overlay.transpose() + 1) * 120
        ).convert("L")

        if self.mode == "manual":
            overlay.show()
            self.interactables_bb_overlay.show()

        scatter_overlay = ImageOps.colorize(overlay, black="white", white="red")
        bb_overlay = ImageOps.colorize(self.interactables_bb_overlay, black="black", white="white")
        scatter_composite = Image.blend(self.screen, scatter_overlay, 0.5)
        bb_composite = Image.blend(self.screen, bb_overlay, 0.5)
        common_composite = Image.blend(scatter_composite, bb_overlay, 0.5)

        scatter_overlay.save(f"temp/{self.prefix}_scatter_overlay.png")
        scatter_composite.save(f"temp/{self.prefix}_scatter_composite.png")
        bb_overlay.save(f"temp/{self.prefix}_bb_overlay.png")
        bb_composite.save(f"temp/{self.prefix}_bb_composite.png")
        common_composite.save(f"temp/{self.prefix}_common_composite.png")
        self.screen.save(f"temp/{self.prefix}_screen.png")

        with open(f"temp/bb_data.json", "+a") as f:
            f.write(f"""{{"source":"{self._url}", "bbox":{self.bbox}, "screen":"temp/{self.prefix}_screen.png" }},\n""")

    def fill_area(
        self,
        origin: tuple[int, int],
        w: int,
        h: int,
        cut_off_d: int | None = None,
        active_d: int | None = None,
        d: int = 0,
    ):
        active_check = True
        if active_d is not None and d >= active_d:
            active_check = False

        x, y = origin
        self.check_square(origin, w, h, active_check)

        if w <= 3 and h <= 3:
            return
        if cut_off_d is not None and d >= cut_off_d:
            return

        self.fill_area(
            (x, y),
            w // 2 + 1,
            h // 2 + 1,
            cut_off_d,
            active_d,
            d + 1,
        )
        self.fill_area(
            (x + w // 2, y),
            (w + 1) // 2,
            h // 2 + 1,
            cut_off_d,
            active_d,
            d + 1,
        )
        self.fill_area(
            (x + w // 2, y + h // 2),
            (w + 1) // 2,
            (h + 1) // 2,
            cut_off_d,
            active_d,
            d + 1,
        )
        self.fill_area(
            (x, y + h // 2),
            w // 2 + 1,
            (h + 1) // 2,
            cut_off_d,
            active_d,
            d + 1,
        )

    def check_square(self, origin: tuple[int, int], w: int, h: int, active=True):
        x, y = origin

        # -----------------------------------------------------------------------------
        # Getting actual corners
        # -----------------------------------------------------------------------------
        ul_c = (x, y)
        ur_c = (x + w - 1, y)
        ll_c = (x, y + h - 1)
        lr_c = (x + w - 1, y + h - 1)

        ul = self.check_pixel(ul_c, active)
        ur = self.check_pixel(ur_c, active)
        ll = self.check_pixel(ll_c, active)
        lr = self.check_pixel(lr_c, active)

        # -----------------------------------------------------------------------------
        # setting positions and default values to unknown
        # -----------------------------------------------------------------------------

        t, b, l, r, m = 0, 0, 0, 0, 0
        t_c = (x + w // 2, y)
        b_c = (x + w // 2, y + h - 1)
        l_c = (x, y + h // 2)
        r_c = (x + w - 1, y + h // 2)
        m_c = (x + w // 2, y + h // 2)

        # ------------------------------------------------------------------------------
        # Border checks
        # ------------------------------------------------------------------------------

        if ul == -1 and ur == -1:
            t = -1
        if ul == -1 and ll == -1:
            l = -1
        if lr == -1 and ur == -1:
            r = -1
        if lr == -1 and ll == -1:
            b = -1
        if w < self.width_space:
            if ul == 1 and ur == 1:
                t = 1
            if ll == 1 and lr == 1:
                b = 1
        if h < self.height_space:
            if ul == 1 and ll == 1:
                l = 1
            if ur == 1 and lr == 1:
                r = 1

        if t == 0:
            t = self.check_pixel(t_c, active)
        if b == 0:
            b = self.check_pixel(b_c, active)
        if r == 0:
            r = self.check_pixel(r_c, active)
        if l == 0:
            l = self.check_pixel(l_c, active)

        # ------------------------------------------------------------------------------
        # Middle checks
        # ------------------------------------------------------------------------------

        if (t == -1 and b == -1) or (r == -1 and l == -1):
            m = -1
        elif w < self.width_space and h < self.height_space:
            if sum([t, r, b, l, ul, ur, lr, ll]) >= 0:
                m = 1
            else:
                m = -1
        elif w < self.width_space:
            if l == 1 and r == 1:
                m = 1
        elif h < self.height_space:
            if t == 1 and b == 1:
                m = 1
        elif w < self.width_space * 2 and h < self.height_space * 2:
            if sum([t, r, b, l, ul, ur, lr, ll]) >= 5:
                m = 1

        if m == 0:
            m = self.check_pixel(m_c, active)

        # ------------------------------------------------------------------------------
        # Recording changes
        # ------------------------------------------------------------------------------

        self.set_pixel(t_c, t)
        self.set_pixel(b_c, b)
        self.set_pixel(l_c, l)
        self.set_pixel(r_c, r)
        self.set_pixel(m_c, m)

    def set_pixel(self, pos: tuple[int, int], value: int):
        x, y = pos
        self.interactables_overlay[x][y] = value

    def check_pixel(self, pos: tuple[int, int], active=True) -> int:
        x, y = pos
        current_state = self.interactables_overlay[x][y]
        if current_state != 0:
            return current_state
        elif active is False:
            return 0
        current_state = self.is_pixel_interactable(x, y)
        self.interactables_overlay[x][y] = current_state
        return current_state

    def is_pixel_interactable(self, x: int, y: int) -> int:
        self._controller.mouse_set_position(self.screen_origin[0], self.screen_origin[1])
        time.sleep(0.1)
        self._controller.mouse_set_position(
            self.screen_origin[0] + x, self.screen_origin[1] + y
        )
        new_screen = self.take_screen_shot(
            (0, 0), self.screen_width, self.screen_height
        )
        time.sleep(self.response_time)
        url_popup = self.take_screen_shot(
            (0, 0), self.url_popup_width, self.url_popup_height,
            self.url_popup_origin
        )

        if new_screen is None:
            return 0

        screen_diff = ImageChops.difference(self.screen, new_screen).convert("L")
        popup_exist = ImageChops.difference(self.url_popup, url_popup).convert("L").getextrema() != (0, 0)

        if screen_diff.getextrema() != (0, 0):
            self.record_group(screen_diff, (x, y))
            self.url_popup_active = popup_exist
            return 1
        
        if popup_exist is True:
            if self.url_popup_active is False:
                self.url_popup_active = True
                return 1
            time.sleep(1)

            url_popup = self.take_screen_shot(
                (0, 0), self.url_popup_width, self.url_popup_height,
                self.url_popup_origin
            )
            popup_exist = ImageChops.difference(self.url_popup, url_popup).convert("L").getextrema() != (0, 0)
            if popup_exist is True:
                self.url_popup_active = True
                return 1
            
        self.url_popup_active = False
        return -1

    def record_group(self, screen_diff, pos):
        # Very unlikely to collide.
        # For 1'000'000'000 interactables, collision probability is 10^-60
        # https://stackoverflow.com/questions/4014090/is-it-safe-to-ignore-the-possibility-of-sha-collisions-in-practice
        image_hash = hashlib.sha256(screen_diff.tobytes()).hexdigest()
        self.interactable_groups[image_hash].append(pos)

    def take_screen_shot(
        self, local_origin: tuple[int, int], w: int, h: int,
        global_origin: tuple[int, int] | None = None
    ) -> Image.Image | None:
        # TODO (Arnas): The splitting is required for macOS, because it returns RGBA
        #               Instead of RGB, so when diff is taken the alpha channel is
        #               always 0. Maybe we could convert the image to gray scale?
        #               I think in most cases we would still catch a diff.
        x_s, y_s = self.screen_origin if global_origin is None else global_origin
        x, y = local_origin
        try:
            screen_shot = Image.merge(
                "RGB",
                ImageGrab.grab(
                    bbox=(x + x_s, y + y_s, x + x_s + w, y + y_s + h)
                ).split()[:3],
            )
        except UnidentifiedImageError:
            print("Unable to take screenshot:")
            print(f"\t bbox - {x + x_s}, {y + y_s}, {x + x_s + w}, {y + y_s + h}")
            return None
        return screen_shot
