from collections import defaultdict
from enum import StrEnum
import speech_recognition as sr
from typing import Any
from collections.abc import Callable
import time
import re


class CommandPhrase(StrEnum):
    """ Definitions of actions to listen for """
    SHOW = "show"
    CLICK = "click"
    STOP = "stop"

class Speech2Text:

    def __init__(self, verbose: bool = False) -> None:        
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source) # calibrate
        self.user_current_phrase: CommandPhrase = None # updated with current phrase
        self.click_num: int = None
        self.verbose = verbose
        self.commands = {p.value for p in list(CommandPhrase)}
        self._executors: dict[CommandPhrase, Callable[[Any]]] = defaultdict(None)

    def attach_exec(
        self, exec_func: Callable, target: CommandPhrase
    ):
        if not isinstance(target, CommandPhrase):
            raise ValueError(f"target {target} does not exist in Speech2Text updaters list")
        self._executors[target] = exec_func

    def execute(self):
        exec = self._executors[self.user_current_phrase]
        if exec is None:
            print(f"Unexpected user phrase: {self.user_current_phrase}")
            return
        
        if self.user_current_phrase == CommandPhrase.CLICK:
            exec(self.click_num)
            self.click_num = None
        elif (
            self.user_current_phrase == CommandPhrase.SHOW or 
            self.user_current_phrase == CommandPhrase.STOP
        ):
            exec()
        else:
            print(f'No phrase registered. user_current_phrase: {self.user_current_phrase}')

    def callback(self, recognizer, audio) -> None:
        print("analysing detected audio...")
        try:
            raw_user_phrase = recognizer.recognize_whisper(audio)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return
        except sr.RequestError as e:
            print(f"Unexpected request error: {e}")
            return
        
        user_phrase = re.sub(r'[^\w]', ' ', raw_user_phrase.lower())
        phrase_tokens = user_phrase.split()

        detected_command = self.commands & set(phrase_tokens)
        print(f"command_tokens: {phrase_tokens}")
        if len(detected_command) != 1:
            print('No valid command detected.')
            return 
        self.user_current_phrase = CommandPhrase(detected_command.pop())

        if self.user_current_phrase == CommandPhrase.CLICK:
            if len(phrase_tokens) < 2:
                print(f"Number was not detected...")
                return
            self.click_num = self._text2int(phrase_tokens[1])
            if self.click_num is None:
                print(f"Number was not detected...")
                return
        self.execute()

    def _text2int(self, textnum: str, numwords: dict = {}):
        """ Convert written numbers (e.g. either '1' or 'one') to int format (e.g. 1) """
        if textnum.isdigit():
            return int(textnum)

        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]
        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
        current = result = 0

        for word in textnum.split():
            if word not in numwords:
                return None
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        return result + current

    def listen(self) -> None:
        """ Start listening in the background. """
        self.stop_listening = self.rec.listen_in_background(self.mic, self.callback, 5)
        print('speak')

    def stop_listen(self) -> None:
        """ background listener stops listening """
        self.stop_listening(wait_for_stop=False)

    def disp(self):
        """ Debug """
        print('Phrase: ', self.user_current_phrase)
        print('Num: ', self.click_num, '\n')


if __name__ == "__main__":
    s2t = Speech2Text()
    s2t.listen()
    for _ in range(10):
        time.sleep(5)
        s2t.disp()
    s2t.stop_listen()