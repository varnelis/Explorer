from enum import StrEnum
import speech_recognition as sr
from typing import Any
from collections.abc import Callable
import time
import re


class CommandPhrases(StrEnum):
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
        self.user_current_phrase: CommandPhrases = None # updated with current phrase
        self.click_num: int = None
        self.verbose = verbose
        self.commands = [
            CommandPhrases.SHOW.value,
            CommandPhrases.CLICK.value,
            CommandPhrases.STOP.value,
        ]
        self._executors: dict[CommandPhrases, Callable[[Any]]] = {}

    def attach_exec(
        self, exec_func: Callable, target: CommandPhrases
    ):
        if not isinstance(target, CommandPhrases):
            raise ValueError(f"target {target} does not exist in Speech2Text updaters list")
        self._executors[target] = exec_func

    def execute(self):
        try:
            exec = self._executors[self.user_current_phrase]
        except KeyError:
            if self.verbose == True:
                raise RuntimeWarning('No executor attached to this action. No execution.')
            return
        
        if self.user_current_phrase == CommandPhrases.CLICK:
            exec(self.click_num)
        elif (
            self.user_current_phrase == CommandPhrases.SHOW or 
            self.user_current_phrase == CommandPhrases.STOP
        ):
            exec()
        else:
            raise RuntimeError('No phrase registered. user_current_phrase = None.')
        return

    def callback(self, recognizer, audio) -> int | None:
        """ Interrupt Service Routine on user-spoken phrases detected by background listener.
        Detects audio and converts to text with Google Speech Recognition. """
        try:
            # using the default API key; look at `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            user_command = recognizer.recognize_whisper(audio)       #recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None
        
        print('detected: ', user_command)
        user_command = re.sub(r'[^\w]', ' ', user_command.lower())
        detected_command = set(self.commands) & set([i for i in user_command.split(' ') if i!=""])
        
        if len(detected_command) != 1:
            print('No valid command detected.')
            return None

        # One of the valid commands detected
        detected_command = list(detected_command)[0]
        # set params
        if detected_command == "click":
            try:
                self.click_num = self._text2int([i for i in user_command.split(' ') if i!=""][1])
            except IndexError as e:
                print('Invalid `click` command. ', str(e))
                return None
        else:
            self.click_num = None
        self.user_current_phrase = CommandPhrases(detected_command)
        # execute attached func
        self.execute()
        return None

    def _text2int(self, textnum: str, numwords: dict = {}):
        """ Convert written numbers (e.g. either '1' or 'one') to int format (e.g. 1) """
        if textnum.isdigit():
            return int(textnum)

        if not numwords:
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
                raise Exception("Illegal word: " + word)
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        return result + current

    def listen(self) -> None:
        """ Start listening in the background. """
        self.stop_listening = self.rec.listen_in_background(self.mic, self.callback)
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