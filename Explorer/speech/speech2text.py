import speech_recognition as sr
from typing import Union, Literal, Any
from collections.abc import Callable
import time
import re


class Speech2Text:

    def __init__(self, disable_warnings: bool = False) -> None:
        super(Speech2Text).__init__()
        
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source) # calibrate

        self.user_current_phrase: str = None # updated with current phrase
        self.click_num: int = None
        self.disable_warnings = disable_warnings

        self.command_phrases = {
                                "show": 0,
                                "click": 1,
                                "stop": 2,
                                }
        self.commands = list(self.command_phrases.keys())

        self._executors: dict[Callable[[Any], None]] = {
            "show": None,
            "click": None,
            "stop": None,
        }

    def attach_exec(
        self, exec_func, target: Literal["show", "click", "stop"]
    ):
        if target not in self._executors.keys():
            raise ValueError(f"target {target} does not exist in Speech2Text updaters list")
        self._executors[target] = exec_func

    def execute(self):
        if self.user_current_phrase == None:
            #print('no user current phrase')
            return -1
        if self._executors[self.user_current_phrase] == None:
            #print('no executor')
            return -2
        exec = self._executors[self.user_current_phrase]
        exec(click=self.click_num) # call func to click on this num
        return 0

    def callback(self, recognizer, audio) -> Union[int | None]:
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
        
        user_command = re.sub(r'[^\w]', ' ', user_command.lower())
        detected_command = set(self.commands) & set([i for i in user_command.split(' ') if i!=""])
        if len(detected_command) == 1:
            """ One of the valid commands detected """
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
            self.user_current_phrase = detected_command
            # execute attached func
            status = self.execute()
            if (status == -1) and (self.disable_warnings == False):
                raise RuntimeWarning('No phrase registered. user_current_phrase = None.')
            elif (status == -2) and (self.disable_warnings == False):
                raise RuntimeWarning('No executor attached to this action. No execution.')
            return status
        else:
            print('No valid command detected.')
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