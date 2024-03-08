import speech_recognition as sr
#from sentence_transformers import SentenceTransformer, util
from typing import Union
import time


class Speech2Text:

    def __init__(self) -> None:
        super(Speech2Text).__init__()
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source) # calibrate

        self.command_phrases = {
                                "show": 0,
                                "click": 1,
                                "stop": 2,
                                }
        self.commands = list(self.command_phrases.keys())
        
        self.user_current_phrase: str = 2 # updated with current phrase
        self.click_num: int = None

    def callback(self, recognizer, audio) -> Union[str | None]:
        """ Interrupt Service Routine on user-spoken phrases detected by background listener.
        Detects audio and converts to text with Google Speech Recognition. """
        try:
            # using the default API key; look at `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            user_command = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None
        
        if self.commands[1] in user_command: # click action
            try:
                self.user_current_phrase = self.commands[1]
                self.click_num = self._text2int(user_command.split(' ')[1])
            except IndexError as e:
                print('Invalid `click` command. ', str(e))
        elif (self.commands[0] in user_command) or (self.commands[2] in user_command):
            self.user_current_phrase = self.command_phrases[user_command]
            self.click_num = None

    def _text2int(self, textnum: str, numwords: dict = {}):
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