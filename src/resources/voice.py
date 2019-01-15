
from gtts import gTTS
import os

tts = gTTS(text="It is too dark.", lang="en")
tts.save("dark.mp3")
# os.system("mpg321 morning.mp3")
