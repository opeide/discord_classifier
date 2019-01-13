import discord
from model.predicting import model_predict_image
import keras
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import bisect
import json

def load_img(url):
    response = requests.get(url, stream=True, timeout=2)
    for bytes in response.iter_content(chunk_size=100000):
        img = Image.open(BytesIO(bytes))
        error_msg = ''
        print(np.array(img).shape)
        if not np.array(img).any() or not np.array(img).shape:
            error_msg = ('no pix shoe here? plz send india jpg')
        elif np.array(img).shape[2] != 3:
            error_msg = ('india intternet onlt work jpg?')
        return img, error_msg


if __name__ == '__main__':
    with open('token.txt', 'r') as f:
        TOKEN = f.read()
        print(TOKEN)
    with open('responses.json', 'r') as f:
        responses = json.load(f)
        print(responses)
        response_keys = sorted([int(key) for key in responses.keys()])
        print(response_keys)

    model = keras.models.load_model('../model/pajeet_v1.33')
    client = discord.Client()

    @client.event
    async def on_message(message):
        # we do not want the bot to reply to itself
        if message.author == client.user:
            return

        if message.content.startswith('!pajeet'):
            print(message.content)
            url = message.content.split(' ')[1]
            img, error_msg = load_img(url)

            if error_msg:
                await client.send_message(message.channel, error_msg)
            else:
                p_woman = model_predict_image(model, img)
                print(p_woman)
                percent_woman = int(100.0*p_woman)
                response_index = bisect.bisect_left(response_keys, percent_woman)
                response_index = response_index - 1 if response_index != 0 else response_index
                response_msg = responses[str(response_keys[response_index])]
                print(response_msg)
                (gender, percent) = ('woman', 100*p_woman) if p_woman>0.5 else ('man', 100*(1-p_woman))
                await client.send_message(message.channel, '({}% {}) '.format(int(percent),gender)+response_msg)


    @client.event
    async def on_ready():
        print('Logged in as')
        print(client.user.name)
        print(client.user.id)
        print('------')


    client.run(TOKEN)