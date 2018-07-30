import discord
from model.predicting import model_predict_image
import keras
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import bisect

def load_img(url):
    response = requests.get(url, timeout=2)
    img = Image.open(BytesIO(response.content))
    if not np.array(img).any():
        raise ('Image is empty')
    if np.array(img).shape[2] != 3:
        raise('Image is not RBG')
    return img


if __name__ == '__main__':
    TOKEN = 'NDcyODc2OTE0MTY0NjI5NTA2.Dj-tgQ.gkQB-QPmpYeSJJrU7_znFM8NGyU'
    client = discord.Client()
    model = keras.models.load_model('../model/pajeet_v1.33')

    responses = {0:'ERROR: Pajeet.py has reached critical PENAS!',
                 20:'send penas and TESTALCLE now i must u have!? i send mine NOW ok',
                 25:'send penes what can i se baby u hot hunk big meat i tak u to village an we can suk on the cob togeter',
                 30: 'shoe penes nice',
                 35:'okey not bad it has pens?? uwu',
                 40: 'no this i not like wher bobs',
                 45:'what even r u can get some bob u have not',
                 50: 'need more bobs',
                 55:'need little bit mor bob beby',
                 60: 'can i touch u on the front bobs bevi u can know me have vagen',
                 65:'HELLO milk truk just arrive i\'m am pajeet. will u send bobs and vagene, u are very sexxii',
                 70:'NICE BOB! DO MILK! SHO VAGINE BABY! PLS SEND VEGANIA PIC FOR ME PLEAS. I WANT TO PUT MY 1 FEET ##### IN UR U WILL HAPPY!',
                 75:'ERROR: Pajeet.py has reached critical bob! pls response vaganie px to restart'}

    response_keys = sorted(list(responses.keys()))
    print(response_keys)

    @client.event
    async def on_message(message):
        # we do not want the bot to reply to itself
        if message.author == client.user:
            return

        if message.content.startswith('!pajeet'):
            print(message.content)
            url = message.content.split(' ')[1]
            img = load_img(url)
            p_woman = model_predict_image(model, img)
            print(p_woman)
            percent_woman = int(100.0*p_woman)
            response_index = bisect.bisect_left(response_keys, percent_woman)
            response_index = response_index - 1 if response_index != 0 else response_index
            response_msg = responses[response_keys[response_index]]
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