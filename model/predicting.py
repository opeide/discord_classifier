import numpy as np
from PIL import Image, ImageFont, ImageDraw
import keras
import glob


def model_predict_image(model, img):
    img = img.resize((100, 100))
    imga = np.array(img, dtype='float32') / 255
    imga = imga.reshape(1, 100, 100, 3)
    prediction = model.predict(imga)
    p_woman = prediction[0][1]
    return p_woman




if __name__ == '__main__':
    model = keras.models.load_model('model/pajeet_v1.32')

    for file in glob.glob('test_data/*/*.jpg'):
        img = Image.open(file).resize((100,100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('resources/arial.ttf', 12)
        testwoman = np.array(img, dtype='float32')/255
        testwoman = testwoman.reshape(1,100,100,3)
        prediction = model.predict(testwoman)
        p_woman = prediction[0][1]
        if p_woman < 0.4:
            draw.text((5,5), 'woman: {}%'.format(int(100*p_woman)), (255,0,0), font=font)
            img.show()