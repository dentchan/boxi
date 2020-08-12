from flask import Flask, request, Response
import time
import cv2

PATH_TO_TEST_IMAGES_DIR = './image'

app = Flask(__name__)

@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype="text/html")

# save the image as a picture
@app.route('/image', methods=['POST'])
def image():
    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))
    cv2.imshow("he",i)
    cv2.waitKey(0)
    return Response("%s saved" % f)

# ,ssl_context='adhoc'
if __name__ == '__main__':
    app.run()
