from flask import Flask, render_template
from segment import capture_image
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('clickable.html');

@app.route('/image_capture/', methods=["POST","GET"])
def open_camera():
    capture_image()

if __name__ == "__main__":
    app.run(debug=True)

