from flask import Flask
from detect_headset_flask import start_headset_detect
app = Flask(__name__)
app.register_blueprint(start_headset_detect)
@app.route('/five')
def index():
    return "Hello"

if __name__ == "__main__":
    app.run(debug=True)