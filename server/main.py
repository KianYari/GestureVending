# main.py
from flask import Flask, jsonify, Response
import cv2
import time
import shared_state
import udp_server
import mqtt_handler
import config

app = Flask(__name__)

# =======================
# Flask Routes
# =======================
@app.route('/latest_result', methods=['GET'])
def get_latest_result():
    with shared_state.frame_lock:
        if shared_state.latest_result is None:
            return jsonify({'error': 'No data available'}), 404
        return jsonify(shared_state.latest_result), 200

@app.route('/video_stream')
def video_stream():
    def generate():
        while True:
            with shared_state.frame_lock:
                if shared_state.latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', shared_state.latest_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html><head><title>UDP Hand Tracking</title></head>
    <body style="background-color:#222;color:white;font-family:Arial;">
        <h1>ESP32 UDP Stream + MediaPipe + MQTT</h1>
        <img src="/video_stream" width="640" height="480">
        <p>Ensure your ESP32 is running the UDP code.</p>
    </body></html>
    '''

if __name__ == '__main__':
    print("="*50)
    print("ESP32 UDP + MediaPipe Server Running")
    print("MQTT Active on: " + config.MQTT_TOPIC_SELECTING)
    print("="*50)
    
    # Start Background Services
    mqtt_handler.start_mqtt()
    udp_server.start_server()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)