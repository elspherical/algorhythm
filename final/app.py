from flask import Flask, request, render_template, jsonify, send_file
import threading
import subprocess
import os
import uuid
import ffmpeg    # <-- add this import

app = Flask(__name__)
PROCESSING = False
CURRENT_FILES = {}  # store current input/output per upload

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_video():
    global PROCESSING, CURRENT_FILES
    if PROCESSING:
        return jsonify({'status': 'busy'}), 429

    file = request.files['video']
    if not file:
        return jsonify({'error': 'No video uploaded'}), 400

    unique_id = str(uuid.uuid4())

    # Browser recordings are usually .webm — save as raw file first
    raw_input_path = f"raw_{unique_id}.webm"
    converted_input_path = f"recorded_{unique_id}.mp4"
    output_path = f"final_{unique_id}.mp4"

    # Save uploaded file
    file.save(raw_input_path)

    # ---- 🔧 CONVERT webm → mp4 (critical fix) ----
    try:
        (
            ffmpeg
            .input(raw_input_path)
            .output(
                converted_input_path,
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                r=30,            # stabilizes weird browser frame rates
                vsync='cfr',
                movflags='faststart'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        return jsonify({
            'error': 'Video conversion failed',
            'details': e.stderr.decode()
        }), 500
    # -------------------------------------------------

    CURRENT_FILES = {'input': converted_input_path, 'output': output_path}
    PROCESSING = True

    def run_processing():
        global PROCESSING
        subprocess.run(['python3', 'process_video.py',
                        converted_input_path, output_path])
        PROCESSING = False

    threading.Thread(target=run_processing).start()

    return jsonify({
        'status': 'started',
        'input': converted_input_path,
        'output': output_path
    })

# Status endpoint
@app.route('/status')
def status():
    return jsonify({'processing': PROCESSING})

# Get processed result
@app.route('/result')
def result():
    if not CURRENT_FILES:
        return jsonify({'ready': False})

    output_path = CURRENT_FILES['output']
    input_path = CURRENT_FILES['input']

    if not os.path.exists(output_path):
        return jsonify({'ready': False})

    return jsonify({
        'ready': True,
        'original': input_path,
        'processed': output_path
    })

# Serve videos
@app.route('/videos/<filename>')
def videos(filename):
    # Look in current directory only (matching your original design)
    if not os.path.exists(filename):
        return "File not found", 404
    return send_file(filename, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
