import ffmpeg
from vidgear.gears import CamGear
from vidgear.gears import StreamGear
stream_params = {"-input_framerate": 9, "-livestream": True,
                "-streams": [
                {"-resolution": "640x480", "-framerate": 20.0},  # Stream3: 640x480 at 60fps framerate
                ],
}
streamer = StreamGear(output="../try.m3u8", format = "hls", **stream_params)

def stream(frame):
    streamer.stream(frame)
def terminate_stream():
    streamer.terminate
def write_frame(process, result):
    process.stdin.write(
        result
        #.astype('np.uint8')
        .tobytes()
    )

def close_process(process):
    process.stdin.close()
    process.wait()