#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

Gst.init(None)

class RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, video_path):
        super(RTSPServer, self).__init__()
        self.factory = GstRtspServer.RTSPMediaFactory()
        # Pipeline: read file → demux → decode → re-encode → RTP H264
        self.factory.set_launch(
            f"( filesrc location={video_path} ! qtdemux ! h264parse ! rtph264pay name=pay0 pt=96 )"
        )
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/stream", self.factory)

if __name__ == "__main__":
    video_file = "data/drone_video.mp4"  # change to your file path
    server = RTSPServer(video_file)
    server.attach(None)
    print("RTSP server running at rtsp://<jetson-ip>:8554/stream")
    loop = GObject.MainLoop()
    loop.run()
