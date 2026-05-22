#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
################################################################################

import argparse
import configparser
import ctypes
import os
import sys
import time

import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ''))
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
import pyds

emb_array = []
names = []
id_name_dic = {}
frame_count = 0
start_time = time.time()
fps = 0.0


def load_face_embeddings(embeddings_path, names_path):
    global emb_array, names
    emb_array = []
    names = []

    with np.load(embeddings_path) as embeddings:
        for key in embeddings.files:
            emb = embeddings[key]
            if emb is not None:
                emb_array.append(emb)

    with open(names_path, 'r') as f:
        for line in f:
            value = line.strip().split()[0]
            if value:
                names.append(value)


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- FPS calculation (frame-level) ---
        global frame_count, start_time, fps
        frame_count += 1
        now = time.time()
        elapsed = now - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = now

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if str(obj_meta.object_id) in id_name_dic:
                obj_meta.text_params.display_text = id_name_dic[str(obj_meta.object_id)]

            l_user = obj_meta.obj_user_meta_list
            if l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    v = np.ctypeslib.as_array(ptr, shape=(1, 128))
                    v = v / np.linalg.norm(v)
                    for i, emb in enumerate(emb_array):
                        dist = np.linalg.norm(v - emb)
                        if dist < 1.0:
                            id_name_dic[str(obj_meta.object_id)] = names[i]
                            break
                except Exception:
                    pass

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

        # Add FPS display on the frame using display meta
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            text_params = display_meta.text_params
            text_params[0].display_text = f"FPS: {fps:.2f}"
            text_params[0].x_offset = 10
            text_params[0].y_offset = 12
            text_params[0].font_params.font_name = "Serif"
            text_params[0].font_params.font_size = 12
            # set white font color if available
            try:
                text_params[0].font_params.font_color.red = 1.0
                text_params[0].font_params.font_color.green = 1.0
                text_params[0].font_params.font_color.blue = 1.0
                text_params[0].font_params.font_color.alpha = 1.0
            except Exception:
                pass
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        except Exception:
            pass

    return Gst.PadProbeReturn.OK


def on_pad_added(src, new_pad, sink_element):
    sink_pad = sink_element.get_static_pad('sink')
    if sink_pad.is_linked():
        return

    new_pad_caps = new_pad.get_current_caps()
    if not new_pad_caps:
        return

    new_pad_struct = new_pad_caps.get_structure(0)
    new_pad_type = new_pad_struct.get_name()
    if not new_pad_type.startswith('video/'):
        return

    new_pad.link(sink_pad)


def set_property_safe(element, prop, value):
    try:
        element.set_property(prop, value)
    except Exception as e:
        sys.stderr.write(f"Skipping unsupported property '{prop}' on {element.get_name()}: {e}\n")


def main(input_path):
    Gst.init(None)

    print('Creating Pipeline')
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('Unable to create Pipeline\n')
        return -1

    source = Gst.ElementFactory.make('uridecodebin', 'uri-source')
    nvvidconvsrc = Gst.ElementFactory.make('nvvidconv', 'convertor_src')
    caps_vidconvsrc = Gst.ElementFactory.make('capsfilter', 'nvmm_caps')
    streammux = Gst.ElementFactory.make('nvstreammux', 'Stream-muxer')
    pgie = Gst.ElementFactory.make('nvinfer', 'primary-inference')
    tracker = Gst.ElementFactory.make('nvtracker', 'tracker')
    sgie1 = Gst.ElementFactory.make('nvinfer', 'secondary1-nvinference-engine')
    nvvidconv = Gst.ElementFactory.make('nvvidconv', 'convertor')
    nvosd = Gst.ElementFactory.make('nvdsosd', 'onscreendisplay')
    nvvidconv_postosd = Gst.ElementFactory.make('nvvideoconvert', 'convertor_postosd')
    caps = Gst.ElementFactory.make('capsfilter', 'caps')
    transform = Gst.ElementFactory.make('nvegltransform', 'nvegl-transform')
    sink = Gst.ElementFactory.make('nveglglessink', 'nvvideo-renderer')

    required = [source, nvvidconvsrc, caps_vidconvsrc, streammux, pgie, tracker, sgie1,
                nvvidconv, nvosd, nvvidconv_postosd, caps, transform, sink]
    for element in required:
        if not element:
            sys.stderr.write(f'Unable to create {element.get_name() if element else "element"}\n')
            return -1

    source.set_property('uri', Gst.filename_to_uri(input_path))
    source.connect('pad-added', on_pad_added, nvvidconvsrc)
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM)'))

    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('live-source', 0)
    streammux.set_property('batched-push-timeout', 4000000)

    pgie.set_property('config-file-path', './configs/deepstream_pgie_config.txt')
    sgie1.set_property('config-file-path', './configs/deepstream_sgie1_config.txt')

    config = configparser.ConfigParser()
    config.read('./configs/deepstream_tracker_config.txt')
    for key in config['tracker']:
        value = config.get('tracker', key)
        if key == 'tracker-width':
            set_property_safe(tracker, 'tracker-width', config.getint('tracker', key))
        elif key == 'tracker-height':
            set_property_safe(tracker, 'tracker-height', config.getint('tracker', key))
        elif key == 'gpu-id':
            set_property_safe(tracker, 'gpu-id', config.getint('tracker', key))
        elif key == 'll-lib-file':
            set_property_safe(tracker, 'll-lib-file', value)
        elif key == 'll-config-file':
            set_property_safe(tracker, 'll-config-file', value)
        elif key == 'enable-batch-process':
            set_property_safe(tracker, 'enable-batch-process', config.getint('tracker', key))
        elif key == 'enable-past-frame':
            set_property_safe(tracker, 'enable-past-frame', config.getint('tracker', key))
        elif key == 'user-meta-pool-size':
            set_property_safe(tracker, 'user-meta-pool-size', config.getint('tracker', key))
        elif key == 'display-tracking-id':
            set_property_safe(tracker, 'display-tracking-id', config.getint('tracker', key))

    if is_aarch64():
        try:
            sink.set_property('sync', False)
        except Exception:
            pass

    caps.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM), format=I420'))

    pipeline.add(source)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(transform)
    pipeline.add(sink)

    nvvidconvsrc.link(caps_vidconvsrc)
    sinkpad = streammux.request_pad_simple('sink_0')
    if not sinkpad:
        sys.stderr.write('Unable to get the sink pad of streammux\n')
        return -1

    srcpad = caps_vidconvsrc.get_static_pad('src')
    if not srcpad:
        sys.stderr.write('Unable to get source pad of caps_vidconvsrc\n')
        return -1
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(transform)
    transform.link(sink)

    osdsinkpad = nvosd.get_static_pad('sink')
    if not osdsinkpad:
        sys.stderr.write('Unable to get sink pad of nvosd\n')
        return -1
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream file input realtime display pipeline')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_input = os.path.join(script_dir, 'data/drone_video.mp4')

    parser.add_argument('-i', '--input', default=default_input,
                        help='Path to input MP4 video file')

    args = parser.parse_args()
    input_path = os.path.expanduser(args.input)
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    return input_path


if __name__ == '__main__':
    input_path = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    load_face_embeddings(os.path.join(script_dir, 'models/embeddings.npz'),
                         os.path.join(script_dir, 'models/names.txt'))
    sys.exit(main(input_path))
