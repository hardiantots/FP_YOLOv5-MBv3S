import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration, VideoProcessorBase
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import av
from utils.turn import get_ice_servers

cfg_model_path = "models/mobilenetv3s.pt"

def imageInput(device, src):

    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=cfg_model_path, force_reload=True)
            model.cuda() if device == 'cuda' else model.cpu()
            model.conf = 0.7
            model.iou = 0.5
            model.max_det = 1
            pred = model(imgpath, size=256)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.':
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1)
        image_file = imgpath[imgsel - 1]
        submit = st.button("Predict!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                # call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True)
                model.conf = 0.7
                model.iou = 0.5
                model.max_det = 1
                pred = model(image_file, size=256)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                # --Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=False)

class VideoProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(img)
        st.model.conf = 0.4
        st.model.iou = 0.4
        results = st.model(im_pil, size=256)
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


def main():
    # -- Sidebar
    with st.sidebar:
        logoteam_img = Image.open("documentationproject/logo2_transparent.png")
        st.image(logoteam_img, width=128)
        st.title("Cuan Hunter Final Project")

        st.title('⚙️Options')
        datasrc = st.radio("Select input source.", ['From test set.', 'Upload your own data.'])

        option = st.radio("Select input type.", ['Image', 'Webcam Stream'])
        if torch.cuda.is_available():
            deviceoption = st.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=1)
        else:
            deviceoption = st.radio("Select compute Device.", ['cpu', 'cuda'], disabled=True, index=0)        


    st.header('🤚 Indonesian Sign Language Detector (SIBI & BISINDO)')
    st.subheader('👈🏽 Select options left-handed menu bar.')
    if option == "Image":
        imageInput(deviceoption, datasrc)
    elif option == "Webcam Stream":
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=VideoProcessor,
        )

if __name__ == '__main__':

    main()
