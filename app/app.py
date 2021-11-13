MODEL_SAVE_PATH = "../savefolder/all_14.pt"
import streamlit as st

import numpy as np
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import meta

from libs.dummy import outputs as dummy_outputs
from libs.utils import (
    set_session_state,
    get_session_state,
    local_css,
    remote_css,
    plot_result
)

class ImageProcessor:
    def __init__(
            self,
            model_name="tf_efficientnet_b4_ns"
        ):
        self.model_name = model_name
        self.mt_diseases = ['Effusion', 'Nodule', 'Fibrosis', 'Cardiomegaly', 'Mass', 'Pneumothorax', 'Edema', 'Consolidation', 'Pneumonia', 'Hernia', 'Infiltration', 'Pleural_Thickening', 'Atelectasis', 'Emphysema']
        self.consider_diseases = [1,     0,         0,         1,             1,       0,              0,       0,               0,           0,        1,              0,                    1,             0,]


        self.label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.id2label = {v: k for k,v in self.label2id.items()}
        IMG_SIZE    = 224

        self.debug = False
        self.dummy_outputs = dummy_outputs
        self.transform = self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize(size=(IMG_SIZE,IMG_SIZE)), # Resizing the image to be 224 by 224
                                # torchvision.transforms.RandomRotation(degrees=(-20,+20)), #Randomly Rotate Images by +/- 20 degrees, Image argumentation for each epoch
                                torchvision.transforms.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                                torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                            ])
        self.loader = torchvision.datasets.folder.default_loader


    def recording(self, duration_in_seconds=10):
        recording = sd.rec(
            frames=int((duration_in_seconds + 0.5) * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            blocking=True,
        )
        sd.wait()
        return recording

    def load(self):
        self.model = timm.create_model(self.model_name, pretrained=True)
        prev_state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))
        if prev_state_dict['classifier.5.bias'].shape[0] == 2:
            self.multitask = False
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=1792, out_features=625),
                nn.LeakyReLU(), nn.Dropout(p=0.3),
                nn.Linear(in_features=625, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=2),
            )
        else:
            self.multitask = True
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=1792, out_features=625),
                nn.LeakyReLU(), nn.Dropout(p=0.3),
                nn.Linear(in_features=625, out_features=256),
                nn.LeakyReLU(),
                nn.Linear(in_features=256, out_features=14),
                nn.Sigmoid()
            )
        self.model.load_state_dict(prev_state_dict)
        self.model.eval()
        print(self.model.classifier[0].weight[:5, :6])

    def _speech_file_to_array_fn(self):
        None

    def predict_cf(self, path):

        sample = self.loader(path)
        sample = torch.tensor(self.transform(sample)).unsqueeze(0)
        print(sample[0, 0, 50:55, 50:55])
        print(sample.shape)

        with torch.no_grad():
            logits = self.model(sample).squeeze(0)
            print(logits)
        if self.multitask:
            scores = [(d, l) for l, d, c in zip(logits.tolist(), self.mt_diseases, self.consider_diseases) if c==1]
        else:
            raise NotImplementedError
        outputs = [{"label": dis,
                    "score": float(sc)
                    } for dis, sc in scores
                ]
        print(outputs)

        return outputs

    def predict(self, path):
        if self.debug:
            print(self.debug)
            return self.dummy_outputs

        cf = self.predict_cf(path)
        # cf = self.predict_cf(path, ctc[0] if len(ctc) > 0 else ctc)

        return {"cf": cf}


@st.cache(allow_output_mutation=True)
def load_tts():
    tts = ImageProcessor()
    tts.load()
    return tts


def main():
    st.set_page_config(
        page_title="Prompt Engineer",
        page_icon="<3",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    remote_css("https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css")
    set_session_state("_is_recording", False)
    local_css("assets/style.css")
    # st.write(f"DEVICES: {sd.query_devices()}")

    tts = load_tts()

    col1, col2 = st.columns([5, 7])
    with col2:
        st.markdown('<div class="mt"></div>', unsafe_allow_html=True)
        st.markdown('<br><br><br>', unsafe_allow_html=True)
        image_shower = st.empty()
        speech_text = st.empty()

    with col1:
        st.markdown(meta.INFO, unsafe_allow_html=True)
        image_file = st.file_uploader("Upload an Audio File",type=['png'])
        # duration = st.slider('Choose your recording duration (seconds)', 5, 20, 5)
        # recorder_btn = st.button("Recording")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    info = st.empty()

    # if recorder_btn:
        # if not get_session_state("_is_recording"):
        #     set_session_state("_is_recording", True)

        #     info.info(f"{duration} of Recording in seconds ...")
        #     np_audio = tts.recording(duration_in_seconds=duration)
        #     if len(np_audio) > 0:
        #         filename = tempfile.mktemp(prefix='tmp_sf_', suffix='.wav', dir='')
        #         with sf.SoundFile(
        #                 filename,
        #                 mode='x',
        #                 samplerate=tts.samplerate,
        #                 channels=tts.channels,
        #                 subtype=tts.subtype
        #         ) as tmp_audio:
        #             tmp_audio.write(np_audio)


        #         audio_player.audio(filename)
        #         speech_text.info(f"Converting speech to text ...")
        #         result = tts.predict(filename)
        #         speech_text.markdown(
        #             f'<p class="ctc-box ltr"><strong>Text: </strong>{result["ctc"]}</p>',
        #             unsafe_allow_html=True
        #         )

        #         info.info(f"Recognizing disease ...")
        #         plot_result(result["cf"])

        #         if os.path.exists(filename):
        #             os.remove(filename)

        #         info.empty()
        #         set_session_state("_is_recording", False)

    if image_file is not None:
        # file_details = {"FileName":audio_file.name,"FileType":audio_file.type}
        # st.write(file_details)
        with open(image_file.name, "wb") as f:
            f.write(image_file.getbuffer())
        st.success("Saved File")

        # print(type(audio_file))
        image_shower.image(image_file.name)
        speech_text.info(f"Recognizing Diseases ...")
        result = tts.predict(image_file.name)

        # info.info(f"Recognizing emotion ...")
        plot_result(result["cf"])

        if os.path.exists(image_file.name):
            os.remove(image_file.name)

            info.empty()
        speech_text.empty()


if __name__ == '__main__':
    main()
