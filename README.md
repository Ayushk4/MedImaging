# Multitasked Adaptively Fine-tuned CNNs For X-Ray Disease Detection

About 12 million misdiagnosis occur every year in USA alone leading to 40,000-80,000 aviodable deaths. Moreover, the high costs of more comprehensive medical diagnostic tests like Chest CT Imaging makes healthcare less afforadable to all. With these motivation, our problem statement, aims to leverage the more feasible X-Ray tests to perform the diagnostics. However, diagnostics from X-Ray tests can be more difficult than their CT Imaging counterparts, making it difficult for humans to diagnose with good accuracy. The recent advancements in Deep Learning have shown tremendous successes on various Visual, Audio and Text Processing task even outperforming experts.

Our best model consists of an adapatively trained Efficient-CNN architecture achieving the top performance on the leaderboard for this task with 83.95 F1 Score on the test set and 91.82 F1-Score on the dev-set across averaged across all 14 diseases. In our approach we consider the EfficientNet architecture as the backbone initialized with already pretrained weights on the large-scale dataset ImageNet an Medical images. We reinitialize the classification layer to a 2-layered MLP and first fine-tune it on the provided dataset consisting of Cardiomegaly vs No Finding followed by changing the classification head for multi-task and training across the extended dataset with all the 14 diseases. We observe that though the performance on Cardiomegaly slightly decrease after second fine-tuning from 83.95 Test-Set F1 to 81.96 F1, the first fine-tuning step improves the average performance considerably across all the 14 diseases from 87.61 F1-Score to 91.82 F1-Score on the dev set.

We performed numerous experiments to ablate the proposed model. Our experiments with other models like VGG, ResNet, Vision-Transformers show that EfficientNet outperform these, while being computationaly feasible than more heavier networks DarkNet, ViT-Large. We also find that keeping the initial layers of the pretrained model frozen, leads to a considerable performance degradation. We believe this is because the X-Ray images represents a considerable domain shift from the Natural Images in the ImageNet dataset. This is also reflected in poorer performance of inductive-biases free (Vision) Transformers, which are believed to be a lot data-hungry.

Finally we study the feasibility and deployability of our proposed model opening room for a lot of future work of our prototype. First we create a simple user-friendly app for our best-performing model which runs in near real-time without GPU on a MacBook with an i3 processor requiring about 200 MB of RAM. With the recent advancements in Deep Learning techniques of Network Compression, Pruning, Quantization and Knowledge Distillation, these models can be further compressed improved in performance.

# Dependencies and Using the Code

- Python
- Pytorch
- TorchVision
- Streamlit
- Scikit-learn
- Scipy
- Numpy
- Pillow
- Tqdm

0. Download the dependencies
1. Clone the repository
2. Place the datasets in `dataset_full` and `dataset_cardiomegaly` folders respectively.
3. For each experiment directory, add a symbolic link to both the datasets from within the folder.
4. If needed download the pretrained weights and keep them in `<folder_name>/pretrained_models` folder.
5. To run our app download the pretrained models from [here](https://github.com/Ayushk4/MedImaging/releases/download/tag/all_14.pt)  (all 14). We also provide the weights for our only cardiomegaly model [here](https://github.com/Ayushk4/MedImaging/releases/download/tag/single_disease.pt). The models should be kept in the `savefolder` folder. Their performances are close to 91 F1 Dev and 82 F1 Test on all-14 and cardiomegaly respectively.
6. Execute `streamlit run app.py` to run our app from within the `app` folder.
7. You may re-run these same experiments from within the Notebook, `multi-task`, `single-task` or `transformer` folder. If you want to experiment with models other than the used pre-trained `EfficientNet`, then change the ModelName accordingly in their respective trainer files.

