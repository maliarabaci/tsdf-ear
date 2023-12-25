# Egocentric Activity Recognition Using Two-Stage Decision Fusion

Implementation of Two-Stage Decision Fusion for multi-modal egocentric activity recognition in Python. This repository also chooses to adopt the specific transformer architecture from PaLM, for both the unimodal and multimodal transformers as well as the cross attention blocks (parallel SwiGLU feedforwards)

Egocentric video capture, known as First Person Vision (FPV), exhibits distinct characteristics such as significant ego-motions and frequent scene changes, rendering conventional vision-based methods ineffective. This repository introduces a novel audio-visual decision fusion framework for egocentric activity recognition (EAR) that addresses these challenges. The proposed framework employs a two-stage decision fusion pipeline with explicit weight learning, integrating both audio and visual cues to enhance overall recognition performance. A new publicly available dataset, the [Egocentric Outdoor Activity Dataset (EOAD)](https://zenodo.org/records/7742660) , comprising 1392 video clips featuring 30 diverse outdoor activities, is also introduced to facilitate comparative evaluations of EAR algorithms and spur further research in the field. 

Experimental results demonstrate that the integration of audio and visual information significantly improves activity recognition performance, outperforming single modality approaches and equally weighted decisions from multiple modalities. 

![alt text](https://github.com/maliarabaci/tsdf-ear/blob/main/tsdf.png?raw=true)

## Install
To overcome the configuration issues, we shared Dockerfile to easily run the codebase in a container. For that purpose, you first need to install Docker environment in your OS. Additionally, pre-trained model and feature files that were stored in pickle format can be downloaded [here](https://drive.google.com/file/d/1lti-i9xiFkVWKrop6mHok1ckwLu6jF7z/view?usp=drive_link).

After installing Docker and copying the required model and feature files into data folder, you can build a Docker image with the following command.

```
docker build . -t tsdf-ear:latest
```

## Usage

You can run the previously builded Docker image with the following command.
```
docker run -it tsdf-ear:latest bash
```

This will attach you into a container in which you can run test scripts. Run the following Python script in the root folder to test two-stage decision fusion algorithm for EOAD test features. The result files will be saved into *results* folder.
```
python deep_feat_fusion/test_tsdf.py --config=data/config/config.txt
```

## Citations

Submitted to [Neural Computing and Applications](https://link.springer.com/journal/521). Bibtex information will be given later. 
<!-- This content will not appear in the rendered Markdown 
```bibtex
@inproceedings{Yu2022CoCaCC,
  title   = {CoCa: Contrastive Captioners are Image-Text Foundation Models},
  author  = {Jiahui Yu and Zirui Wang and Vijay Vasudevan and Legg Yeung and Mojtaba Seyedhosseini and Yonghui Wu},
  year    = {2022}
}
```
-->
