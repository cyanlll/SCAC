# Training-free Video Corpus Moment Retrieval via Synergistic Collaboration and Adaptive Calibration

## 1.Requiments
Please install the necessary dependencies listed in requirements.txt.

## 2.Data Preparation
We use ActivityNet Captions as an example, while the data preparation procedures for the other datasets are similar.

First, download the [ActivityNet Captions](http://activity-net.org/index.html) dataset from its official website.

For the LLM, the required model can be downloaded and deployed through [Ollama](https://ollama.com/).

For the VLM, please refer to the model preparation and configuration provided in [RefCap](https://github.com/BUAAPY/RefCap).

## 3.Inference
python main.py --config configs/activitynet.yaml pipeline
