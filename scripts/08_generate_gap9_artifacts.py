(.gap9-venv) root@a3c5d51fb04f:/app/TinyDisasterVQA# cd /app/gap9tutorial/gap9_nn_getting_started/nn_end_to_end

python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("yamnet.ipynb").read_text())

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))
    keywords = [
        "NNGraph",
        "load_graph",
PY      print(src)* 100)")in keywords):
====================================================================================================
CELL 1
====================================================================================================
import librosa

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import yamnet.params as yamnet_params
import yamnet.yamnet as yamnet_model
from yamnet.yamnet import PreProcessingKerasWrapper

from nntool.api import NNGraph
from nntool.api.types import MFCCPreprocessingNode
from nntool.api.utils import quantization_options, model_settings
from nntool.api.quantization import QType

import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
====================================================================================================
CELL 12
====================================================================================================
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=YAMNET_TFLITE)
interpreter.allocate_tensors()

# Run inference on the TFLite model with the given input
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scores = []
for feat in features:
    # Set the input data
    interpreter.set_tensor(input_details[0]['index'], feat.astype(np.float32).reshape(input_details[0]["shape"]))

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
    scores.append(output_data)

scores = np.array(scores)
====================================================================================================
CELL 15
====================================================================================================
preproc_graph = NNGraph(name="logmel")
inp = preproc_graph.add_input((N_SAMPLES_PER_INF, ))
logmel = MFCCPreprocessingNode(
    "logmel",
    window="hanning",
    n_fft=512,
    power=1,
    frame_size=round(YAMNET_PARAMS.sample_rate * YAMNET_PARAMS.stft_window_seconds),
    frame_step=round(YAMNET_PARAMS.sample_rate * YAMNET_PARAMS.stft_hop_seconds),
    sample_rate=YAMNET_PARAMS.sample_rate,
    n_frames=96,
    # NO MFCC output before that
    n_dct=0,
    # Mel spect settings
    fbank_type="tensorflow",
    # if tensorflow is not available you can specify here a np.array with the precomputed values
    filterbanks_values=None,
    fmin=YAMNET_PARAMS.mel_min_hz,
    fmax=YAMNET_PARAMS.mel_max_hz,
    n_fbanks=YAMNET_PARAMS.mel_bands,
    # Log settings
    log_type="natural",
    log_offset=YAMNET_PARAMS.log_offset,
)(inp)
output = preproc_graph.add_output()(logmel)
preproc_graph.add_dimensions()

preproc_graph.quantize(
    graph_options=quantization_options(scheme="FLOAT", float_type="float32"),
    node_options={
        # input comes from wav file or microphone in fixed point format Q15
        "input_1": quantization_options(qtype_ind=QType(q=15, dtype=np.int16)),
        # Mfcc in floating 16
        # "logmel": quantization_options(scheme="FLOAT", float_type="bfloat16"),
    }
)

====================================================================================================
CELL 17
====================================================================================================
outs = preproc_graph.execute(waveform[:N_SAMPLES_PER_INF], dequantize=True)
nntool_spectrogram = outs[-1][0]
====================================================================================================
CELL 22
====================================================================================================
G = NNGraph.load_graph(YAMNET_TFLITE)
G.adjust_order()
G.fusions()
====================================================================================================
CELL 29
====================================================================================================
from glob import glob

quant_dataset = glob("calibration_dataset/*")
stats = G.collect_statistics(
    representative_dataset(
        quant_dataset,
        frame_step=round(YAMNET_PARAMS.patch_hop_seconds * 16000),
        frame_size=G[0].out_dims[0].size()
    )
)

====================================================================================================
CELL 31
====================================================================================================
G.quantize(
    statistics=stats,
    graph_options=quantization_options(use_ne16=True),
    node_options={
        # input comes from wav file or microphone in fixed point format Q15
        "input_1": quantization_options(qtype_ind=QType(q=15, dtype=np.int16)),
        # Mfcc in floating 32
        "logmel": quantization_options(scheme="FLOAT", float_type="float32"),
    }
)
====================================================================================================
CELL 33
====================================================================================================
framed_waveform = get_framed_waveform(
    waveform,
    frame_step=round(YAMNET_PARAMS.patch_hop_seconds * 16000),
    frame_size=G[0].out_dims[0].size()
)

def execute_graph(graph, frames, quant_exec=True):
    run_every = round(YAMNET_PARAMS.patch_hop_seconds * 16000)
    frame_size = graph[0].out_dims[0].size()
    scores = []
    spectrogram = np.array([])
    for i, frame in enumerate(tqdm(frames, desc=f"Running frame {'Quantized' if quant_exec else 'Float'}")):
        outs = graph.execute([frame], dequantize=quant_exec)
        if i < len(frames) - 1:
            spectrogram = np.concatenate([spectrogram, outs[graph["logmel"].step_idx][0][:(run_every-frame_size) // run_every].flatten()])
        else:
            spectrogram = np.concatenate([spectrogram, outs[graph["logmel"].step_idx][0].flatten()])

        scores.append(outs[graph["output_1"].step_idx][0])
    spectrogram = spectrogram.reshape(-1, YAMNET_PARAMS.mel_bands)
    scores = np.array(scores)
    return scores, spectrogram
====================================================================================================
CELL 37
====================================================================================================
qqout_nntool = G.execute([framed_waveform[0]], quantize=True)
====================================================================================================
CELL 38
====================================================================================================
res = G.execute_on_target(
    directory="/tmp/my_test_run",
    finput_tensors=framed_waveform[0],
    output_tensors=4,
    print_output=True,
    settings=model_settings(
        tensor_directory="tensors",
        model_directory="at_model",

        l3_ram_device="AT_MEM_L3_DEFAULTRAM",
        l3_flash_device="AT_MEM_L3_DEFAULTFLASH",

        privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
        privileged_l3_flash_size=1800000,

        graph_size_opt=2,
        graph_dump_tensor_to_file=True,
        graph_const_exec_from_flash=True,
        graph_l1_promotion=2,
    )
)
====================================================================================================
CELL 39
====================================================================================================
qsnrs = G.dict_qsnrs(qqout_nntool, res.output_tensors)
print(f"{'Layer':50}: {'QSNR':>10}")
print(f"{'-'*50}--{'-'*10}")
for lay, q in qsnrs.items():
    if q is not None:
        print(f"{lay:50}: {q:10}")
====================================================================================================
CELL 41
====================================================================================================
res = G.execute_on_target(
    directory="/tmp/my_test_run",
    # finput_tensors=framed_waveform[0],
    # output_tensors=4,
    print_output=True,
    settings=model_settings(
        tensor_directory="tensors",
        model_directory="at_model",

        l3_ram_device="AT_MEM_L3_DEFAULTRAM",
        l3_flash_device="AT_MEM_L3_DEFAULTFLASH",

        privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
        privileged_l3_flash_size=1800000,

        graph_size_opt=2,
        graph_dump_tensor_to_file=True,
        graph_const_exec_from_flash=True,
        graph_l1_promotion=2,
    )
)
(.gap9-venv) root@a3c5d51fb04f:/app/gap9tutorial/gap9_nn_getting_started/nn_end_to_end# 