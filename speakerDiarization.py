
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch ,torchaudio
from typing import List, Dict
from datetime import datetime
import json
import csv
import os
import asyncio
import io
from dotenv import load_dotenv 
load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN)
   
def predictSpeaker(json_data,minSec=0):
    oneMin = []
    if minSec == 0 :
        oneMin= json_data
    else:
     for om in json_data:
        totalDuration = float(om['stop']) - float(om['start']) 
        if int(totalDuration) >= minSec:
            oneMin.append(om)
    arrLast = sorted(oneMin, key=lambda x: float(x["start"]),reverse=True)
    arrFrist = sorted(oneMin, key=lambda x: float(x["start"]),reverse=False)
    filtered_data = [entry for entry in oneMin if entry['speaker'] in {arrFrist[0]['speaker']}]
    print(filtered_data,"filtered_data")
    return filtered_data

def Diarization(baseAudioFlile, targetAudioFile):
    print(f"Base Audio: {baseAudioFlile}, Target Audio: {targetAudioFile}") 
    
    base_waveform, sample_rate_base = torchaudio.load(baseAudioFlile) 
    target_waveform, sample_rate_target = torchaudio.load(targetAudioFile) 

    if sample_rate_base != sample_rate_target:
        raise print("Sample rates of base and target audio files must match.") 
    combined_waveform = torch.cat((base_waveform, target_waveform), dim=1)  
    
   
    st = datetime.now()
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": combined_waveform, "sample_rate": sample_rate_base}, hook=hook)
    
    
    my_array = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        my_array.append({"start": f"{turn.start:.1f}", "stop": f"{turn.end:.1f}", "speaker": f"{speaker}"})
    
    
    my_array = sorted(my_array, key=lambda x: x['speaker'])
    
  
    file_name = os.path.splitext(os.path.basename(baseAudioFlile))[0] + "_diarization"
    output_file_json = os.path.join("json", f"{file_name}.json")
    
    
    with open(output_file_json, 'w') as json_file:
        json.dump(my_array, json_file)
    
    
    print(f"Transcribe start at: {st} & Transcribe end at: {datetime.now()}")
    my_array = predictSpeaker(my_array,1)
    return my_array

def process_and_save_audio(audio_file, diarization_data, output_file):
    """
    Cuts the given audio file based on diarization start and stop times,
    merges the chunks into a single waveform in memory, and saves it as a WAV file.
    
    Parameters:
        audio_file (str): Path to the audio file to process.
        diarization_data (list): List of dictionaries containing "start", "stop", and "speaker".
        output_file (str): Path to save the merged audio as a WAV file.
    
    Returns:
        None
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Initialize an empty list to store the chunks
    chunks = []
    
    for segment in diarization_data:
        # Convert start and stop times to sample indices
        start_sample = int(float(segment["start"]) * sample_rate)
        stop_sample = int(float(segment["stop"]) * sample_rate)
        
        # Extract the chunk
        chunk = waveform[:, start_sample:stop_sample]
        chunks.append(chunk)
    
    # Merge all chunks into a single waveform
    if chunks:
        merged_waveform = torch.cat(chunks, dim=1)  # Concatenate along time axis
    else:
        raise print("No valid segments found in diarization data.")
    
     
    torchaudio.save(output_file, merged_waveform, sample_rate) 

baseAudioFolder="audioData/dataset" 
if not os.path.exists(baseAudioFolder):
    print("Folder does not exist")
else:
    for file_name in os.listdir(baseAudioFolder):
      if os.path.isfile(os.path.join(baseAudioFolder, file_name)):
        
        target_voice_path = r"F:\Targeted Voice Separation\utils\tommy.wav"
        audio_file_path = os.path.join(baseAudioFolder, file_name)
        outputFile=rf"F:\Targeted Voice Separation\audioData\segment"

        sourceArr = Diarization(audio_file_path,target_voice_path)
        
        if len(sourceArr) > 0 :
            process_and_save_audio(audio_file_path,sourceArr,"test.wav")
            
          
 