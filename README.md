# Speaker Diarization and Targeted Voice Separation

This project performs speaker diarization and targeted voice separation using the `pyannote.audio` library. 
It identifies and processes specific speaker segments from audio files, combining them into a single output file.

## **Features**
- Diarizes audio to detect different speakers.
- Filters speaker segments based on duration and speaker matching.
- Merges and saves targeted audio segments into a single output file.

## **Setup Instructions**

### **1. Clone the Repository**

git clone https://github.com/Chiyan200/targeted-voice-separation.git
cd targeted-voice-separation.git

```bash

### **2. Install Dependencies**
Install the required Python packages from the requirements.txt file

### **3. Set Up Environment Variables**

Create a .env file in the root directory and add your Hugging Face token:
HF_TOKEN=your_hugging_face_token

### **4. Prepare Audio Files**
Place the dataset audio files in the audioData/dataset folder.
Provide the target voice file as target_voice_path.

### **5. Run the Script**
Place the dataset audio files in the audioData/dataset folder.
Provide the target voice file as target_voice_path.

python script_name.py

### **5. Project Structure**
.
├── audioData/
│   ├── dataset/                 # Folder for input audio files
│   ├── segment/                 # Folder for processed audio output
├── json/                        # Folder for diarization JSON outputs
├── script_name.py               # Main Python script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # Documentation

Functions
    Diarization
    Performs speaker diarization and filters speaker data based on minimum duration.

predictSpeaker
    Filters the diarization data to include segments for the targeted speaker.

process_and_save_audio
    Cuts and merges specific audio segments into a single waveform based on diarization results.

Output
    A JSON file containing diarization results is saved in the json/ folder.
    A merged WAV file containing targeted audio segments is saved in the specified output path.

```

