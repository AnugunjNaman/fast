import os
import numpy as np
from wav2vec import Wav2Vec

def create_model(model_checkpoint):
    print("Creating model")
    wav_2_vec_model = Wav2Vec(model_checkpoint)
    wav_2_vec_model
    return wav_2_vec_model


model_checkpoint = "./CLSRIL-23.pt" 
wav_2_vec_model = create_model(model_checkpoint)
print(wav_2_vec_model)
print(sum(p.numel() for p in wav_2_vec_model.parameters() if p.requires_grad))
exit()

# Assuming you have a function 'extract_features' that takes a filename and returns numpy features
def extract_features(file_name):
    # Replace this with your actual implementation to extract features
    # This is just a placeholder
    features = wav_2_vec_model.extract_features(file_name).cpu().detach().numpy()
    return features

# Path to the folder containing WAV files
folder_path = './files/SC_audio_Hindi/'

# Get a list of all WAV files in the folder
wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]

# Process each WAV file and save numpy features as .npy file with the same filename
from tqdm import tqdm

for wav_file in tqdm(wav_files):
    file_path = os.path.join(folder_path, wav_file)

    # Extract features using your function
    features = extract_features(file_path)

    # Create the corresponding .npy file path
    npy_file_path = os.path.splitext(file_path)[0] + '.npy'

    # Save the numpy features as .npy file
    np.save(npy_file_path, features)

print("Conversion complete!")

