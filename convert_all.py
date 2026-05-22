import os
from pydub import AudioSegment

input_folder = "voice_dataset"
output_folder = "converted_wav"

os.makedirs(output_folder, exist_ok=True)

for person in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person)

    out_person_path = os.path.join(output_folder, person)
    os.makedirs(out_person_path, exist_ok=True)

    for file in os.listdir(person_path):
        if file.endswith(".m4a"):
            in_path = os.path.join(person_path, file)

            out_file = file.replace(".m4a", ".wav")
            out_path = os.path.join(out_person_path, out_file)

            audio = AudioSegment.from_file(in_path, format="m4a")
            audio.export(out_path, format="wav")

            print("Converted:", file)