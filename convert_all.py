import os
from pydub import AudioSegment

input_folder = "voice_dataset"
output_folder = "wav_files"

os.makedirs(output_folder, exist_ok=True)

converted = 0
skipped = 0

for person in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person)

    if not os.path.isdir(person_path):
        continue

    out_person_path = os.path.join(output_folder, person)
    os.makedirs(out_person_path, exist_ok=True)

    for file in os.listdir(person_path):
        if not file.endswith(".m4a"):
            continue

        in_path = os.path.join(person_path, file)
        out_file = file.replace(".m4a", ".wav")
        out_path = os.path.join(out_person_path, out_file)

        try:
            audio = AudioSegment.from_file(in_path, format="m4a")
            # Normalize to 16kHz mono to match training pipeline
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(out_path, format="wav")
            print(f"Converted: {person}/{file}")
            converted += 1
        except Exception as e:
            print(f"Failed: {person}/{file} — {e}")
            skipped += 1

print(f"\nDone. Converted: {converted}, Skipped: {skipped}")
