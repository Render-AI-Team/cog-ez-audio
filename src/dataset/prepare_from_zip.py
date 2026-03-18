import os
import zipfile
import csv
import shutil
from mutagen import File as MutagenFile
from mutagen.easyid3 import EasyID3
import tempfile
import sys

def extract_metadata(mp3_path):
    try:
        audio = MutagenFile(mp3_path)
        title = os.path.splitext(os.path.basename(mp3_path))[0]
        comment_fields = []
        if audio is not None:
            # Extract both comment fields
            if 'COMM::eng' in audio:
                comm = audio['COMM::eng']
                if hasattr(comm, 'text'):
                    comment_fields.append(comm.text if isinstance(comm.text, str) else ' '.join(comm.text))
            if 'COMM:ID3v1 Comment:eng' in audio:
                comm_v1 = audio['COMM:ID3v1 Comment:eng']
                if hasattr(comm_v1, 'text'):
                    comment_fields.append(comm_v1.text if isinstance(comm_v1.text, str) else ' '.join(comm_v1.text))
            # Also try mutagen's easy interface for 'comment'
            easy_audio = MutagenFile(mp3_path, easy=True)
            if easy_audio is not None:
                easy_comment = easy_audio.get('comment', [''])[0]
                if easy_comment:
                    comment_fields.append(easy_comment)
        caption = ' | '.join([c for c in comment_fields if c]) if comment_fields else title
        return title, caption
    except Exception as e:
        return os.path.splitext(os.path.basename(mp3_path))[0], ''

def create_csv_from_zip(zip_path, out_dir, csv_path):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    mp3_files = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, f))
    prepend = getattr(create_csv_from_zip, 'prepend', '')
    add_folders = getattr(create_csv_from_zip, 'add_folders', False)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['audio_path', 'caption', 'split'])
        for i, mp3 in enumerate(mp3_files):
            title, caption = extract_metadata(mp3)
            rel_path = os.path.relpath(mp3, out_dir)
            folder_prefix = ''
            if add_folders:
                folder_parts = os.path.dirname(rel_path).split(os.sep)
                folder_prefix = ' '.join(folder_parts).strip()
            full_caption = ' '.join([s for s in [prepend, folder_prefix, caption] if s])
            writer.writerow([rel_path, full_caption, 'train'])
    print(f"Wrote CSV: {csv_path} with {len(mp3_files)} entries.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create dataset CSV from zip of mp3s')
    parser.add_argument('--zip', type=str, required=True, help='Path to zip file of mp3s')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to extract mp3s')
    parser.add_argument('--csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--prepend', type=str, default='', help='String to prepend to each caption')
    parser.add_argument('--add_folders', action='store_true', help='If set, prepend folder names to caption')
    args = parser.parse_args()
    create_csv_from_zip.prepend = args.prepend
    create_csv_from_zip.add_folders = args.add_folders
    create_csv_from_zip(args.zip, args.out_dir, args.csv)
