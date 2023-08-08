import os
import torch
import openai
import librosa
import argparse
import subprocess
from pytube import YouTube
from create_captions import caption
from faster_whisper import WhisperModel

result = subprocess.run("mkdir videos", capture_output=True, text=True, shell=True)

parser = argparse.ArgumentParser(description="Parse command-line arguments")
parser.add_argument('--device',default = 'cuda:0',type = str,help = "device id")
parser.add_argument('--url',type = str,help = "enter url of youtube video")
parser.add_argument('--video_loc',type = str ,help = "enter video location")
parser.add_argument('--key',type = str,help = "Enter open ai api key")
parser.parse_args()


def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a AI model good at summarizing text."},
                  {"role": "user", "content": prompt}],
        max_tokens=120
    )
    
    return response['choices'][0]['message']['content'].strip()

def download_youtube_video(url, output_path=None):
    
    try:
        
        video = YouTube(url)

        video_stream = video.streams.get_highest_resolution()

        if not output_path:
            output_path = "."

        video_stream.download(output_path)

        print("Download completed successfully!")
        
    except Exception as e:
        
        print("Error:", e)
        
def main(args):
    current_directory = os.getcwd()
    if not hasattr(args,'gm_loc'):
        print("Please download GM flow sintel model from gmflow github repo")
        return
    if hasattr(args, 'device'):
        device = torch.device(args.device)
    if hasattr(args,'url'):
        download_youtube_video(args.url,os.path.join(current_directory,'/videos'))
        video_loc = os.path.join(os.path.join(current_directory,'/videos'),
                                 os.listdir(os.path.join(current_directory,'/videos'))[0])
        video_name = os.listdir(os.path.join(current_directory,'/videos'))[0]
    else:
        video_loc = args.video_loc
        video_name = args.video_loc.split('/')[-1]
    openai.api_key = args.key
    model = caption(device)
    caps = model.captions(video_loc,args.gm_loc)
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    audio,sr = librosa.load(video_loc,sr = 16000)
    segments, info = model.transcribe(audio, beam_size=5,word_timestamps = True)
    caps  = {x[0][0]:x[1] for x in caps}
    captions = ' \n'.join(x for x in caps)
    transcript = ' '.join(x.text for x in segments)
    prompt = f"""You are provided with a video name,video transcript and captions on the most important frames of the video. 
    Use this information to generate a concise summary of the video's content. you will analyze
    the transcript and captions to produce a summary without mentioning transcript ,captions and mention key sequences inside frames.

    Video name:
    [{video_name}]

    Video Transcript:
    [{transcript}]

    Captions on Most Important Video Frames:

    [{captions}]
    """
    print(generate_response(prompt))

        