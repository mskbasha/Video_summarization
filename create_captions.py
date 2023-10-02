import torch
import cv2 as cv
import numpy as np
from PIL import Image
from typing import List
from .Pbar import progressbar
from .gmflow.gmflow.gmflow import GMFlow
from lavis.models import load_model_and_preprocess
from cnocr import CnOcr
class caption:
    
    def __init__(self, device : torch.device, gm_loc : str,model_type="pretrain_flant5xl") -> None:
        self.device = device
        self.model_type = model_type
        self.load_models(gm_loc)
        self.vel = 60
        self.batch_size = 10
        self.framesToSkip = 5
        self.prompt =  """Caption this image:
Use below given text in square brackets [{}] which are text on the image in no particular order. Generate a caption describing the entire image with only the text provided in the brackets."""
        self.ocr = CnOcr(device = device)  
    def load_models(self,gm_loc : str):
        self.model = GMFlow(feature_channels = 128,
                        num_scales = 1,
                        upsample_factor = 8,
                        num_head = 1,
                        attention_type = 'swin',
                        ffn_dim_expansion = 4,
                        num_transformer_layers = 6,
                        ).to(self.device)
        checkpoint = torch.load(gm_loc, map_location= self.device)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict='store_true')
        self.model_blip, self.vis_processors, _ = load_model_and_preprocess(name = "blip2_t5", 
                                                                model_type = "flant5xxl", 
                                                                is_eval = True, 
                                                                device = self.device)
    def cap(self,image):
        image = np.flip(image,axis = -1)
        result = self.ocr.ocr(image)
        frame_0 = Image.fromarray(image)
        image = self.vis_processors["eval"](frame_0).unsqueeze(0).to(self.device)
        prompt = self.prompt.format(' , '.join([i['text'] for i in result if i['score']>0.5]))
        return self.model_blip.generate({"image": image, "prompt":prompt}),prompt
    def captions(self, video_loc :str ,pr = 1,total = 1) :  
        video = cv.VideoCapture(video_loc)
        prev_frames,next_frames,framesForCaptions,times,goodframes = [],[],[],[],[]
        pb = progressbar(1)
        fps = video.get(cv.CAP_PROP_FPS)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        c = 0
        count = 0
        time = 0
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        goodframes.append([self.cap(video.read()[1])[0],0])
        while True:
            _,frame1 = video.read()
            if not _:break
            for i in range(self.framesToSkip*2): _,frame2 = video.read()
            if not _:break
            pb.print(c*2*self.framesToSkip,total_frames,f""" 
            \r Total frames read {c*10}/{total_frames} 
            \r time = {time}
            \r good frames ={len(goodframes)}
            \r {pr}/{total} videos""")
            c+=1
            time = c*2*self.framesToSkip/fps
            framesForCaptions.append(frame2)
            prev_frames.append(torch.from_numpy(np.flip( cv.resize(frame1,(320,160))  ,axis=-1).copy() ).permute(2,0,1))
            next_frames.append(torch.from_numpy( np.flip( cv.resize(frame2,(320,160))  ,axis=-1).copy()).permute(2,0,1))
            times.append(time)
            count+=1
            with torch.no_grad():
                if count==self.batch_size:
                    image1 = torch.stack(prev_frames).float()
                    image2 = torch.stack(next_frames).float()
                    results_dict = self.model(image1.to(self.device), image2.to(self.device),
                                            attn_splits_list=[2],
                                            corr_radius_list=[-1],
                                            prop_radius_list=[-1],
                                            pred_bidir_flow=False,
                                        )
                    velocity = results_dict['flow_preds'][-1]
                    velocity = torch.sqrt(velocity[:,0]**2+velocity[:,1]**2)
                    velocity = torch.sum(velocity,axis=[1,2])/sum(velocity.shape[1:])>self.vel
                    for ind,i in enumerate(velocity):
                        if i:goodframes.append([self.cap(framesForCaptions[ind]),times[ind]])
                    count = 0
                    prev_frames,next_frames,framesForCaptions,times = [],[],[],[]
        return goodframes
