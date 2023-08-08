import torch
import cv2 as cv
import numpy as np
from PIL import Image
from typing import List
from Pbar import progressbar
from gmflow.gmflow import GMFlow
from lavis.models import load_model_and_preprocess

class caption:
    
    def __init__(self, device : torch.device) -> None:
        self.device = device
        
    def load_models(self,gm_loc : str):
        
        model = GMFlow(feature_channels = 128,
                        num_scales = 1,
                        upsample_factor = 8,
                        num_head = 1,
                        attention_type = 'swin',
                        ffn_dim_expansion = 4,
                        num_transformer_layers = 6,
                        ).to(self.device)
        checkpoint = torch.load(gm_loc, map_location= self.device)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(weights, strict='store_true')
        model_blip, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", 
                                                                model_type="pretrain_flant5xxl", 
                                                                is_eval=True, 
                                                                device=self.device)
        return model,model_blip,vis_processors
    
    def captions(self, video_loc :str , gm_loc : str,vel = 60) -> List[str,float]:
        
        model,model_blip,vis_processors = self.load_models(self.device,gm_loc)
        video = cv.VideoCapture(video_loc)
        prev_frames,next_frames,framesForCaptions,times,goodframes = [],[],[],[],[]
        pb = progressbar(1)
        fps = video.get(cv.CAP_PROP_FPS)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        c = 0
        count = 0
        time = 0
        frame_no = 0
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        while True:
            for i in range(5):
                frame_no+=1
                _,frame1 = video.read()
            if not _:break
            for i in range(5):
                frame_no+=1
                _,frame2 = video.read()
            if not _:break
            pb.print(c*10,total_frames,f""" 
            Total frames read {c*10}/{total_frames} 
            time = {time}
            good frames ={len(goodframes)}""")
            c+=1
            time = c*10/fps
            framesForCaptions.append(frame2)
            prev_frames.append(torch.from_numpy(np.flip( cv.resize(frame1,(320,160))  ,axis=-1).copy() ).permute(2,0,1))
            next_frames.append(torch.from_numpy( np.flip( cv.resize(frame2,(320,160))  ,axis=-1).copy()).permute(2,0,1))
            times.append(time)
            count+=1
            batch_size = 20
            with torch.no_grad():
                if count==batch_size:
                    image1 = torch.stack(prev_frames).float()
                    image2 = torch.stack(next_frames).float()
                    results_dict = model(image1.to(self.device), image2.to(self.device),
                                            attn_splits_list=[2],
                                            corr_radius_list=[-1],
                                            prop_radius_list=[-1],
                                            pred_bidir_flow=False,
                                        )
                    velocity = results_dict['flow_preds'][-1]
                    velocity = torch.sqrt(velocity[:,0]**2+velocity[:,1]**2)
                    velocity = torch.sum(velocity,axis=[1,2])/sum(velocity.shape[1:])>vel
                    torch.cuda.empty_cache()
                    for ind,i in enumerate(velocity):
                        if i:
                            frame_0 = Image.fromarray(np.flip(framesForCaptions[ind],axis = -1))
                            image = vis_processors["eval"](frame_0).unsqueeze(0).to(self.device)
                            goodframes.append([model_blip.generate({"image": image, 
                                                                    "prompt": "Caption this image"}),times[ind]])
                    count = 0
                    prev_frames,next_frames,framesForCaptions,times = [],[],[],[],[]
        torch.cuda.empty_cache()
        return goodframes