import json 
from langchain.llms.base import LLM
from typing import Mapping, Optional, Any, List
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests  

from env import STABLE_DIFFUSION_API_KEY, STABLE_DIFFUSION_TEXT2IMG_URL

class PictureGeneratorLLM(LLM):
    model_name = "Stable Diffusion" 
    text_2_img_url = STABLE_DIFFUSION_TEXT2IMG_URL 

    @property
    def _llm_type(self) -> str:
        return "custom" 
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        params_dict = {
            "model_name": self.model_name,
            "text_2_img_url": self.text_2_img_url
        }
        return params_dict

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str: 
        payload = {
            "key": STABLE_DIFFUSION_API_KEY,
            "prompt": None,
            "negative_prompt": None,
            "width": "512",
            "height": "512",
            "samples": "1",
            "num_inference_steps": "20",
            "seed": None,
            "guidance_scale": 7.5,
            "safety_checker": "yes",
            "multi_lingual": "no",
            "panorama": "no",
            "self_attention": "no",
            "upscale": "no",
            "embeddings_model": None,
            "webhook": None,
            "track_id": None
            }

        headers = {
            'Content-Type': 'application/json'
        }
        payload["prompt"] = prompt 
        payload_str = json.dumps(payload)
        response = requests.request("POST", self.text_2_img_url, 
                                    headers=headers, data=payload_str)
        response_data = response.text
        response_json = json.loads(response_data) 
        output = response_json.get("output", []) 
        if output:
            return output[0] 
        return None