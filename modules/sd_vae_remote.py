import io
import time
import json
import torch
import requests
from PIL import Image
from safetensors.torch import _tobytes


hf_endpoints = {
    'sd': 'https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud',
    'sdxl': 'https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud',
    'f1': 'https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud',
    'hunyuanvideo': 'https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud',
}
dtypes = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


def remote_decode(latents: torch.Tensor, width: int = 0, height: int = 0, model_type: str = None) -> Image.Image:
    from modules import devices, shared, errors, modelloader
    tensors = []
    content = 0
    model_type = model_type or shared.sd_model_type
    url = hf_endpoints.get(model_type, None)
    if url is None:
        shared.log.error(f'Decode: type="remote" type={model_type} unsuppported')
        return tensors
    t0 = time.time()
    modelloader.hf_login()
    latents = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
    for i in range(latents.shape[0]):
        try:
            latent = latents[i].detach().clone().to(device=devices.cpu, dtype=devices.dtype).unsqueeze(0)
            params = {
                "do_scaling": True,
                "input_tensor_type": "binary",
                "shape": list(latent.shape),
                "dtype": str(latent.dtype).split(".", maxsplit=1)[-1],
            }
            headers = { "Content-Type": "tensor/binary" }
            if shared.opts.remote_vae_type == 'png':
                params["image_format"] = "png"
                params["output_type"] = "pil"
                headers["Accept"] = "image/png"
            elif shared.opts.remote_vae_type == 'jpg':
                params["image_format"] = "jpg"
                params["output_type"] = "pil"
                headers["Accept"] = "image/jpeg"
            elif shared.opts.remote_vae_type == 'raw':
                params["partial_postprocess"] = False
                params["output_type"] = "pt"
                params["output_tensor_type"] = "binary"
                headers["Accept"] = "tensor/binary"
            if (model_type == 'f1') and (width > 0) and (height > 0):
                params['width'] = width
                params['height'] = height
            response = requests.post(
                url=url,
                headers=headers,
                params=params,
                data=_tobytes(latent, "tensor"),
                timeout=300,
            )
            if not response.ok:
                shared.log.error(f'Decode: type="remote" model={model_type} code={response.status_code} headers={response.headers} {response.json()}')
            else:
                content += len(response.content)
                if shared.opts.remote_vae_type == 'raw':
                    shape = json.loads(response.headers["shape"])
                    dtype = response.headers["dtype"]
                    tensor = torch.frombuffer(bytearray(response.content), dtype=dtypes[dtype]).reshape(shape)
                    tensors.append(tensor)
                elif shared.opts.remote_vae_type == 'jpg' or shared.opts.remote_vae_type == 'png':
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    tensors.append(image)
        except Exception as e:
            shared.log.error(f'Decode: type="remote" model={model_type} {e}')
            errors.display(e, 'VAE')
    if len(tensors) > 0 and shared.opts.remote_vae_type == 'raw':
        tensors = torch.cat(tensors, dim=0)
    t1 = time.time()
    shared.log.debug(f'Decode: type="remote" model={model_type} mode={shared.opts.remote_vae_type} args={params} bytes={content} time={t1-t0:.3f}s')
    return tensors
