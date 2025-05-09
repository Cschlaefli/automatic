from typing import Optional, List
from threading import Lock
from pydantic import BaseModel, Field, ConfigDict # pylint: disable=no-name-in-module
from modules import errors, shared, processing_helpers
from modules.api import models, helpers
from modules.control import run


errors.install()


class ItemControl(BaseModel):
    process: str = Field(title="Preprocessor", default="", description="")
    model: str = Field(title="Control Model", default="", description="")
    strength: float = Field(title="Control model strength", default=1.0, description="")
    start: float = Field(title="Control model start", default=0.0, description="")
    end: float = Field(title="Control model end", default=1.0, description="")
    override: str = Field(title="Override image", default=None, description="")

class StableDiffusionProcessingControl(BaseModel):
    model_config = ConfigDict(extra='allow') # to allow unit and additional params
    state: str = ''
    units: List[str] = Field(default_factory=list) # technically this is modules.control.unit.Unit, but API represents it as string
    # input inits and mask are Image.Image when passed to run, but are are base64 encoded images
    inputs: List[str] = Field(default_factory=list)
    inits: List[str] = Field(default_factory=list)
    mask: Optional[str] = None
    unit_type: Optional[str] = None
    is_generator: Optional[bool] = True
    input_type: int = 0
    prompt: str = ''
    negative_prompt: str = ''
    styles: list[str] = Field(default_factory=list)
    steps: int = 20
    sampler_index: Optional[int] = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    cfg_scale: float = 6.0
    clip_skip: float = 1.0
    image_cfg_scale: float = 6.0
    diffusers_guidance_rescale: float = 0.7
    pag_scale: float = 0.0
    pag_adaptive: float = 0.5
    cfg_end: float = 1.0
    full_quality: bool = True
    tiling: bool = False
    hidiffusion: bool = False
    detailer_enabled: bool = True
    detailer_prompt: str = ''
    detailer_negative: str = ''
    detailer_steps: int = 10
    detailer_strength: float = 0.3
    hdr_mode: int = 0
    hdr_brightness: float = 0
    hdr_color: float = 0
    hdr_sharpen: float = 0
    hdr_clamp: bool = False
    hdr_boundary: float = 4.0
    hdr_threshold: float = 0.95
    hdr_maximize: bool = False
    hdr_max_center: float = 0.6
    hdr_max_boundry: float = 1.0
    hdr_color_picker: str = None
    hdr_tint_ratio: float = 0
    resize_mode_before: int = 0
    resize_name_before: str = 'None'
    resize_context_before: str = 'None'
    width_before: int = 512
    height_before: int = 512
    scale_by_before: float = 1.0
    selected_scale_tab_before: int = 0
    resize_mode_after: int = 0
    resize_name_after: str = 'None'
    resize_context_after: str = 'None'
    width_after: int = 0
    height_after: int = 0
    scale_by_after: float = 1.0
    selected_scale_tab_after: int = 0
    resize_mode_mask: int = 0
    resize_name_mask: str = 'None'
    resize_context_mask: str = 'None'
    width_mask: int = 0
    height_mask: int = 0
    scale_by_mask: float = 1.0
    selected_scale_tab_mask: int = 0
    denoising_strength: float = 0.3
    batch_count: int = 1
    batch_size: int = 1
    enable_hr: bool = False
    hr_sampler_index: Optional[int] = None
    hr_denoising_strength: float = 0.0
    hr_resize_mode: int = 0
    hr_resize_context: str = 'None'
    hr_upscaler: Optional[str] = None
    hr_force: bool = False
    hr_second_pass_steps: int = 20
    hr_scale: float = 1.0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    refiner_steps: int = 5
    refiner_start: float = 0.0
    refiner_prompt: str = ''
    refiner_negative: str = ''
    video_skip_frames: int = 0
    video_type: str = 'None'
    video_duration: float = 2.0
    video_loop: bool = False
    video_pad: int = 0
    video_interpolate: int = 0
    sampler_name : str = "UniPC"
    script_name: Optional[str] = None
    script_args: list = Field(default_factory=list)
    send_images: bool = True
    save_images: bool = False
    alwayson_scripts: dict = Field(default_factory=dict)
    ip_adapter : Optional[List[models.ItemIPAdapter]] = Field(default=None, exclude=True)
    face : Optional[models.ItemFace]= Field(default= None, exclude= True)
    control : Optional[List[ItemControl]]= Field(default_factory=list, exclude=True)
    extra : Optional[dict]= Field(default_factory=dict, exclude= True)



class ResControl(BaseModel):
    images: List[str] = Field(default=None, title="Images", description="")
    processed: List[str] = Field(default=None, title="Processed", description="")
    params: dict = Field(default={}, title="Settings", description="")
    info: str = Field(default="", title="Info", description="")


class APIControl():
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg = []

    def sanitize_args(self, args: dict):
        args = vars(args)
        args.pop('sampler_name', None)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        args.pop('face', None)
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
        args.pop('save_images', None)
        return args

    def sanitize_b64(self, request):
        def sanitize_str(args: list):
            for idx in range(0, len(args)):
                if isinstance(args[idx], str) and len(args[idx]) >= 1000:
                    args[idx] = f"<str {len(args[idx])}>"
        if hasattr(request, "alwayson_scripts") and request.alwayson_scripts:
            for script_name in request.alwayson_scripts.keys():
                script_obj = request.alwayson_scripts[script_name]
                if script_obj and "args" in script_obj and script_obj["args"]:
                    sanitize_str(script_obj["args"])
        if hasattr(request, "script_args") and request.script_args:
            sanitize_str(request.script_args)

    def prepare_face_module(self, req):
        if hasattr(req, "face") and req.face and not req.script_name and (not req.alwayson_scripts or "face" not in req.alwayson_scripts.keys()):
            req.script_name = "face"
            req.script_args = [
                req.face.mode,
                req.face.source_images,
                req.face.ip_model,
                req.face.ip_override_sampler,
                req.face.ip_cache_model,
                req.face.ip_strength,
                req.face.ip_structure,
                req.face.id_strength,
                req.face.id_conditioning,
                req.face.id_cache,
                req.face.pm_trigger,
                req.face.pm_strength,
                req.face.pm_start,
                req.face.fs_cache
            ]
            del req.face

    def prepare_ip_adapter(self, request):
        if hasattr(request, "ip_adapter") and request.ip_adapter:
            args = { 'ip_adapter_names': [], 'ip_adapter_scales': [], 'ip_adapter_crops': [], 'ip_adapter_starts': [], 'ip_adapter_ends': [], 'ip_adapter_images': [], 'ip_adapter_masks': [] }
            for ipadapter in request.ip_adapter:
                if not ipadapter.images or len(ipadapter.images) == 0:
                    continue
                args['ip_adapter_names'].append(ipadapter.adapter)
                args['ip_adapter_scales'].append(ipadapter.scale)
                args['ip_adapter_starts'].append(ipadapter.start)
                args['ip_adapter_ends'].append(ipadapter.end)
                args['ip_adapter_crops'].append(ipadapter.crop)
                args['ip_adapter_images'].append([helpers.decode_base64_to_image(x) for x in ipadapter.images])
                if ipadapter.masks:
                    args['ip_adapter_masks'].append([helpers.decode_base64_to_image(x) for x in ipadapter.masks])

            del request.ip_adapter
            return args
        else:
            return {}

    def prepare_control(self, req):
        from modules.control.unit import Unit, unit_types
        req.units = []
        if req.unit_type is None:
            return req.control
        if req.unit_type not in unit_types:
            shared.log.error(f'Control uknown unit type: type={req.unit_type} available={unit_types}')
            return req.control
        for u in req.control:
            unit = Unit(
                enabled = True,
                unit_type = req.unit_type,
                model_id = u.model,
                process_id = u.process,
                strength = u.strength,
                start = u.start,
                end = u.end,
            )
            if u.override is not None:
                unit.override = helpers.decode_base64_to_image(u.override)
            req.units.append(unit)
        return req.control

    def post_control(self, req: StableDiffusionProcessingControl):
        self.prepare_face_module(req)
        orig_control = self.prepare_control(req)
        del req.control

        # prepare args
        args = req.model_copy(update={  # Override __init__ params
            "sampler_index": processing_helpers.get_sampler_index(req.sampler_name),
            "is_generator": True,
            "inputs": [helpers.decode_base64_to_image(x) for x in req.inputs] if req.inputs else None,
            "inits": [helpers.decode_base64_to_image(x) for x in req.inits] if req.inits else None,
            "mask": helpers.decode_base64_to_image(req.mask) if req.mask else None,
        }) # since this doesn't validate the model, this works fine

        args = self.sanitize_args(args)
        send_images = args.pop('send_images', True)

        # run
        with self.queue_lock:
            shared.state.begin('API-CTL', api=True)
            output_images = []
            output_processed = []
            output_info = ''
            # TODO control: support scripts via api
            # init script args, call scripts.script_control.run, call scripts.script_control.after
            run.control_set({ 'do_not_save_grid': not req.save_images, 'do_not_save_samples': not req.save_images, **self.prepare_ip_adapter(req) })
            run.control_set(getattr(req, "extra", {}))
            res = run.control_run(**args)
            for item in res:
                if len(item) > 0 and (isinstance(item[0], list) or item[0] is None): # output_images
                    output_images += item[0] if item[0] is not None else []
                    output_processed += [item[1]] if item[1] is not None else []
                    output_info += item[2] if len(item) > 2 and item[2] is not None else ''
                elif isinstance(item, str):
                    output_info += item
                else:
                    pass
            shared.state.end(api=False)

        # return
        b64images = list(map(helpers.encode_pil_to_base64, output_images)) if send_images else []
        b64processed = list(map(helpers.encode_pil_to_base64, output_processed)) if send_images else []
        self.sanitize_b64(req)
        req.units = orig_control
        return ResControl(images=b64images, processed=b64processed, params=vars(req), info=output_info)
