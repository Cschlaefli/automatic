import re
import time
import numpy as np
import networks
import lora_patches
from modules import extra_networks, shared


# from https://github.com/cheald/sd-webui-loractl/blob/master/loractl/lib/utils.py
def get_stepwise(param, step, steps):
    def sorted_positions(raw_steps):
        steps = [[float(s.strip()) for s in re.split("[@~]", x)]
                 for x in re.split("[,;]", str(raw_steps))]
        # If we just got a single number, just return it
        if len(steps[0]) == 1:
            return steps[0][0]

        # Add implicit 1s to any steps which don't have a weight
        steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps]

        # Sort by index
        steps.sort(key=lambda k: k[1])

        steps = [list(v) for v in zip(*steps)]
        return steps

    def calculate_weight(m, step, max_steps, step_offset=2):
        if isinstance(m, list):
            if m[1][-1] <= 1.0:
                if max_steps > 0:
                    step = (step) / (max_steps - step_offset)
                else:
                    step = 1.0
            v = np.interp(step, m[1], m[0])
            return v
        else:
            return m
    return calculate_weight(sorted_positions(param), step, steps)


class ExtraNetworkLora(extra_networks.ExtraNetwork):

    def __init__(self):
        super().__init__('lora')
        self.active = False
        self.errors = {}
        networks.originals = lora_patches.LoraPatches()

    def prompt(self, p):
        if shared.opts.lora_apply_tags == 0:
            return
        all_tags = []
        for loaded in networks.loaded_networks:
            page = [en for en in shared.extra_networks if en.name == 'lora'][0]
            item = page.create_item(loaded.name)
            tags = (item or {}).get("tags", {})
            loaded.tags = list(tags)
            if len(loaded.tags) == 0:
                loaded.tags.append(loaded.name)
            if shared.opts.lora_apply_tags > 0:
                loaded.tags = loaded.tags[:shared.opts.lora_apply_tags]
            all_tags.extend(loaded.tags)
        if len(all_tags) > 0:
            shared.log.debug(f"Load network: type=LoRA max={shared.opts.lora_apply_tags} tags={all_tags} apply")
            all_tags = ', '.join(all_tags)
            p.extra_generation_params["LoRA tags"] = all_tags
            if p.all_prompts is not None:
                for i in range(len(p.all_prompts)):
                    if '_tags_' in p.all_prompts[i]:
                        p.all_prompts[i] = p.all_prompts[i].replace('_tags_', all_tags)
                    else:
                        p.all_prompts[i] = f"{p.all_prompts[i]}, {all_tags}"

    def infotext(self, p):
        names = [i.name for i in networks.loaded_networks]
        if len(names) > 0:
            p.extra_generation_params["LoRA networks"] = ", ".join(names)
        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                if not item.network_on_disk.shorthash:
                    continue
                network_hashes.append(item.network_on_disk.shorthash)
            if len(network_hashes) > 0:
                p.extra_generation_params["LoRA hashes"] = ", ".join(network_hashes)

    def parse(self, p, params_list, step=0):
        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items
            names.append(params.positional[0])
            te_multiplier = params.named.get("te", params.positional[1] if len(params.positional) > 1 else shared.opts.extra_networks_default_multiplier)
            if isinstance(te_multiplier, str) and "@" in te_multiplier:
                te_multiplier = get_stepwise(te_multiplier, step, p.steps)
            else:
                te_multiplier = float(te_multiplier)
            unet_multiplier = [params.positional[2] if len(params.positional) > 2 else te_multiplier] * 3
            unet_multiplier = [params.named.get("unet", unet_multiplier[0])] * 3
            unet_multiplier[0] = params.named.get("in", unet_multiplier[0])
            unet_multiplier[1] = params.named.get("mid", unet_multiplier[1])
            unet_multiplier[2] = params.named.get("out", unet_multiplier[2])
            for i in range(len(unet_multiplier)):
                if isinstance(unet_multiplier[i], str) and "@" in unet_multiplier[i]:
                    unet_multiplier[i] = get_stepwise(unet_multiplier[i], step, p.steps)
                else:
                    unet_multiplier[i] = float(unet_multiplier[i])
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
        return names, te_multipliers, unet_multipliers, dyn_dims

    def activate(self, p, params_list, step=0):
        t0 = time.time()
        self.errors.clear()
        if len(params_list) > 0:
            self.active = True
            networks.originals.apply() # apply patches
            if networks.debug:
                shared.log.debug(f"LoRA activate: networks={len(params_list)}")
        t1 = time.time()
        names, te_multipliers, unet_multipliers, dyn_dims = self.parse(p, params_list, step)
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
        t2 = time.time()
        if len(names) > 0 and step == 0:
            self.infotext(p)
            self.prompt(p)
            shared.log.info(f'Load network: type=LoRA apply={names} patch={t1-t0:.2f} te={te_multipliers} unet={unet_multipliers} dims={dyn_dims} load={t2-t1:.2f}')
        elif self.active:
            self.active = False

    def deactivate(self, p):
        t0 = time.time()
        if shared.native and hasattr(shared.sd_model, "unload_lora_weights") and hasattr(shared.sd_model, "text_encoder"):
            if not (shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled is True):
                try:
                    if shared.opts.lora_fuse_diffusers:
                        shared.sd_model.unfuse_lora()
                    shared.sd_model.unload_lora_weights() # fails for non-CLIP models
                    # shared.log.debug("LoRA unload")
                except Exception:
                    # shared.log.warning(f"LoRA unload: {e}")
                    pass
        t1 = time.time()
        networks.timer['restore'] += t1 - t0
        if self.active and networks.debug:
            shared.log.debug(f"LoRA end: load={networks.timer['load']:.2f} apply={networks.timer['apply']:.2f} restore={networks.timer['restore']:.2f}")
        if self.active and getattr(networks, "originals", None ) is not None:
            networks.originals.undo() # remove patches
            if networks.debug:
                shared.log.debug("LoRA deactivate")
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            for k, v in self.errors.items():
                shared.log.error(f'LoRA errors: file="{k}" errors={v}')
            self.errors.clear()
