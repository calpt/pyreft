import os
import torch
import json
import re


def convert_checkpoint(path: str, name: str):
    state_dict = {}
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)
    files = [f"intkey_{n}.bin" for n in config["sorted_keys"]]
    files = [os.path.join(path, f) for f in files]
    for int_type, key, file in zip(config["intervention_types"], config["sorted_keys"], files):
        # should match for Roberta
        m = re.match(r"comp\.(\S+)\.output\..+#(\d)", key)
        if m:
            prefix = m.group(1).replace("[", ".").replace("]", ".")
            idx = int(m.group(2))
        # handling for Llama
        else:
            prefix = "layers." + key.split(".")[1] + "."
            idx = int(key.split("#")[1])
        reft_dict = torch.load(file, map_location="cpu")
        if "Loreft" in int_type:
            assert set(reft_dict.keys()) == {"weight", "bias", "rotate_layer"}, reft_dict.keys()
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.learned_source.weight"] = reft_dict["weight"]
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.learned_source.bias"] = reft_dict["bias"]
            # adapted from pyreft interventions implementation
            overload_w = reft_dict["rotate_layer"]
            overload_w_width = overload_w.shape[-1]
            new_w = torch.zeros(overload_w.shape[0], overload_w.shape[0])
            new_w[:,:overload_w_width] = overload_w
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.projection.parametrizations.weight.0.base"] = new_w
        elif "Noreft" in int_type or "Nodireft" in int_type:
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.learned_source.weight"] = reft_dict["learned_source.weight"]
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.learned_source.bias"] = reft_dict["learned_source.bias"]
            state_dict[f"{prefix}reft_layer.refts.{name}.units.{idx}.projection.weight"] = reft_dict["proj_layer.weight"]

    return state_dict
