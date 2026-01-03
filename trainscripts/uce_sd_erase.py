import os
import time
import copy
import argparse
from typing import Dict, List, Tuple

import torch
torch.set_grad_enabled(False)

from safetensors.torch import save_file
from diffusers import DiffusionPipeline

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Utilities
# ============================================================

def _pick_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "float32").lower().strip()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        out.append(k)
    return out


def apply_prompt_templates(concepts: List[str], concept_type: str) -> List[str]:
    """
    Turn concept phrases into diffusion-like prompts.
    For policy removal, templates improve generalization.
    """
    concept_type = (concept_type or "object").lower().strip()
    if concept_type == "art":
        templates = [
            "{c}",
            "art by {c}",
            "style of {c}",
            "painting by {c}",
            "artwork by {c}",
        ]
    else:
        templates = [
            "{c}",
            "a photo of {c}",
            "an image of {c}",
            "a picture of {c}",
            "a realistic photo of {c}",
            "high quality image of {c}",
            "{c}, detailed",
        ]

    out = []
    for c in concepts:
        c = " ".join(c.strip().split())
        if not c:
            continue
        for t in templates:
            out.append(t.format(c=c))

    return _dedupe_preserve_order(out)


# ============================================================
# LLM-based concept expansion (lightweight)
# ============================================================

def expand_concepts_with_llm(
    base_concepts: List[str],
    device: str,
    llm_model_id: str,
    llm_max_new_tokens: int,
    llm_temperature: float,
    llm_top_p: float,
) -> List[str]:
    """
    Expand high-level policy concepts into many phrases using an open-source LLM.
    Output is a list of phrases (not templated prompts).
    """

    base_concepts = _dedupe_preserve_order([c.strip() for c in base_concepts if c.strip()])
    if not base_concepts:
        return []

    # Load LLM
    tok = AutoTokenizer.from_pretrained(llm_model_id, use_fast=True)
    # Use fp16 on CUDA, fp32 on CPU for stability
    torch_dtype = torch.float16 if ("cuda" in device and torch.cuda.is_available()) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if ("cuda" in device and torch.cuda.is_available()) else None,
    )

    system_prompt = (
        "You are an expert in content moderation and prompt analysis for text-to-image models.\n"
        "Task: expand high-level concepts into a very comprehensive list of short phrases that represent them.\n"
        "Output rules:\n"
        "- Output only newline-separated phrases\n"
        "- No numbering, no bullets, no explanations\n"
        "- No markdown\n"
        "- Each line is a distinct phrase someone might type in an image generation prompt\n"
        "- Include synonyms, euphemisms, indirect phrasing, and related terms\n"
        "- Keep phrases short (2 to 8 words)\n"
    )

    user_prompt = (
        "Concepts to expand:\n"
        + "\n".join(base_concepts)
        + "\n\nGenerate as many distinct phrases as possible."
    )

    # Qwen instruct format typically works with plain text prompts too.
    prompt = system_prompt + "\n" + user_prompt + "\n"

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(llm_max_new_tokens),
            do_sample=True,
            temperature=float(llm_temperature),
            top_p=float(llm_top_p),
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # Keep only lines after the prompt (best-effort)
    # If the model echoes the prompt, we still filter aggressively.
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Remove likely prompt/instruction echoes
    bad_prefixes = (
        "you are", "task:", "output rules", "concepts to expand", "generate", "concepts:",
    )

    phrases = []
    for l in lines:
        ll = l.lower()
        if any(ll.startswith(p) for p in bad_prefixes):
            continue
        # Remove bullet markers if the model adds them
        l = l.lstrip("-•*0123456789. \t").strip()
        if len(l) < 3:
            continue
        if len(l.split()) > 12:
            continue
        phrases.append(l)

    return _dedupe_preserve_order(phrases)


# ============================================================
# Text embedding extraction (robust pooling)
# ============================================================

def _encode_concept_vector(pipe, prompt: str, device: str) -> torch.Tensor:
    """
    Returns [1, D] concept vector using masked mean pooling across tokens.
    More robust than picking the last token.
    """
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    attn_mask = tok["attention_mask"].to(device)  # [1, T]

    enc = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    emb = enc[0]  # typically [1, T, D]
    if emb.dim() != 3:
        raise RuntimeError(f"Unexpected encode_prompt embedding shape: {tuple(emb.shape)}")

    mask = attn_mask.unsqueeze(-1).to(dtype=emb.dtype)  # [1, T, 1]

    # Exclude BOS token position (often 0)
    if mask.shape[1] > 0:
        mask[:, 0:1, :] = 0

    # Exclude last real token (often EOS)
    lengths = attn_mask.sum(dim=1)  # [1]
    last_idx = torch.clamp(lengths - 1, min=0)
    for b in range(mask.shape[0]):
        li = int(last_idx[b].item())
        if 0 <= li < mask.shape[1]:
            mask[b, li:li+1, :] = 0

    denom = mask.sum(dim=1)
    pooled = (emb * mask).sum(dim=1) / denom.clamp(min=1.0)

    # If the prompt is empty, excluding BOS/EOS leaves no tokens. Fall back to pooling over
    # all non-pad tokens to produce a non-zero unconditional-style vector.
    if torch.any(denom == 0):
        fallback_mask = attn_mask.unsqueeze(-1).to(dtype=emb.dtype)
        fallback_denom = fallback_mask.sum(dim=1).clamp(min=1.0)
        pooled_fallback = (emb * fallback_mask).sum(dim=1) / fallback_denom
        use_fallback = denom.squeeze(-1) == 0
        pooled[use_fallback] = pooled_fallback[use_fallback]
    return pooled  # [1, D]


# ============================================================
# UCE core (moderation)
# ============================================================

def _find_uce_modules(pipe) -> Tuple[List[torch.nn.Module], List[str]]:
    modules = []
    names = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_v") or name.endswith("to_k")):
            modules.append(module)
            names.append(name)
    return modules, names


def _stable_solve_right(mat1: torch.Tensor, mat2: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Compute mat1 @ inv(mat2) without explicit inverse.
    """
    X_t = torch.linalg.solve(mat2.float().T, mat1.float().T)
    return X_t.T.to(dtype=out_dtype)


def UCE_moderation(
    pipe,
    edit_prompts: List[str],
    preserve_prompts: List[str],
    erase_scale: float,
    preserve_scale: float,
    lamb: float,
    save_dir: str,
    exp_name: str,
    device: str,
    save_format: str,
    guide_prompt: str = "",
) -> Dict[str, object]:
    """
    Moderation UCE: map edit prompts to the output of guide_prompt (default empty, unconditional).
    Saves delta or absolute weights for attn2.to_k/to_v.
    """

    guide_prompt = (guide_prompt or "").strip()

    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    uce_modules, uce_module_names = _find_uce_modules(pipe)
    if not uce_modules:
        raise RuntimeError("No attn2.to_k/to_v modules found. Check your model architecture.")

    original_modules = uce_modules
    edited_modules = copy.deepcopy(uce_modules)

    # Collect concept vectors once
    concept_vecs: Dict[str, torch.Tensor] = {}
    all_prompts = _dedupe_preserve_order(edit_prompts + preserve_prompts)
    for p in all_prompts:
        if p in concept_vecs:
            continue
        concept_vecs[p] = _encode_concept_vector(pipe, p, device=device)  # [1, D]

    # Precompute guide outputs and preserve outputs using original weights
    with torch.no_grad():
        guide_vec = _encode_concept_vector(pipe, guide_prompt, device=device)
        guide_outputs = [m(guide_vec).detach() for m in original_modules]

        preserve_outputs: Dict[str, List[torch.Tensor]] = {}
        for p in preserve_prompts:
            vec = concept_vecs[p]
            preserve_outputs[p] = [m(vec).detach() for m in original_modules]

    applied = 0

    for module_idx, old_module in enumerate(original_modules):
        w_old = old_module.weight.detach()  # [out, in]
        in_dim = w_old.shape[1]

        mat1 = lamb * w_old
        mat2 = lamb * torch.eye(in_dim, device=device, dtype=torch.float32)

        # Erase terms: force edit prompts to match guide output
        for ep in edit_prompts:
            c = concept_vecs[ep].T                # [D, 1]
            v_star = guide_outputs[module_idx].T  # [out, 1]
            mat1 = mat1 + erase_scale * (v_star @ c.T)
            mat2 = mat2 + erase_scale * (c @ c.T).float()

        # Preserve terms: keep preserve prompts stable
        for pp in preserve_prompts:
            c = concept_vecs[pp].T
            v_star = preserve_outputs[pp][module_idx].T
            mat1 = mat1 + preserve_scale * (v_star @ c.T)
            mat2 = mat2 + preserve_scale * (c @ c.T).float()

        w_new = _stable_solve_right(mat1, mat2, out_dtype=w_old.dtype)
        edited_modules[module_idx].weight = torch.nn.Parameter(w_new)
        applied += 1

    # Save
    save_format = (save_format or "delta").lower().strip()
    if save_format not in ("delta", "absolute"):
        raise ValueError("save_format must be 'delta' or 'absolute'")

    state: Dict[str, torch.Tensor] = {}
    for name, new_mod, old_mod in zip(uce_module_names, edited_modules, original_modules):
        key = name + ".weight"
        if save_format == "delta":
            state[key] = (new_mod.weight.detach() - old_mod.weight.detach()).contiguous()
        else:
            state[key] = new_mod.weight.detach().contiguous()

    out_path = os.path.join(save_dir, exp_name + ".safetensors")
    save_file(state, out_path)

    end_time = time.time()
    return {
        "saved_to": out_path,
        "modules_edited": applied,
        "edit_prompts": len(edit_prompts),
        "preserve_prompts": len(preserve_prompts),
        "erase_scale": erase_scale,
        "preserve_scale": preserve_scale,
        "lamb": lamb,
        "save_format": save_format,
        "guide_prompt": guide_prompt,
        "seconds": end_time - start_time,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="uce_sd_erase",
        description="UCE moderation patch trainer (LLM-expanded concepts, edits attn2.to_k/to_v)",
    )

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pipe_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--edit_concepts", type=str, required=True,
                        help="User concepts separated by ;  Example: 'nudity; porn; sex'")
    parser.add_argument("--concept_type", type=str, default="object", choices=["object", "art"])

    # LLM controls
    parser.add_argument("--llm_model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Light open-source LLM for expansion")
    parser.add_argument("--llm_max_new_tokens", type=int, default=700)
    parser.add_argument("--llm_temperature", type=float, default=0.7)
    parser.add_argument("--llm_top_p", type=float, default=0.9)
    parser.add_argument("--llm_phrases_limit", type=int, default=250,
                        help="Cap phrases produced by LLM before templating")
    parser.add_argument("--no_llm_expand", action="store_true",
                        help="Disable LLM expansion (use edit_concepts directly)")

    # Preserve and guide
    parser.add_argument("--preserve_concepts", type=str, default=None,
                        help="Optional preserve concepts separated by ;. If omitted, safe defaults are used.")
    parser.add_argument("--guide_prompt", type=str, default="",
                        help="Guide prompt for moderation. Default empty (unconditional).")

    # UCE parameters
    parser.add_argument("--erase_scale", type=float, default=2.0)
    parser.add_argument("--preserve_scale", type=float, default=2.0)
    parser.add_argument("--lamb", type=float, default=0.1)

    # Output
    parser.add_argument("--save_dir", type=str, default="uce_models")
    parser.add_argument("--exp_name", type=str, default="NSFW_wa")
    parser.add_argument("--save_format", type=str, default="delta", choices=["delta", "absolute"])

    args = parser.parse_args()

    device = args.device
    pipe_dtype = _pick_dtype(args.pipe_dtype)

    base_concepts = [c.strip() for c in args.edit_concepts.split(";") if c.strip()]
    if not base_concepts:
        raise ValueError("No edit concepts provided.")

    # Preserve prompts
    if args.preserve_concepts:
        preserve = [c.strip() for c in args.preserve_concepts.split(";") if c.strip()]
    else:
        preserve = ["person", "man", "woman", "face", "portrait", "human", "clothing"]

    preserve = _dedupe_preserve_order(preserve)

    # Expand concepts
    if args.no_llm_expand:
        phrases = _dedupe_preserve_order(base_concepts)
    else:
        print("\nExpanding concepts with LLM...")
        phrases = expand_concepts_with_llm(
            base_concepts=base_concepts,
            device=device,
            llm_model_id=args.llm_model_id,
            llm_max_new_tokens=args.llm_max_new_tokens,
            llm_temperature=args.llm_temperature,
            llm_top_p=args.llm_top_p,
        )
        if not phrases:
            print("LLM returned empty list, falling back to original concepts.")
            phrases = _dedupe_preserve_order(base_concepts)

    # Limit phrases before templating to control runtime
    phrases = phrases[: max(1, int(args.llm_phrases_limit))]

    edit_prompts = apply_prompt_templates(phrases, concept_type=args.concept_type)

    print("\nUCE configuration\n")
    print(f"Model: {args.model_id}")
    if args.revision:
        print(f"Revision: {args.revision}")
    print(f"Device: {device}")
    print(f"Pipeline dtype: {args.pipe_dtype}")
    print(f"Guide prompt: {args.guide_prompt!r} (empty means unconditional)")
    print(f"Edit concepts (base): {base_concepts}")
    print(f"LLM phrases: {len(phrases)}")
    print(f"Edit prompts after templating: {len(edit_prompts)}")
    print(f"Preserve prompts: {len(preserve)}")
    print(f"erase_scale={args.erase_scale}, preserve_scale={args.preserve_scale}, lamb={args.lamb}")
    print(f"save_format={args.save_format}\n")

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=pipe_dtype,
        safety_checker=None,
        vae=None,
    ).to(device)

    info = UCE_moderation(
        pipe=pipe,
        edit_prompts=edit_prompts,
        preserve_prompts=preserve,
        erase_scale=float(args.erase_scale),
        preserve_scale=float(args.preserve_scale),
        lamb=float(args.lamb),
        save_dir=args.save_dir,
        exp_name=args.exp_name,
        device=device,
        save_format=args.save_format,
        guide_prompt=args.guide_prompt,
    )

    print("\nDone\n")
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
