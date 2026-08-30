def vlm(image, system, question, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0):
    """One VLM call: PIL image + question -> (raw text reply, generate seconds).

    The timer starts only AFTER the image has been preprocessed and moved to the GPU, so the
    figure is the time Qwen spends looking and answering -- not tokenisation. Stage 3's cost
    is a paper claim, so it has to be measured the same way Stage 1 measures its own.
    """
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)

    if torch.cuda.is_available():
        torch.cuda.synchronize()          # do not time work still queued from before
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()          # generate is async; wait before stopping the clock
    secs = time.time() - t0

    reply = processor.decode(out[0][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
    return reply, secs


def grab_json(raw):
    """Pull the first JSON object out of a reply, tolerating stray prose or code fences."""
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))
