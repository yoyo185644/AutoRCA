import os
import yaml
import time

def load_config(config_path="rca/api_config.yaml"):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    if isinstance(yaml_data, dict):
        for key, value in yaml_data.items():
            if value is None or value == "":
                continue
            configs[key] = value
    return configs

configs = load_config()

def _config_value(*names, default=None):
    for name in names:
        value = configs.get(name)
        if value is not None and str(value).strip() != "":
            text = str(value).strip()
            if text.startswith("="):
                text = text[1:].strip()
            return text
    return default


def _sdk_http_client():
    import httpx

    return httpx.Client(timeout=httpx.Timeout(600.0), trust_env=True)


def _model_id():
    return str(_config_value("MODEL", "MODEL_ID", default=""))


def _api_key(*extra_names):
    return str(_config_value(*extra_names, "API_KEY", default=""))


def _extract_anthropic_text(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def OpenAI_chat_completion(messages, temperature):    
    from openai import OpenAI
    client = OpenAI(
        api_key=_api_key("OPENAI_API_KEY"),
        http_client=_sdk_http_client(),
    )
    return client.chat.completions.create(
        model = _model_id(),
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def Google_chat_completion(messages, temperature):
    import google.generativeai as genai
    genai.configure(
        api_key=configs["API_KEY"]
    )
    genai.GenerationConfig(temperature=temperature)
    system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None
    messages = [item for item in messages if item["role"] != "system"]
    messages = [{"role": "model" if item["role"] == "assistant" else item["role"], "parts": item["content"]} for item in messages]
    history = messages[:-1]
    message = messages[-1]
    return genai.GenerativeModel(
        model_name=configs["MODEL"],
        system_instruction=system_instruction
        ).start_chat(
            history=history if history != [] else None
            ).send_message(message).text

def Anthropic_chat_completion(messages, temperature):
    import anthropic
    system_parts = [str(item.get("content", "")) for item in messages if item.get("role") == "system"]
    anthropic_messages = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in messages
        if item.get("role") in ("user", "assistant")
    ]
    kwargs = {
        "model": _model_id(),
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": int(_config_value("ANTHROPIC_MAX_TOKENS", "MAX_TOKENS", default=4096)),
    }
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    base_url = _config_value("ANTHROPIC_BASE_URL", "API_BASE", default=None)
    client = anthropic.Anthropic(
        api_key=_api_key("ANTHROPIC_API_KEY"),
        base_url=base_url,
        http_client=_sdk_http_client(),
    )
    return _extract_anthropic_text(client.messages.create(**kwargs).content)

# for 3-rd party API which is compatible with OpenAI API (with different 'API_BASE')
def AI_chat_completion(messages, temperature):    
    from openai import OpenAI
    client = OpenAI(
        api_key=_api_key("OPENAI_API_KEY"),
        base_url=_config_value("API_BASE", "OPENAI_BASE_URL"),
        http_client=_sdk_http_client(),
    )
    return client.chat.completions.create(
        model = _model_id(),
        messages = messages,
        temperature = temperature,
    ).choices[0].message.content

def get_chat_completion(messages, temperature=0.0):

    def send_request():
        if configs["SOURCE"] == "AI":
            return AI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "OpenAI":
            return OpenAI_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Google":
            return Google_chat_completion(messages, temperature)
        elif configs["SOURCE"] == "Anthropic":
            return Anthropic_chat_completion(messages, temperature)
        else:
            raise ValueError("Invalid SOURCE in api_config file.")
    
    for i in range(3):
        try:
            return send_request()
        except Exception as e:
            print(e)
            if '429' in str(e):
                print("Rate limit exceeded. Waiting for 1 second.")
                time.sleep(1)
                continue
            else:
                raise e
