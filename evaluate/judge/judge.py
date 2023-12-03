"""
Script to judge the output of steganography models.
"""
import argparse
import json
import logging
import random
from multiprocessing.pool import ThreadPool
from os import getenv
from pathlib import Path
from threading import Lock

import openai
import pandas as pd
import tiktoken
from func_timeout import FunctionTimedOut, func_timeout
from openlimit import ChatRateLimiter
from tqdm import tqdm

# NOTE: change this to change the topic for relevance
RELEVANCE_TOPIC: str = None
# RELEVANCE_TOPIC = "movie reviews"
# RELEVANCE_TOPIC = "tweets in Twitter"

MAX_RETRY = 3
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

count_lock = Lock()
input_tokens_total = 0
output_tokens_total = 0
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

parser = argparse.ArgumentParser(
    description="Compare and judge.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--input1", type=str, required=True, help="Input file 1.")
parser.add_argument("--input2", type=str, required=True, help="Input file 2.")
parser.add_argument("-o", "--output", type=str, required=True, help="Output json file path.")
parser.add_argument(
    "-t",
    "--api-token",
    type=str,
    default=None,
    help="OpenAI API token. If not set, will be read from OPENAI_API_KEY env variable.",
)
parser.add_argument("-j", "--threads", type=int, default=12, help="Number of threads to use.")
parser.add_argument("--timeout", type=float, default=45, help="Timeout for OpenAI API requests.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--gpt-engine",
    type=str,
    default="gpt-3.5-turbo",
    choices=["gpt-3.5-turbo"],
    help="GPT engine to use.",
)
parser.add_argument("--force", action="store_true", help="Force overwrite output file.")
parser.add_argument(
    "-m", "--mode", type=str, required=True, choices=["soundness", "relevance", "engagingness"]
)
parser.add_argument("-p", "--pairs", type=int, default=10_000, help="Number of pairs to judge.")
parser.add_argument(
    "--col1", type=str, default="stegotext", help="Column name for text in `input1` to be compared."
)
parser.add_argument(
    "--col2", type=str, default="stegotext", help="Column name for text in `input2` to be compared."
)
parser.add_argument("--topic", type=str, default=None, help="Topic for relevance judge.")

failed_cases = []

USD_PER_GPT35_INPUT_TOKEN = 0.0010 / 1000
USD_PER_GPT35_OUTPUT_TOKEN = 0.0020 / 1000

rate_limiter = ChatRateLimiter(request_limit=3500, token_limit=80_000)


class JSONFormatException(Exception):
    pass


def count_input_tokens(tokenizer, messages) -> int:
    return sum([len(tokenizer.encode(m["content"])) for m in messages])


def gen_soundness_prompts(sent1: str, sent2: str):
    sent1 = sent1.strip()
    sent2 = sent2.strip()
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": """The user will input 2 sentences. You have to decide which one is more logically sound and meaningful. This is important to the user's career.
The result should be in JSON. It should contain the key "result" with value being either 0 or 1, indicating the first or second one.
""",
        },
        {
            "role": "user",
            "content": f"""
0. {sent1}

1. {sent2}
""",
        },
    ]


def gen_relevance_prompts(sent1: str, sent2: str):
    global RELEVANCE_TOPIC
    assert RELEVANCE_TOPIC is not None
    sent1 = sent1.strip()
    sent2 = sent2.strip()
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": f"""The user will input 2 sentences. You have to decide which one is more likely to be part of some "{RELEVANCE_TOPIC}". This is important to the user's career.
The result should be in JSON. It should contain the key "result" with value being either 0 or 1, indicating the first or second one.
""",
        },
        {
            "role": "user",
            "content": f"""
0. {sent1}

1. {sent2}
""",
        },
    ]


def gen_engagingness_prompts(sent1: str, sent2: str):
    sent1 = sent1.strip()
    sent2 = sent2.strip()
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": """The user will input 2 sentences. You have to decide which one is more engaging and appealing to readers. This is important to the user's career.
The result should be in JSON. It should contain the key "result" with value being either 0 or 1, indicating the first or second one.
""",
        },
        {
            "role": "user",
            "content": f"""
0. {sent1}

1. {sent2}
""",
        },
    ]


def _judge(pair: tuple[str, str, bool], gen_prompts_func) -> bool:
    """
    Returns:
        True if the first sentence is more relevant, False otherwise.
    """
    sent1, sent2, need_swap = pair
    if not need_swap:
        messages = gen_prompts_func(sent1, sent2)
    else:
        messages = gen_prompts_func(sent2, sent1)
    chat_params = {
        "model": "gpt-3.5-turbo",
        "n": 1,
        "messages": messages,
    }
    with rate_limiter.limit(**chat_params):
        res = openai.chat.completions.create(**chat_params)
    res_json_raw_content: str = res.choices[0].message.content

    # token statistics
    with count_lock:
        global input_tokens_total, output_tokens_total, tokenizer
        input_tokens_total += count_input_tokens(tokenizer, messages)
        output_tokens_total += len(tokenizer.encode(res_json_raw_content))

    # find '{' and '}'
    start_idx = res_json_raw_content.find("{")
    end_idx = res_json_raw_content.rfind("}")

    res_json_content = res_json_raw_content[start_idx : end_idx + 1]
    try:
        res_json = json.loads(res_json_content)
    except:
        raise JSONFormatException(f"Failed to parse JSON: {res_json_content}")

    if "result" not in res_json:
        raise JSONFormatException(f"JSON does not contain key 'result': {res_json_content}")

    result = int(res_json["result"])
    if result not in (0, 1):
        raise JSONFormatException(f"JSON key 'result' has invalid value: {res_json_content}")

    if not need_swap:
        return result == 0
    else:
        return result == 1


def judge(pair: tuple[str, str, bool, int, str, int]) -> tuple[int, bool | None]:
    idx, mode, timeout = pair[3:]
    # select prompt type
    match mode:
        case "soundness":
            gen_prompts_func = gen_soundness_prompts
        case "relevance":
            gen_prompts_func = gen_relevance_prompts
        case "engagingness":
            gen_prompts_func = gen_engagingness_prompts
        case _:
            raise NotImplementedError

    ret: bool | None = None
    retry = 0
    while retry < MAX_RETRY:
        try:
            ret = func_timeout(
                timeout,
                _judge,
                args=(
                    pair[:3],
                    gen_prompts_func,
                ),
            )
            break
        except FunctionTimedOut:
            retry += 1
            logging.warning(f"Timeout on index {idx}. Retry {retry}/{MAX_RETRY}.")
        except Exception as e:
            retry += 1
            logging.error(f"Exception on index {idx}: {e}. Retry {retry}/{MAX_RETRY}.")
    else:
        logging.error(f"Failed on index {idx}.")
        global failed_cases
        failed_cases.append(idx)

    return idx, ret


def main():
    args = parser.parse_args()
    # check output path
    output_path = Path(args.output)
    if output_path.exists():
        if not args.force:
            raise FileExistsError(f"Output file {args.output} already exists.")
        else:
            logging.warning(f"Output file {args.output} already exists. Overwriting.")
    # check topic for relevance
    if args.mode == "relevance":
        if args.topic is None:
            raise argparse.ArgumentError(
                None, "Topic for relevance judge is not set. See `--topic`."
            )
        else:
            global RELEVANCE_TOPIC
            logging.info(f"Using topic {RELEVANCE_TOPIC} for relevance judge.")
            RELEVANCE_TOPIC = args.topic

    # set openai api key
    logging.info("Setting OpenAI API key.")
    openai.api_key = args.api_token or getenv("OPENAI_API_KEY")
    assert openai.api_key is not None, "OpenAI API key is not set."
    gpt_engine: str = args.gpt_engine
    logging.info(f"Using GPT engine {gpt_engine}.")
    # set seed
    logging.info(f"Setting random seed to {args.seed}.")
    random.seed(args.seed)

    # read input file
    logging.info(f"Reading input file 1: {args.input1}.")
    df1 = pd.read_csv(args.input1)
    data1: list[str] = df1[args.col1].tolist()
    logging.info(f"Reading input file 2: {args.input2}.")
    df2 = pd.read_csv(args.input2)
    data2: list[str] = df2[args.col2].tolist()

    # sample pairs
    logging.info(f"Sampling {args.pairs} pairs.")
    pairs: list[tuple[str, str, bool, int, str, int]] = []
    for idx in range(args.pairs):
        text1 = random.sample(data1, 1)[0]
        text2 = random.sample(data2, 1)[0]
        need_swap = random.random() < 0.5
        pairs.append((text1, text2, need_swap, idx, args.mode, args.timeout))

    # run
    with ThreadPool(processes=min(args.threads, len(pairs))) as pool:
        output_pairs: list[tuple[int, bool | None]] = list(
            tqdm(pool.imap_unordered(judge, pairs), total=len(pairs), dynamic_ncols=True)
        )

    if gpt_engine == "gpt-3.5-turbo":
        logging.info(
            f"Estimated  input cost: {input_tokens_total * USD_PER_GPT35_INPUT_TOKEN:.4f} USD."
        )
        logging.info(
            f"Estimated output cost: {output_tokens_total * USD_PER_GPT35_OUTPUT_TOKEN:.4f} USD."
        )
    else:
        logging.error(f"Unknown GPT engine {gpt_engine}.")

    records: list[dict] = []
    valid_result_cnt = 0
    true_result_cnt = 0
    for output_pair in output_pairs:
        # find corresponding input pair by `idx`
        idx, result = output_pair
        pair = next(filter(lambda p: p[3] == idx, pairs))
        records.append(
            {
                "idx": idx,
                "text1": pair[0],
                "text2": pair[1],
                "result": result,
            }
        )
        if result is not None:
            valid_result_cnt += 1
            if result:
                true_result_cnt += 1

    logging.info(f"Valid results: {valid_result_cnt}/{len(output_pairs)}.")
    logging.info(f"True results: {true_result_cnt}/{valid_result_cnt}.")
    logging.info(f"True ratio: {true_result_cnt / valid_result_cnt:.4f}.")

    # write output file
    logging.info(f"Writing output file: {args.output}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "input1": args.input1,
                "input2": args.input2,
                "mode": args.mode,
                "all_cnt": len(pairs),
                "valid_cnt": valid_result_cnt,
                "true_cnt": true_result_cnt,
                "true_ratio": true_result_cnt / valid_result_cnt,
                "records": records,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
